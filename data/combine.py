'''
Utility script to combine the flatten datasets to a single sdf file.

Some records may be eliminated if the structure cannot be converted to RDKit mol, or if it cannot be
standardised.

The user can decide the level of the aggregation, e.g. by
- in vitro/in vivo
- endpoint
- assay
- cell line/species
- metabolic activation
- genotoxicity mode of action
- gene
and the script will use the canonical smiles to aggregate a summary call. If there are more than one genotoxicity calls
after aggregation, the call will be set to ambiguous. Hence, an ambiguous final genotoxicity call may be because it was
ambgiguous in the source data. The different tasks can use different aggregation levels.

The SDF file will contain one mol block per canonical smiles and a record ID will be set using a running index. Each
mol block will contain the following fields:
- genotoxicity: the genotoxicity call, including its full lineage

The mol block name will be set to the dataset record ID.

This script can be called repeatedly to examine how different aggregations affect the modelling performance in multitask modelling
'''

# setup logging
import logger
log = logger.get_logger(__name__)

from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import json
from functools import reduce

from cheminformatics.rdkit_toolkit import (convert_smiles_to_mol, standardise_mol, remove_stereo,
                                           normalise_mol, remove_fragments, check_mol, apply_tautomerisation_transformations)

from rdkit import Chem

# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr',False)
# maximum number of columns
pd.set_option("display.max_columns",50)
# maximum number of rows
pd.set_option("display.max_rows",500)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True




def process_smiles(smiles: str,
                   ) -> tuple[Union[str, None], Union[Chem.Mol, None], str]:
    '''
     Utility function to process an input smiles. It checks that the smiles is suitable for modelling and it
     produces a standardised smiles and standardised RDKit mol. The latter can be used for modelling. The function also
     returns a dictionary with the processing details.

     This utility function is used for both preparing the training/test set but also for using the developed models.

     In case the smiles cannot be processed or if it is not suitable for modelling the function will return a tuple with None, None, and a string
     with the errors from the first operation that failed to produce a structure.

    :param smiles: input smiles string
    :return: tuple of standardised smiles, standardised RDKit mol, and processing details
    '''

    processing_details = dict()

    # convert the smiles to RDKit mol, sanitisation carries out a first set of checks
    rdkit_mol, error_warnings = convert_smiles_to_mol(smiles, sanitize=True)
    msgs = [msg.strip() for msg in re.split(r'\s*\[[\d:]+\]\s*', error_warnings) if msg] if error_warnings else []
    msgs = ', '.join(sorted(msgs))
    if rdkit_mol is not None:
        processing_details['warnings (smiles to RDKit mol)'] = msgs if msgs else None
    else:
        return None, None, msgs

    # disconnect metals
    tfs = 'disconnect_alkali_metals\t[Li,Na,K,Rb:1]-[A:2]>>([*+:1].[*-:2])'
    rdkit_mol_std, ops_applied = normalise_mol(rdkit_mol, tfs)
    if rdkit_mol_std is not None:
        msgs = ', '.join(sorted(ops_applied))
        processing_details['operations applied (normalisation)'] = msgs if msgs else None
    else:
        return None, None, 'Failed to disconnect alkali metals'

    # standardise the structure (first cleanup)
    STANDARDISER_OPS = ['cleanup']
    rdkit_mol_std, error_warnings = standardise_mol(rdkit_mol_std, STANDARDISER_OPS)
    if rdkit_mol_std is not None:
        msgs = [msg_strip for msg in re.split(r'\s*\[[\d:]+\]\s*', error_warnings) if msg and
                (msg_strip:=msg.strip()) not in ['Initializing MetalDisconnector', 'Initializing Normalizer', 'Running MetalDisconnector', 'Running Normalizer', 'Running LargestFragmentChooser', 'Running Uncharger']] if error_warnings else []
        msgs = ', '.join(sorted(msgs))
        processing_details['warnings (standardisation, cleanup)'] = msgs if msgs else None
    else:
        return None, None, error_warnings

    # remove toxicologically insignificant fragments, if all fragments are removed None is returned
    rdkit_mol_std, frags_removed = remove_fragments(rdkit_mol_std)
    msgs = ', '.join(sorted(frags_removed))
    processing_details['removed fragments'] = msgs if msgs else None
    if rdkit_mol_std is None:
        return None, None, 'All fragments were removed as toxicologically insignificant'

    # tautomerise, if it fails the original structure is returned
    rdkit_mol_std, msgs, error_warnings = apply_tautomerisation_transformations(rdkit_mol_std)
    processing_details['tautomerisation'] = msgs if msgs else None

    # remove stereo
    rdkit_mol_std = remove_stereo(rdkit_mol_std, stereo_types=None)

    # standardise the structures (other operations)
    STANDARDISER_OPS = ['uncharge']  # ['cleanup', 'addHs']
    rdkit_mol_std, error_warnings = standardise_mol(rdkit_mol_std, STANDARDISER_OPS)
    if rdkit_mol_std is not None:
        msgs = [msg_strip for msg in re.split(r'\s*\[[\d:]+\]\s*', error_warnings) if msg and
                (msg_strip := msg.strip()) not in ['Initializing MetalDisconnector', 'Initializing Normalizer',
                                                   'Running MetalDisconnector', 'Running Normalizer',
                                                   'Running LargestFragmentChooser',
                                                   'Running Uncharger']] if error_warnings else []
        msgs = ', '.join(sorted(msgs))
        processing_details['warnings (standardisation, other operations)'] = msgs if msgs else None
    else:
        return None, None, error_warnings

    # check if the structure is suitable for modelling
    CHECK_OPS = {'allowed_atoms': ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'B', 'Si', 'I', 'H'],
                 'min_num_carbon_atoms': 1,
                 'min_num_bonds': 1,
                 'max_num_fragments': 1,
                 'allowed_bonds': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
                 'molecular_weight': {'min': 0, 'max': 1000},
                 'max_number_rings': 5,
                 'allowed_hybridisations': ['UNSPECIFIED', 'SP2', 'SP3', 'SP'],
                 'allowed_total_charge': [0],
                 }
    check_outcome = check_mol(rdkit_mol_std, ops=CHECK_OPS)
    if check_outcome:
        return Chem.MolToSmiles(rdkit_mol_std), rdkit_mol_std, processing_details
    else:
        return None, None, 'Structure checker rejected the structure'





def create_sdf(flat_datasets: list,
               task_specifications: list[dict],
               outp_sdf: Path, outp_tab: Path) -> list[str]:
    '''
    Read in the provided flatten datasets, preprocess the structures and produce an sdf file suitable for creating
     a PyTorch Geometric dataset.
    :param flat_datasets: list of paths to the flattened datasets
    :param task_specifications: list of dictionaries to define which records to keep and how to aggregate them into tasks
    :param outp_sdf: path to the output sdf file
    :param outp_tab: path to the output tabular (excel) file
    :return: list of task names
    '''
    dataset = []
    for flat_dataset in flat_datasets:
        log.info(f'reading in {flat_dataset}')
        dataset.append(pd.read_excel(flat_dataset))
    dataset = pd.concat(dataset, ignore_index=True, axis='index', sort=False)
    # set the "record ID" and rename the source "record ID" to "source record ID"
    dataset = (dataset
               .rename({'record ID': 'source record ID'}, axis='columns')
               .reset_index(drop=True).reset_index().rename({'index': 'record ID'}, axis=1))

    aggregated_datasets = []
    for i_task, task_specification in enumerate(task_specifications):

        filters = task_specification['filters']
        task_aggregation_cols = task_specification['task aggregation columns']
        msk = []
        for filter_col, filter_values in filters.items():
            msk.append(dataset[filter_col].isin(filter_values))
        msk = reduce(lambda x, y: x & y, msk)
        log.info(f'task {i_task}: {msk.sum()} records kept after applying the filters')

        # preprocess the structures, some may be removed by the checker and due to the applied operations
        smiles_list = dataset.loc[msk, 'smiles'].drop_duplicates().to_list()
        structures = []
        for smiles in tqdm(smiles_list):
            smiles_std, rdkit_mol_std, processing_details = process_smiles(smiles)
            if smiles_std is not None:
                structures.append({'smiles': smiles, 'smiles_std': smiles_std, 'rdkit_mol_std': rdkit_mol_std,
                                   'processing details': processing_details})
        log.info(f'{len(structures)} out of {len(smiles_list)} structures remain after preprocessing')

        # merge structures into the dataset
        structures = pd.DataFrame(structures)
        aggregated_dataset = dataset.loc[msk].merge(structures, on='smiles', how='inner')

        # aggregate all genotoxicity calls for the same source, smiles_std and task
        # set to positive if at least one positive, ambiguous if at least one ambiguous, otherwise negative
        def f(group):
            genotoxicity = 'positive' if 'positive' in group['genotoxicity'].to_list() else 'ambiguous' if 'ambiguous' in group['genotoxicity'].to_list() else 'negative'
            source_details = group.groupby('genotoxicity')['source record ID',].agg(lambda x: x.drop_duplicates().to_list()).reset_index().to_dict(orient='records')
            return pd.Series({'genotoxicity': genotoxicity, 'source details': source_details})
        aggregated_dataset = aggregated_dataset.groupby(['smiles_std', 'source'] + task_aggregation_cols)[['genotoxicity', 'CAS number', 'source record ID']].apply(f).reset_index()

        # aggregate all genotoxicity calls for the same smiles_std and task
        # set to positive if all are positive, negative if all are negative, otherwise ambiguous
        def f(group):
            if {'positive'} == set(group['genotoxicity']):
                genotoxicity = 'positive'
            elif {'negative'} == set(group['genotoxicity']):
                genotoxicity = 'negative'
            else:
                genotoxicity = 'ambiguous'
            genotoxicity_details = group[['genotoxicity', 'source', 'source details']].rename({'genotoxicity': 'genotoxicity (source)'}, axis='columns').to_dict(orient='records')
            return pd.Series({'genotoxicity': genotoxicity, 'genotoxicity details': genotoxicity_details})
        aggregated_dataset = aggregated_dataset.groupby(['smiles_std'] + task_aggregation_cols)[['genotoxicity', 'source', 'source details']].apply(f).reset_index()

        # # aggregate, set to positive if at least one positive, ambiguous if at least one ambiguous, otherwise negative
        # aggregated_dataset = (aggregated_dataset.groupby(['smiles_std'] + task_aggregation_cols)[['genotoxicity', 'CAS number', 'source record ID', 'smiles']]
        #        .agg({
        #                 # 'genotoxicity': lambda vals: x[0] if len(x:=vals.dropna().drop_duplicates().to_list())==1 else 'ambiguous' if len(x)>1 else 'not available',
        #                 'genotoxicity':  lambda vals: 'positive' if 'positive' in vals.to_list() else 'ambiguous' if 'ambiguous' in vals.to_list() else 'negative',
        #                 'CAS number': lambda rns: ', '.join(sorted(rns.dropna().drop_duplicates().to_list())),
        #                 'source record ID': lambda srids: ', '.join(sorted(srids.dropna().drop_duplicates().to_list())),
        #                 'smiles':  lambda smis: ', '.join(sorted(smis.dropna().drop_duplicates().to_list()))
        #           })
        #        .reset_index()
        #        )
        # # .. genotoxicity outcomes per database to enable concordance analysis
        # aggregated_dataset = aggregated_dataset.merge(tmp, on=['smiles_std'] + task_aggregation_cols, how='left')
        aggregated_dataset['task aggregation'] = aggregated_dataset[task_aggregation_cols].apply(lambda row: ', '.join(row.index), axis='columns')
        aggregated_dataset['task'] = aggregated_dataset[task_aggregation_cols].apply(lambda row: ', '.join(row.to_list()), axis='columns')
        aggregated_dataset = aggregated_dataset.drop(task_aggregation_cols, axis='columns')

        aggregated_datasets.append(aggregated_dataset)

    aggregated_datasets = pd.concat(aggregated_datasets, axis='index', ignore_index=True, sort=False)

    # list of tasks
    tasks = aggregated_datasets['task'].drop_duplicates().to_list()

    # create the tabular file
    log.info(f'writing genotoxicity dataset to {outp_tab}')
    aggregated_datasets['genotoxicity details'] = aggregated_datasets['genotoxicity details'].apply(json.dumps)
    aggregated_datasets = aggregated_datasets.reset_index(drop=True).reset_index(drop=False).rename({'index': 'datapoint ID'}, axis=1)
    aggregated_datasets.to_excel(outp_tab, index=False)

    # create the sdf file
    log.info(f'writing genotoxicity dataset to {outp_sdf}')
    aggregated_datasets['genotoxicity details'] = aggregated_datasets['genotoxicity details'].apply(json.loads)
    aggregated_datasets = aggregated_datasets.groupby('smiles_std')[[ 'datapoint ID', 'smiles_std', 'task aggregation', 'task',  'genotoxicity', 'genotoxicity details']].apply(lambda x: x.to_json(orient='records')).rename('genotoxicity').reset_index()
    with Chem.SDWriter(outp_sdf) as sdf_writer:
        for idx, row in tqdm(aggregated_datasets.iterrows()):
            mol = Chem.MolFromSmiles(row['smiles_std'])
            mol.SetProp('genotoxicity', row['genotoxicity'])
            sdf_writer.write(mol)

    return tasks

