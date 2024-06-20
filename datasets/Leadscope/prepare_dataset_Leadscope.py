'''
Utility script to convert the Leadscope genotoxicity datasets to an
sdf file. The sdf file is then used to create a PyTorch Geometric dataset. The molecule standardisation and feature
creation is not performed here, but in the PyTorch Geometric dataset creation script. This is to allow flexibility,
e.g. to use different feature sets, or to use different standardisation methods when testing predictive performance.

In order to include a datapoint in the sdf file for modelling it is necessary to be able to read the structure in rdKit,
standardise with the default cleanup operation and calculate a non-empty canonical smiles.

The xlsx output file contains the assay results in a tabular format.

The sdf output file contains one mol block for each unique canonical smiles. The assay results are stored as a json dump.

A script like this will be created for each dataset that is used in the project. This script can be seen as a standalone,
run-once utility script.
'''

# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/prepare_dataset_Hansen_2009_sdf.log')

import yaml
from pathlib import Path
import pandas as pd
from rdkit import Chem
import json
from collections import Counter
from io import StringIO, BytesIO
from tqdm import tqdm

from cheminformatics.rdkit_toolkit import convert_smiles_to_mol, Rdkit_operation
from rdkit.Chem.MolStandardize import rdMolStandardize

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


# read in the Leadscope dataset yaml file and create the datasets dictionary
with open(r'datasets/Leadscope/raw/2024_05_20/datasets.yaml', 'r') as inf:
    datasets = yaml.safe_load(inf)


# read in the Leadscope datasets and add the mols to the datasets dictionary
for dataset in datasets:
    log.info(f'reading dataset {dataset}')
    dataset_info = datasets[dataset]
    inp_sdf = Path(dataset_info['sdf'])
    mols = []
    with open(inp_sdf, 'rt', encoding='cp1252') as inf:
        with Rdkit_operation() as sio:
            with Chem.ForwardSDMolSupplier(BytesIO(inf.read().encode('utf-8'))) as suppl:
                for i_mol, mol in enumerate(suppl):
                    if mol is not None:
                        mol = rdMolStandardize.Cleanup(mol)  # default rdKit cleanup
                        mols.append(mol)
                    else:
                        log.info(f'could not read molecule {i_mol} from {inp_sdf}')
    log.info('read ' + str(len(mols)) + ' molecules from ' + str(inp_sdf))
    datasets[dataset]['mols'] = mols



# first pass through the datasets to examine if
# -- the needed fields are present
# -- the structures are valid
# -- there are no conflicting results for the same assay or endpoint
assay_results = []
datapoint_ID = 0
for dataset in datasets:
    dataset_info = datasets[dataset]
    log.info('processing dataset ' + dataset)
    mols = dataset_info['mols']
    for index, mol in tqdm(enumerate(mols)):
        assay_result = {'datapoint ID': datapoint_ID}
        assay_result['assay'] = dataset
        for k, v in dataset_info.items():
            if k not in ['mols']:
                assay_result[k] = v
        assay_result['parsing notes'] = []

        # standardise and compute the InChi and InChiKey
        with Rdkit_operation() as sio:
            smiles = Chem.MolToSmiles(mol).strip() # canonical smiles
            error_warning = sio.getvalue()
        if error_warning:
            assay_result['parsing notes'].append('parsing molecules in rdKit: ' + error_warning)
        if pd.isnull(smiles) or len(smiles) == 0:
            smiles = None
            assay_result['parsing notes'].append('empty smiles')
        assay_result['smiles (canonical)'] = smiles

        # fetch the needed properties from the original mol
        mol_name = mol.GetPropsAsDict().get('Substance ID', '-')
        assay_result['Substance ID'] = mol_name
        cas_number = mol.GetPropsAsDict().get('CAS.Number', '-')
        assay_result['CAS number'] = cas_number

        # genotoxicity outcome
        test_outcome_field = dataset_info['test outcome field']
        genotoxicity = mol.GetPropsAsDict().get(test_outcome_field)
        assay_result['genotoxicity'] = genotoxicity
        if genotoxicity is None:
            log.warning(f'no genotoxicity data for {mol_name}')
            assay_result['parsing notes'].append('no genotoxicity outcome')

        # append the datapoint increment counter
        assay_result['parsing notes'] = '; '.join(assay_result['parsing notes'])
        assay_results.append(assay_result)
        datapoint_ID += 1
assay_results = pd.DataFrame(assay_results).reset_index(drop=True)
assay_results['parsing notes'] = assay_results['parsing notes'].replace('', None)

# mark the structures with conflicting genotoxicity outcomes for the same assay and endpoint
tmp = (assay_results.groupby(['smiles (canonical)', 'assay'])['genotoxicity'].nunique()>1).rename('conflicting outcomes (assay)').reset_index()
assay_results = assay_results.merge(tmp, on=['smiles (canonical)', 'assay'], how='left')
log.info(f'assay/structure combinations with multiple calls: {assay_results["conflicting outcomes (assay)"].sum()}')
tmp = (assay_results.groupby(['smiles (canonical)', 'endpoint'])['genotoxicity'].nunique()>1).rename('conflicting outcomes (endpoint)').reset_index()
assay_results = assay_results.merge(tmp, on=['smiles (canonical)', 'endpoint'], how='left')
log.info(f'endpoint/structure combinations with multiple calls: {assay_results["conflicting outcomes (endpoint)"].sum()}')
# set the molecule ID
tmp = assay_results['smiles (canonical)'].dropna().drop_duplicates().reset_index().rename({'index': 'molecule ID'}, axis='columns').fillna('-').astype(str)
assay_results = assay_results.merge(tmp, on='smiles (canonical)', how='left')


# store the tabular data
outf = Path(r'datasets\Leadscope\processed\tabular')
outf.mkdir(parents=True, exist_ok=True)
log.info('stored assay results in ' + str(outf / 'Leadscope_genotoxicity.xlsx'))
with pd.ExcelWriter(outf / 'Leadscope_genotoxicity.xlsx', engine='openpyxl') as writer:
    # detailed
    assay_results.to_excel(writer, sheet_name='genotoxicity detailed')
    # summary (assay level), this eliminates records with null smiles (canonical)
    tmp_cas = assay_results.groupby('smiles (canonical)')['CAS number'].unique().rename('CAS number').apply(lambda x: '; '.join(sorted(x))).reset_index()
    tmp = assay_results.copy()
    tmp['genotoxicity'] = assay_results['genotoxicity'].replace({1.0: 'positive', 0.0: 'negative'})
    tmp = tmp[['molecule ID', 'smiles (canonical)', 'endpoint', 'assay', 'genotoxicity']].merge(tmp_cas, on='smiles (canonical)', how='left')
    tmp = tmp.groupby(['molecule ID', 'smiles (canonical)', 'CAS number', 'endpoint', 'assay'], dropna=False)['genotoxicity'].agg(lambda x: x.dropna().drop_duplicates().to_list()).apply(lambda x: x[0] if len(x)==1 else 'ambiguous' if len(x)>1 else 'not available').reset_index()
    tmp = tmp.pivot(index=['molecule ID', 'smiles (canonical)', 'CAS number'], columns=['endpoint', 'assay'], values='genotoxicity').fillna('not available').reset_index()
    tmp = tmp.loc[tmp['smiles (canonical)'].notnull()]
    tmp.to_excel(writer, sheet_name='genotoxicity summary (assay)')
    assay_results_assay = tmp.drop(('CAS number',''), axis='columns')
    cols = [col for col in assay_results_assay.columns if col != ('molecule ID', '') and  col != ('smiles (canonical)', '') ]
    assay_results_assay = assay_results_assay.melt(id_vars=[('molecule ID', ''), ('smiles (canonical)','')], value_vars=cols, value_name='genotoxicity').rename({('smiles (canonical)',''): 'smiles (canonical)', ('molecule ID', ''): 'molecule ID'}, axis='columns').rename({('smiles (canonical)',''):'smiles (canonical)'}, axis='columns')

    # summary (endpoint level), this eliminates records with null smiles (canonical)
    tmp_cas = assay_results.groupby('smiles (canonical)')['CAS number'].unique().rename('CAS number').apply(lambda x: '; '.join(sorted(x))).reset_index()
    tmp = assay_results.copy()
    tmp['genotoxicity'] = assay_results['genotoxicity'].replace({1.0: 'positive', 0.0: 'negative'})
    tmp = tmp[['molecule ID', 'smiles (canonical)', 'endpoint', 'genotoxicity']].merge(tmp_cas, on='smiles (canonical)', how='left')
    tmp = tmp.groupby(['molecule ID', 'smiles (canonical)', 'CAS number', 'endpoint'], dropna=False)['genotoxicity'].agg(lambda x: x.dropna().drop_duplicates().to_list()).apply(lambda x: x[0] if len(x)==1 else 'ambiguous' if len(x)>1 else 'not available').reset_index()
    tmp = tmp.pivot(index=['molecule ID', 'smiles (canonical)', 'CAS number'], columns='endpoint', values='genotoxicity').fillna('not available').reset_index()
    tmp = tmp.loc[tmp['smiles (canonical)'].notnull()]
    tmp.to_excel(writer, sheet_name='genotoxicity summary (endpoint)')
    assay_results_endpoint = tmp.drop('CAS number', axis='columns').melt(id_vars=['molecule ID', 'smiles (canonical)'], value_name='genotoxicity')


# create the sdf file to be used for modelling (assay level)
unique_smiles = assay_results['smiles (canonical)'].dropna().unique()
outf = Path(r'datasets\Leadscope\processed\sdf')
outf.mkdir(parents=True, exist_ok=True)
log.info('writing molecules to ' + str(outf / 'Leadscope_genotoxicity.sdf'))
with Chem.SDWriter(outf / 'Leadscope_genotoxicity.sdf') as sdf_writer:
    for smiles in tqdm(unique_smiles):
        genotoxicity_outcome_assay = assay_results_assay.loc[assay_results_assay['smiles (canonical)']==smiles].drop('smiles (canonical)', axis='columns').to_json(orient='records')
        genotoxicity_outcome_endpoint = assay_results_endpoint.loc[assay_results_endpoint['smiles (canonical)'] == smiles].drop('smiles (canonical)', axis='columns').to_json(orient='records')
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp('genotoxicity_assay_level', genotoxicity_outcome_assay)
        mol.SetProp('genotoxicity_endpoint_level', genotoxicity_outcome_endpoint)
        sdf_writer.write(mol)



