'''
Utility script to flatten the Leadscope genotoxicity dataset to a single excel file.

This script is not changing the number of rows in the dataset, i.e. if the original dataset contains multiple
rows for the same CAS number or structure they are retained. The structure is not standardised or otherwise modified.

The purpose of this script is to normalise the dataset so that it contains the following columns:
- record ID
- source = Hansen 2009
- raw input file
- CAS number
- smiles
- in vitro/in vivo
- endpoint
- assay
- cell line/species
- metabolic activation
- genotoxicity mode of action
- gene
- notes
- genotoxicity
- reference

Tne idea is that each source will generate one mor more such files that can be combined into a single dataset and
aggregated at different levels (e.g. endpoint, assay, molecule) for modelling.
'''

# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/Leadscope_flatten.log')

import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
import json
from collections import Counter
from io import StringIO, BytesIO
from tqdm import tqdm

from cheminformatics.rdkit_toolkit import Rdkit_operation

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
with open(r'data/Leadscope/raw/2024_05_20/datasets.yaml', 'r') as inf:
    datasets = yaml.safe_load(inf)

# skip the datasets that are not to be processed
datasets = {k: v for k, v in datasets.items() if v['process']}

# read in the Leadscope datasets and add the mols to the datasets dictionary
for dataset in datasets:
    log.info(f'reading dataset {dataset}')
    inp_sdf = Path(datasets[dataset]['raw input file'])
    mols = []
    n_fail = 0
    with open(inp_sdf, 'rt', encoding='cp1252') as inf:
        with Rdkit_operation() as sio:
            with Chem.ForwardSDMolSupplier(BytesIO(inf.read().encode('utf-8'))) as suppl:
                for i_mol, mol in enumerate(suppl):
                    if mol is not None:
                         mols.append(mol)
                    else:
                        log.info(f'could not read molecule {i_mol}')
                        n_fail += 1
    log.info(f'{dataset}: read successfully {len(mols)} structures, failed to read {n_fail} structures')
    datasets[dataset]['mols'] = mols

# produce the flat dataset
tox_data = []
for dataset in datasets:
    log.info(f'flattening dataset {dataset}')
    n_fail = 0
    for i_mol, mol in enumerate(datasets[dataset]['mols']):
        # convert to isomeric, non-canonical smiles,  checking and standardisation of molecular structures
        # is done later for all sources
        with Rdkit_operation() as sio:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)  # canonical smiles
            if smiles is None:
                log.info(f'could not convert molecule {i_mol} to smiles')
                n_fail += 1
                continue
        # fetch the needed properties from the original mol
        substance_id = mol.GetPropsAsDict().get('Substance ID', None)
        cas_number = mol.GetPropsAsDict().get('CAS.Number', None)
        genotoxicity = mol.GetPropsAsDict().get(datasets[dataset]['test outcome field'], None)
        if genotoxicity is None:
            log.warning(f'no genotoxicity data for molecule {i_mol}')
            n_fail += 1
            continue
        else:
            genotoxicity = 'positive' if genotoxicity == 1 else 'negative'
        # create entry
        entry = {'source': 'Leadscope',
                 'raw input file': datasets[dataset]['raw input file'],
                 'CAS number': cas_number,
                 'substance name': None,
                 'smiles': smiles,
                 'in vitro/in vivo': datasets[dataset]['in vitro_in vivo'],
                 'endpoint': datasets[dataset]['endpoint'],
                 'assay': datasets[dataset]['assay'],
                 'cell line/species': datasets[dataset]['cell line_species'],
                 'metabolic activation': datasets[dataset]['metabolic activation'],
                 'genotoxicity mode of action': datasets[dataset]['genotoxicity mode of action'],
                 'gene': datasets[dataset]['gene'],
                 'notes': datasets[dataset]['notes'],
                 'genotoxicity': genotoxicity,
                 'reference': datasets[dataset]['reference'],
                 'additional source data': json.dumps({'Leadscope substance ID': substance_id}),}
        tox_data.append(entry)

tox_data = pd.DataFrame(tox_data).reset_index().rename({'index': 'record ID'}, axis='columns')
tox_data['record ID'] = tox_data['source'] + ' ' + tox_data['record ID'].astype(str)

# reorder the columns
tox_data = tox_data[['record ID', 'source', 'raw input file', 'CAS number', 'substance name', 'smiles', 'in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation', 'genotoxicity mode of action', 'gene', 'notes', 'genotoxicity', 'reference', 'additional source data']]

# export the flat dataset
outf = Path('data/Leadscope/tabular/Leadscope_genotoxicity.xlsx')
log.info(f'exporting {outf}')
tox_data.to_excel(outf, index=False)


