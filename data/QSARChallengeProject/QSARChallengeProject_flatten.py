'''
Utility script to flatten the QSAR Challenge genotoxicity datasets to a single excel file.
Given that many records have a SMILES but no CAS number, we rely on the provided structures and we do not attempt to derive them from the provided identifers.

The raw data were in a pdf that was difficult to parse. Hence a small number of records was deleted.
'''

# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/QSARChallengeProject_flatten.log')

from pathlib import Path
import pandas as pd
import numpy as np
import json
import re

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

# read the raw data
inpfs = [r'data\QSARChallengeProject\raw\first_challenge\ClassA183.xlsx',
            r'data\QSARChallengeProject\raw\first_challenge\ClassA236.xlsx',
            r'data\QSARChallengeProject\raw\first_challenge\ClassA253.xlsx',
            r'data\QSARChallengeProject\raw\second_challenge\ClassA_80chemicals_2ndProject.xlsx'
]
datasets = []
for inpf in inpfs:
    datasets.append(pd.read_excel(inpf))
datasets = pd.concat(datasets, axis='index', sort=False,  ignore_index=True).drop_duplicates()
datasets = datasets.rename({'CAS#': 'CAS number', 'SMILES': 'smiles', 'Chemical name': 'substance name'}, axis='columns')[['CAS number', 'smiles', 'substance name']]
datasets['CAS number'] = datasets['CAS number'].str.replace(r'[\n\r\s]+', '', regex=True).fillna('-').apply(lambda cas: cas if re.match(r'\d{2,7}-\d{2}-\d', cas) else 'not available')
datasets['smiles'] = datasets['smiles'].str.replace(r'[\n\r\s]+','', regex=True)

# drop records without a structure
msk = datasets['smiles'].notnull()
datasets = datasets[msk]

# cast the genotoxicity data to the expected, flat format
tox_data = []

# assay: bacterial reverse mutation assay, endpoint: in vitro gene mutation study in bacteria
for idx, datapoint in datasets.iterrows():
    entry = dict()
    entry['CAS number'] = datapoint['CAS number']
    entry['substance name'] = datapoint['substance name']
    entry['smiles'] = datapoint['smiles']
    entry['source'] = 'QSAR Challenge project'
    entry['raw input file'] = str(inpf)
    entry['in vitro/in vivo'] = 'in vitro'
    entry['endpoint'] = 'in vitro gene mutation study in bacteria'
    entry['assay'] = 'bacterial reverse mutation assay'
    entry['cell line/species'] = 'unknown'
    entry['metabolic activation'] = 'unknown'
    entry['genotoxicity mode of action'] = 'unknown'
    entry['gene'] = 'unknown'
    entry['notes'] = None
    entry['reference'] = 'https://doi.org/10.1080/1062936X.2023.2284902'
    entry['genotoxicity'] = 'positive'
    entry['additional source data'] = json.dumps(dict())
    tox_data.append(entry)

# put everything together
tox_data = pd.DataFrame(tox_data)
tox_data = tox_data.reset_index().rename({'index': 'record ID'}, axis='columns')
tox_data['record ID'] = tox_data['source'] + ' ' + tox_data['record ID'].astype(str)

# reorder the columns
tox_data = tox_data[['record ID', 'source', 'raw input file', 'CAS number', 'substance name', 'smiles', 'in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation', 'genotoxicity mode of action', 'gene', 'notes', 'genotoxicity', 'reference', 'additional source data']]

# export the flat dataset
outf = Path('data/QSARChallengeProject/tabular/QSARChallengeProject.xlsx')
log.info(f'exporting {outf}')
tox_data.to_excel(outf, index=False)

