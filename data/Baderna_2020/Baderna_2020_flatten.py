'''
Utility script to flatten the Baderna 2020 in vitro micronucleus dataset to a single excel file.

This script is not changing the number of rows in the dataset, i.e. if the original dataset contains multiple
rows for the same CAS number or structure they are retained. The structure is not standardised or otherwise modified.

The purpose of this script is to normalise the dataset so that it contains the following columns:
- record ID
- source
- raw input file
- CAS number
- substance name
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
log = logger.setup_applevel_logger(file_name ='logs/Baderna_2020_flatten.log')

from pathlib import Path
import pandas as pd
import json

from cheminformatics.DSStox_structure_retrieval import retrieve_structure

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


# read the raw Baderna 2020 mutagenicity dataset
inpf = Path(r'data\Baderna_2020\raw\1-s2.0-S0304389419315924-mmc2.xlsx')
datasets = pd.read_excel(inpf, header=0,  usecols=['CAS',  'MN activity'], sheet_name='DATASET')
datasets['MN activity'] = datasets['MN activity'].astype(str)
datasets.columns = datasets.columns.str.strip()

# rename existing and add missing columns
datasets = datasets.rename({'CAS': 'CAS number'}, axis='columns')
datasets = datasets.reset_index(drop=True)

# extract the CAS numbers
tmp = datasets['CAS number'].str.extractall(r'(\d{2,7}-\d{2}-\d)').reset_index()
tmp = tmp.dropna().rename({'level_0': 'index', 'match': 'CAS number index', 0: 'CAS number'}, axis='columns')
# ..convert the IUPAC names, CAS names and CAS numbers to DSSTox structures
identifiers = tmp['CAS number'].to_list()
dsstox_data = retrieve_structure(identifiers)
tmp = tmp.merge(dsstox_data, left_on='CAS number', right_on='identifier', how='inner')
tmp = tmp[['index', 'CAS number', 'smiles', 'preferredName']]
tmp = tmp.dropna().drop_duplicates().rename({'preferredName': 'substance name'}, axis='columns')
datasets = datasets.drop('CAS number', axis='columns').merge(tmp, left_index=True, right_on='index', how='inner')


# cast the genotoxicity data to the expected, flat format
tox_data = []

# assay: in vitro mammalian cell micronucleus test, endpoint: in vitro micronucleus study
msk_keep = datasets['MN activity'].isin(['1', '0'])
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    entry = dict()
    entry['CAS number'] = datapoint['CAS number']
    entry['substance name'] = datapoint['substance name']
    entry['smiles'] = datapoint['smiles']
    entry['source'] = 'Baderna 2020 dataset'
    entry['raw input file'] = str(inpf)
    entry['in vitro/in vivo'] = 'in vitro'
    entry['endpoint'] = 'in vitro micronucleus study'
    entry['assay'] = 'in vitro mammalian cell micronucleus test'
    entry['cell line/species'] = 'unknown'
    entry['metabolic activation'] = 'unknown'
    entry['genotoxicity mode of action'] = 'unknown'
    entry['gene'] = 'unknown'
    entry['notes'] = None
    entry['reference'] = 'https://doi.org/10.1016/j.jhazmat.2019.121638'
    outcome = datapoint['MN activity']
    entry['genotoxicity'] = 'positive' if ('1' == outcome) else 'negative' if ('0' == outcome) else 'ambiguous'
    entry['additional source data'] = json.dumps(dict())
    tox_data.append(entry)

# put everything together
tox_data = pd.DataFrame(tox_data)
tox_data = tox_data.reset_index().rename({'index': 'record ID'}, axis='columns')
tox_data['record ID'] = tox_data['source'] + ' ' + tox_data['record ID'].astype(str)

# reorder the columns
tox_data = tox_data[['record ID', 'source', 'raw input file', 'CAS number', 'substance name', 'smiles', 'in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation', 'genotoxicity mode of action', 'gene', 'notes', 'genotoxicity', 'reference', 'additional source data']]

# export the flat dataset
outf = Path('data/Baderna_2020/tabular/Baderna_2020_genotoxicity.xlsx')
log.info(f'exporting {outf}')
tox_data.to_excel(outf, index=False)

