'''
Utility script to flatten the Hansen genotoxicity dataset to a single excel file.

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
log = logger.setup_applevel_logger(file_name ='logs/ECVAM_AMES_negative_flatten.log')

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


# read the raw ECVAM AMES negative mutagenicity dataset
inpf = Path(r'data\ECVAM_AmesNegative\raw\1-s2.0-S1383571820300693-mmc1.xls')
datasets = pd.read_excel(inpf, skiprows=[0],  header=0,  usecols=['CAS No.', 'AMES Overall',	'in vitro MCGM Overall', 'in vitro MN Overall', 'in vitro CA  Overall'],
                         sheet_name='Database')

# rename existing and add missing columns
datasets = datasets.rename({'CAS No.': 'CAS number',}, axis='columns')
datasets = datasets.reset_index(drop=True)




# extract the CAS numbers
tmp = datasets['CAS number'].str.extractall(r'(\d{2,7}-\d{2}-\d)').reset_index()
tmp = tmp.dropna().rename({'level_0': 'index', 'match': 'CAS number index',0: 'CAS number'}, axis='columns')
# ..convert the CAS numbers to DSSTox structures
identifiers = tmp['CAS number'].to_list()
dsstox_data = retrieve_structure(identifiers)
tmp = tmp.merge(dsstox_data, left_on='CAS number', right_on='identifier', how='inner')
tmp = tmp[['index', 'CAS number', 'smiles', 'preferredName']]
tmp = tmp.dropna().drop_duplicates().rename({'preferredName': 'substance name'}, axis='columns')
datasets = datasets.drop('CAS number', axis='columns').merge(tmp, left_index=True, right_on='index', how='inner')


# cast the genotoxicity data to the expected, flat format
tox_data = []

# assay: bacterial reverse mutation assay, endpoint: in vitro gene mutation study in bacteria
msk_keep = datasets['AMES Overall'].str.contains('\[[+-E]\]', na=False)
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    entry = dict()
    entry['CAS number'] = datapoint['CAS number']
    entry['substance name'] = datapoint['substance name']
    entry['smiles'] = datapoint['smiles']
    entry['source'] = 'ECVAM Ames negative dataset'
    entry['raw input file'] = str(inpf)
    entry['in vitro/in vivo'] = 'in vitro'
    entry['endpoint'] = 'in vitro gene mutation study in bacteria'
    entry['assay'] = 'bacterial reverse mutation assay'
    entry['cell line/species'] = 'unknown'
    entry['metabolic activation'] = 'unknown'
    entry['genotoxicity mode of action'] = 'unknown'
    entry['gene'] = 'unknown'
    entry['notes'] = None
    entry['reference'] = 'https://doi.org/10.1016/j.mrgentox.2020.503199'
    outcome = datapoint['AMES Overall']
    entry['genotoxicity'] = 'positive' if '[+]' in outcome else 'negative' if '[-]' in outcome else 'ambiguous'
    entry['additional source data'] = json.dumps(dict())
    tox_data.append(entry)

# assay: unknown, endpoint: in vitro gene mutation study in mammalian cells
msk_keep = datasets['in vitro MCGM Overall'].str.contains('\[[+-E]\]', na=False)
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    entry = dict()
    entry['CAS number'] = datapoint['CAS number']
    entry['substance name'] = datapoint['substance name']
    entry['smiles'] = datapoint['smiles']
    entry['source'] = 'ECVAM Ames negative dataset'
    entry['raw input file'] = str(inpf)
    entry['in vitro/in vivo'] = 'in vitro'
    entry['endpoint'] = 'in vitro gene mutation study in mammalian cells'
    entry['assay'] = 'unknown'
    entry['cell line/species'] = 'unknown'
    entry['metabolic activation'] = 'unknown'
    entry['genotoxicity mode of action'] = 'unknown'
    entry['gene'] = 'unknown'
    entry['notes'] = None
    entry['reference'] = 'https://doi.org/10.1016/j.mrgentox.2020.503199'
    outcome = datapoint['in vitro MCGM Overall']
    entry['genotoxicity'] = 'positive' if '[+]' in outcome else 'negative' if '[-]' in outcome else 'ambiguous'
    entry['additional source data'] = json.dumps(dict())
    tox_data.append(entry)

# assay: in vitro mammalian chromosome aberration test, endpoint: in vitro chromosome aberration study in mammalian cells
msk_keep = datasets['in vitro CA  Overall'].str.contains('\[[+-E]\]', na=False)
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    entry = dict()
    entry['CAS number'] = datapoint['CAS number']
    entry['substance name'] = datapoint['substance name']
    entry['smiles'] = datapoint['smiles']
    entry['source'] = 'ECVAM Ames negative dataset'
    entry['raw input file'] = str(inpf)
    entry['in vitro/in vivo'] = 'in vitro'
    entry['endpoint'] = 'in vitro chromosome aberration study in mammalian cells'
    entry['assay'] = 'in vitro mammalian chromosome aberration test'
    entry['cell line/species'] = 'unknown'
    entry['metabolic activation'] = 'unknown'
    entry['genotoxicity mode of action'] = 'unknown'
    entry['gene'] = 'unknown'
    entry['notes'] = None
    entry['reference'] = 'https://doi.org/10.1016/j.mrgentox.2020.503199'
    outcome = datapoint['in vitro CA  Overall']
    entry['genotoxicity'] = 'positive' if '[+]' in outcome else 'negative' if '[-]' in outcome else 'ambiguous'
    entry['additional source data'] = json.dumps(dict())
    tox_data.append(entry)

# assay: in vitro mammalian cell micronucleus test, endpoint: in vitro micronucleus study
msk_keep = datasets['in vitro MN Overall'].str.contains('\[[+-E]\]', na=False)
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    entry = dict()
    entry['CAS number'] = datapoint['CAS number']
    entry['substance name'] = datapoint['substance name']
    entry['smiles'] = datapoint['smiles']
    entry['source'] = 'ECVAM Ames negative dataset'
    entry['raw input file'] = str(inpf)
    entry['in vitro/in vivo'] = 'in vitro'
    entry['endpoint'] = 'in vitro micronucleus study'
    entry['assay'] = 'in vitro mammalian cell micronucleus test'
    entry['cell line/species'] = 'unknown'
    entry['metabolic activation'] = 'unknown'
    entry['genotoxicity mode of action'] = 'unknown'
    entry['gene'] = 'unknown'
    entry['notes'] = None
    entry['reference'] = 'https://doi.org/10.1016/j.mrgentox.2020.503199'
    outcome = datapoint['in vitro MN Overall']
    entry['genotoxicity'] = 'positive' if '[+]' in outcome else 'negative' if '[-]' in outcome else 'ambiguous'
    entry['additional source data'] = json.dumps(dict())
    tox_data.append(entry)

# put everything together
tox_data = pd.DataFrame(tox_data)
tox_data = tox_data.reset_index().rename({'index': 'record ID'}, axis='columns')
tox_data['record ID'] = tox_data['source'] + ' ' + tox_data['record ID'].astype(str)

# reorder the columns
tox_data = tox_data[['record ID', 'source', 'raw input file', 'CAS number', 'substance name', 'smiles', 'in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation', 'genotoxicity mode of action', 'gene', 'notes', 'genotoxicity', 'reference', 'additional source data']]

# export the flat dataset
outf = Path('data/ECVAM_AmesNegative/tabular/ECVAM_Ames_negative_genotoxicity.xlsx')
log.info(f'exporting {outf}')
tox_data.to_excel(outf, index=False)

