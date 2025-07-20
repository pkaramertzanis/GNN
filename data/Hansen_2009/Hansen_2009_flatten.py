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
log = logger.setup_applevel_logger(file_name ='logs/Hansen_2009_flatten.log')

from pathlib import Path
import pandas as pd
import numpy as np
import json

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


# read the raw Hansen 2009 mutagenicity dataset, https://doc.ml.tu-berlin.de/toxbenchmark/
inpf = Path(r'data\Hansen_2009\raw\smiles_cas_N6512.smi')
tox_data = pd.read_csv(inpf, sep='\t', header=None, encoding='cp1252', names=['smiles', 'cas_number', 'result'])
tox_data['cas_number'] = tox_data['cas_number'].str.strip()

# rename existing and add missing columns
tox_data = tox_data.rename({'cas_number': 'CAS number'}, axis='columns')
tox_data['substance name'] = None
tox_data['source'] = 'Hansen 2009'
tox_data['raw input file'] = str(inpf)
tox_data['in vitro/in vivo'] = 'in vitro'
tox_data['endpoint'] = 'in vitro gene mutation study in bacteria'
tox_data['assay'] = 'bacterial reverse mutation assay'
tox_data['cell line/species'] = 'unknown'
tox_data['metabolic activation'] = 'unknown'
tox_data['genotoxicity mode of action'] = 'unknown'
tox_data['gene'] = 'unknown'
tox_data['notes'] = None
tox_data['reference'] = 'https://doi.org/10.1021/ci900161g'
tox_data['genotoxicity'] = np.where(tox_data['result']==1, 'positive', 'negative')
tox_data['additional source data'] = [json.dumps(dict())]*len(tox_data)
tox_data = tox_data.reset_index().rename({'index': 'record ID'}, axis='columns')
tox_data['record ID'] = tox_data['source'] + ' ' + tox_data['record ID'].astype(str)

# reorder the columns
tox_data = tox_data[['record ID', 'source', 'raw input file', 'CAS number', 'substance name', 'smiles', 'in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation', 'genotoxicity mode of action', 'gene', 'notes', 'genotoxicity', 'reference', 'additional source data']]

# export the flat dataset
outf = Path('data/Hansen_2009/tabular/Hansen_2009_genotoxicity.xlsx')
log.info(f'exporting {outf}')
tox_data.to_excel(outf, index=False)

