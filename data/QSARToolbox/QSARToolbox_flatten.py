'''
Utility script to flatten the QSAR Toolbox dataset to a single excel file.

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

Useful resources

in vitro, gene mutation in mammalian cells
https://www.eurofins.de/biopharma-product-testing-dach-en/validated-standard-testing-methods/hprt-assay/
https://www.fda.gov/regulatory-information/search-fda-guidance-documents/redbook-2000-ivc1c-mouse-lymphoma-thymidine-kinase-gene-mutation-assay
'''

# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/QSARToolbox_flatten.log')

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


# read the datasets
bacterial_mutagenicity_issty = pd.read_csv(r'data\QSARToolbox\raw\2024_06_22_bacterial_mutagenicity_issty.csv', encoding='utf-16le', sep='\t', low_memory=False).assign(origin='2024_06_22_bacterial_mutagenicity_issty.csv')
genotoxicity_carcinogenicity_ecvam = pd.read_csv(r'data\QSARToolbox\raw\2024_06_22_genotoxicity_carcinogenicity_ecvam.csv', encoding='utf-16le', sep='\t', low_memory=False).assign(origin='2024_06_22_genotoxicity_carcinogenicity_ecvam.csv')
genotoxicity_oasis = pd.read_csv(r'data\QSARToolbox\raw\2024_06_22_genotoxicity_oasis.csv', encoding='utf-16le', sep='\t', low_memory=False).assign(origin='2024_06_22_genotoxicity_oasis.csv')
genotoxicity_pesticides_efsa = pd.read_csv(r'data\QSARToolbox\raw\2024_06_22_genotoxicity_pesticides_efsa.csv', encoding='utf-16le', sep='\t', low_memory=False).assign(origin='2024_06_22_genotoxicity_pesticides_efsa.csv')
micronucleus_issmic = pd.read_csv(r'data\QSARToolbox\raw\2024_06_22_micronucleus_issmic.csv', encoding='utf-16le', sep='\t', low_memory=False).assign(origin='2024_06_22_micronucleus_issmic.csv')
micronucleus_oasis = pd.read_csv(r'data\QSARToolbox\raw\2024_06_22_micronucleus_oasis.csv', encoding='utf-16le', sep='\t', low_memory=False).assign(origin='2024_06_22_micronucleus_oasis.csv')
transgenic_rodent_database = pd.read_csv(r'data\QSARToolbox\raw\2024_06_22_transgenic_rodent_database.csv', encoding='utf-16le', sep='\t', low_memory=False).assign(origin='2024_06_22_transgenic_rodent_database.csv')
datasets = pd.concat([bacterial_mutagenicity_issty, genotoxicity_carcinogenicity_ecvam, genotoxicity_oasis, genotoxicity_pesticides_efsa, micronucleus_issmic, micronucleus_oasis, transgenic_rodent_database], axis=0, ignore_index=True, sort=False)

# keep the required columns and rename them
cols = ['CAS Number', 'SMILES', 'Database', 'Strain', 'Metabolic activation', 'Test organisms (species)',
        'Endpoint', 'Type of method', 'Test type', 'Value.MeanValue', 'origin', 'Route of administration']
datasets = datasets[cols].rename({'CAS Number': 'CAS number',
                                  'SMILES': 'smiles',
                                  'Value.MeanValue': 'genotoxicity',}, axis='columns')
datasets = datasets.reset_index().rename({'index': 'source record ID'}, axis='columns')

# check the type of endpoints and test type
stats = datasets.groupby(['Type of method', 'Endpoint', 'Test type', 'Test organisms (species)'], dropna=False)[['origin', 'CAS number']].agg(**{'origin': pd.NamedAgg(column='origin', aggfunc=lambda x: ';'.join(pd.Series(x).drop_duplicates().sort_values().to_list())),
                                                                                                       'count (CAS number)': pd.NamedAgg(column='CAS number', aggfunc=lambda x: pd.Series(x).dropna().drop_duplicates().nunique())
                                                                                                      }).reset_index().sort_values(by='count (CAS number)', ascending=False)
log.info(stats.to_markdown())

# keep only the records for which the smiles is present
msk = datasets['smiles'].notnull()
log.info(f'removing {(~msk).sum()} records without smiles')
datasets = datasets.loc[msk]

# keep only records that are positive, negative or equivocal (to be converted to ambiguous)
msk = datasets['genotoxicity'].isin(['Positive', 'Negative', 'Equivocal'])
log.info(f'removing {(~msk).sum()} records that are not positive, negative or equivocal')
datasets = datasets.loc[msk]
datasets['genotoxicity'] = datasets['genotoxicity'].replace({'Negative': 'negative',
                                                             'Positive': 'positive',
                                                             'Equivocal': 'ambiguous'})

# cast the genotoxicity data to the expected, flat format
tox_data = []

# assay: bacterial reverse mutation assay, endpoint: in vitro gene mutation study in bacteria, Salmonella typhimurium
assay = 'bacterial reverse mutation assay'
endpoint = 'in vitro gene mutation study in bacteria'
main_strains = ['TA 100', 'TA 98', 'TA 1535', 'TA 1537', 'TA 1538', 'TA 97', 'TA 102', 'TA 104']
msk_keep = ((datasets['Test type'] == 'Bacterial Reverse Mutation Assay (e.g. Ames Test)')
            & (datasets['Test organisms (species)'] == 'Salmonella typhimurium')
            )
log.info('processing bacterial reverse mutation assay, in vitro gene mutation study in bacteria, Salmonella typhimurium')
log.info(f'{msk_keep.sum()} records processed, {(~msk_keep).sum()} records left')
log.info(datasets.loc[msk_keep, ['Test organisms (species)', 'Strain', 'Metabolic activation', 'genotoxicity']].value_counts())
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    # see https://www.sciencedirect.com/science/article/pii/S0027510700000646?via%3Dihub
    moa_map = {'TA 100': 'base pair substitution',
               'TA 1535': 'base pair substitution',
               'TA 98': 'frameshift',
               'TA 1538': 'frameshift',
               'TA 1537': 'frameshift',
               'TA 97': 'frameshift',
               'TA 102': 'transitions/transversions',
               'TA 104': 'transitions/transversions',
               }
    gene_map = {'TA 100': 'hisG46',
               'TA 1535': 'hisG46',
               'TA 98': 'hisD3052',
               'TA 1538': 'hisD3052',
               'TA 1537': 'hisC3076',
               'TA 97': 'hisD6610',
               'TA 102': 'hisG428',
               'TA 104': 'hisG428',
               }
    entry = {'source': f'QSAR Toolbox, {datapoint["Database"]} database',
             'raw input file': datapoint['origin'],
             'CAS number': datapoint['CAS number'],
             'substance name': None,
             'smiles': datapoint['smiles'],
             'in vitro/in vivo': 'in vitro',
             'endpoint': endpoint,
             'assay': assay,
             'cell line/species': 'Salmonella typhimurium'+' ('+(datapoint['Strain'] if datapoint['Strain'] in main_strains else 'unknown')+')',
             'metabolic activation': 'to be filled',
             'genotoxicity mode of action': moa_map.get(datapoint['Strain'], 'unknown'),
             'gene': gene_map.get(datapoint['Strain'], 'unknown'),
             'notes': None,
             'genotoxicity': datapoint['genotoxicity'],
             'reference': None,
             'additional source data': None
             }
    if datapoint['Metabolic activation'] == 'With S9':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'Without S9':
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'With or Without':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    else:
        entry['metabolic activation'] = 'unknown'
        tox_data.append(entry)
datasets = datasets.loc[~msk_keep]

# assay: bacterial reverse mutation assay, endpoint: in vitro gene mutation study in bacteria, Escherichia coli
assay = 'bacterial reverse mutation assay'
endpoint = 'in vitro gene mutation study in bacteria'
main_strains = ['WP2 Uvr A', 'WP2 Uvr A PKM 101', 'WP2']
msk_keep = ((datasets['Test type'] == 'Bacterial Reverse Mutation Assay (e.g. Ames Test)')
       & (datasets['Test organisms (species)'] == 'Escherichia coli')
       )
log.info('processing bacterial reverse mutation assay, in vitro gene mutation study in bacteria, Escherichia coli')
log.info(f'{msk_keep.sum()} records processed, {(~msk_keep).sum()} records left')
log.info(datasets.loc[msk_keep, ['Test organisms (species)', 'Strain', 'Metabolic activation', 'genotoxicity']].value_counts())
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    # see https://www.sciencedirect.com/science/article/pii/S0027510700000646?via%3Dihub
    moa_map = {'WP2 Uvr A': 'transitions/transversions',
               'WP2 Uvr A PKM 101': 'transitions/transversions',
               'WP2': 'transitions/transversions',
               }
    gene_map = {
               }
    entry = {'source': f'QSAR Toolbox, {datapoint["Database"]} database',
             'raw input file': datapoint['origin'],
             'CAS number': datapoint['CAS number'],
             'substance name': None,
             'smiles': datapoint['smiles'],
             'in vitro/in vivo': 'in vitro',
             'endpoint': endpoint,
             'assay': assay,
             'cell line/species': 'Escherichia coli'+' ('+(datapoint['Strain'] if datapoint['Strain'] in main_strains else 'unknown')+')',
             'metabolic activation': 'to be filled',
             'genotoxicity mode of action': moa_map.get(datapoint['Strain'], 'unknown'),
             'gene': gene_map.get(datapoint['Strain'], 'unknown'),
             'notes': None,
             'genotoxicity': datapoint['genotoxicity'],
             'reference': None,
             'additional source data': None
             }
    if datapoint['Metabolic activation'] == 'With S9':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'Without S9':
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'With or Without':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    else:
        entry['metabolic activation'] = 'unknown'
        tox_data.append(entry)
datasets = datasets.loc[~msk_keep]

# assay: in vitro mammalian chromosome aberration test, endpoint: in vitro chromosome aberration study in mammalian cells
# note: we understood these records as chromosome aberration, but it could be that some are micronucleus,
#       this is a general issue with the in vitro cytogenicity data)
assay = 'in vitro mammalian chromosome aberration test'
endpoint = 'in vitro chromosome aberration study in mammalian cells'
msk_keep = (datasets['Type of method']=='in Vitro') & (datasets['Test type'] == 'in Vitro Mammalian Chromosome Aberration Test')
# .. combine the species "Human" with the strain "Lymphocytes"
datasets.loc[msk_keep, 'Strain'] = np.where(datasets.loc[msk_keep, 'Strain']=='Lymphocytes', 'lymphocytes', datasets.loc[msk_keep, 'Strain'])
main_species_cell_line = ['Chinese Hamster Lung (CHL)', 'Chinese Hamster Ovary (CHO)', 'Chinese Hamster Lung Fibroblasts (V79)', 'lymphocytes']
log.info('processing in vitro mammalian chromosome aberration test, in vitro chromosome aberration study in mammalian cells')
log.info(f'{msk_keep.sum()} records processed, {(~msk_keep).sum()} records left')
log.info(datasets.loc[msk_keep, ['Test organisms (species)', 'Strain', 'Metabolic activation', 'genotoxicity']].value_counts())
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    moa_map = {
               }
    gene_map = {
               }
    entry = {'source': f'QSAR Toolbox, {datapoint["Database"]} database',
             'raw input file': datapoint['origin'],
             'CAS number': datapoint['CAS number'],
             'substance name': None,
             'smiles': datapoint['smiles'],
             'in vitro/in vivo': 'in vitro',
             'endpoint': endpoint,
             'assay': assay,
             'cell line/species': datapoint['Strain'] if datapoint['Strain'] in main_species_cell_line else 'unknown',
             'metabolic activation': 'to be filled',
             'genotoxicity mode of action': moa_map.get(datapoint['Strain'], 'unknown'),
             'gene': gene_map.get(datapoint['Strain'], 'unknown'),
             'notes': None,
             'genotoxicity': datapoint['genotoxicity'],
             'reference': None,
             'additional source data': None
             }
    if datapoint['Metabolic activation'] == 'With S9':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'Without S9':
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'With or Without':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    else:
        entry['metabolic activation'] = 'unknown'
        tox_data.append(entry)
datasets = datasets.loc[~msk_keep]


# assay: unknown, endpoint: in vitro gene mutation study in mammalian cells
assay = 'unknown'
endpoint = 'in vitro gene mutation study in mammalian cells'
msk_keep = (datasets['Type of method']=='in Vitro') & (datasets['Test type'].str.contains(r'(?i)mammalian.*gene\s+mutation'))
datasets.loc[msk_keep, 'Strain'] = np.where(datasets.loc[msk_keep, 'Strain']=='Lymphoma L5178Y Cells', 'mouse lymphoma L5178Y cells', datasets.loc[msk_keep, 'Strain'])
main_species_cell_line = ['Chinese Hamster Lung (CHL)', 'Chinese Hamster Ovary (CHO)', 'Chinese Hamster Lung Fibroblasts (V79)', 'mouse lymphoma L5178Y cells']
log.info('processing in vitro gene mutation study in mammalian cells')
log.info(f'{msk_keep.sum()} records processed, {(~msk_keep).sum()} records left')
log.info(datasets.loc[msk_keep, ['Test organisms (species)', 'Strain', 'Metabolic activation', 'genotoxicity']].value_counts())
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    moa_map = {
               }
    gene_map = {
               }
    entry = {'source': f'QSAR Toolbox, {datapoint["Database"]} database',
             'raw input file': datapoint['origin'],
             'CAS number': datapoint['CAS number'],
             'substance name': None,
             'smiles': datapoint['smiles'],
             'in vitro/in vivo': 'in vitro',
             'endpoint': endpoint,
             'assay': assay,
             'cell line/species': datapoint['Strain'] if datapoint['Strain'] in main_species_cell_line else 'unknown',
             'metabolic activation': 'to be filled',
             'genotoxicity mode of action': moa_map.get(datapoint['Strain'], 'unknown'),
             'gene': gene_map.get(datapoint['Strain'], 'unknown'),
             'notes': None,
             'genotoxicity': datapoint['genotoxicity'],
             'reference': None,
             'additional source data': None
             }
    if datapoint['Metabolic activation'] == 'With S9':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'Without S9':
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'With or Without':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    else:
        entry['metabolic activation'] = 'unknown'
        tox_data.append(entry)
datasets = datasets.loc[~msk_keep]



# assay: in vitro mammalian cell micronucleus test, endpoint: in vitro micronucleus study
assay = 'in vitro mammalian cell micronucleus test'
endpoint = 'in vitro micronucleus study'
msk_keep = (datasets['Type of method']=='in Vitro') & (datasets['Test type'].str.contains(r'(?i)micronucleus'))
datasets.loc[msk_keep, 'Strain'] = np.where(datasets.loc[msk_keep, 'Strain']=='Lymphocytes', 'lymphocytes', datasets.loc[msk_keep, 'Strain'])
main_species_cell_line = ['lymphocytes']
log.info('processing in vitro mammalian cell micronucleus test, in vitro micronucleus study')
log.info(f'{msk_keep.sum()} records processed, {(~msk_keep).sum()} records left')
log.info(datasets.loc[msk_keep, ['Test organisms (species)', 'Strain', 'Metabolic activation', 'genotoxicity']].value_counts())
for idx, datapoint in datasets.loc[msk_keep].iterrows():
    moa_map = {
               }
    gene_map = {
               }
    entry = {'source': f'QSAR Toolbox, {datapoint["Database"]} database',
             'raw input file': datapoint['origin'],
             'CAS number': datapoint['CAS number'],
             'substance name': None,
             'smiles': datapoint['smiles'],
             'in vitro/in vivo': 'in vitro',
             'endpoint': endpoint,
             'assay': assay,
             'cell line/species': datapoint['Strain'] if datapoint['Strain'] in main_species_cell_line else 'unknown',
             'metabolic activation': 'to be filled',
             'genotoxicity mode of action': moa_map.get(datapoint['Strain'], 'unknown'),
             'gene': gene_map.get(datapoint['Strain'], 'unknown'),
             'notes': None,
             'genotoxicity': datapoint['genotoxicity'],
             'reference': None,
             'additional source data': None
             }
    if datapoint['Metabolic activation'] == 'With S9':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'Without S9':
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    elif datapoint['Metabolic activation'] == 'With or Without':
        entry['metabolic activation'] = 'yes'
        tox_data.append(entry)
        entry['metabolic activation'] = 'no'
        tox_data.append(entry)
    else:
        entry['metabolic activation'] = 'unknown'
        tox_data.append(entry)
datasets = datasets.loc[~msk_keep]









# set the record ID for all processed records
tox_data = pd.DataFrame(tox_data).reset_index(drop=True).reset_index().rename({'index': 'record ID'}, axis='columns')
tox_data['record ID'] = tox_data['source'] + ' ' + tox_data['record ID'].astype(str)

# reorder the columns
tox_data = tox_data[['record ID', 'source', 'raw input file', 'CAS number', 'substance name', 'smiles', 'in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation', 'genotoxicity mode of action', 'gene', 'notes', 'genotoxicity', 'reference', 'additional source data']]

# export the flat dataset
outf = Path('data/QSARToolbox/tabular/QSARToolbox_genotoxicity.xlsx')
log.info(f'exporting {outf}')
tox_data.to_excel(outf, index=False)


