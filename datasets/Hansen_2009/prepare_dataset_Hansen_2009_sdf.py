'''
Utility script to convert the original Hansen 2009 mutagenicity dataset (https://doc.ml.tu-berlin.de/toxbenchmark/) to
sdf file. The sdf file is then used to create a PyTorch Geometric dataset. The molecule standardisation and feature
creation is not performed here, but in the PyTorch Geometric dataset creation script. This is to allow flexibility,
e.g. to use different feature sets, or to use different standardisation methods when testing predictive performance.

A script like this will be created for each dataset that is used in the project. This script can be seen as a standalone,
run-once utility script.
'''

# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/prepare_dataset_Hansen_2009_sdf.log')

from pathlib import Path
import pandas as pd
from rdkit import Chem
import json
from collections import Counter

from cheminformatics.rdkit_toolkit import convert_smiles_to_mol


# read the raw Hansen 2009 mutagenicity dataset, https://doc.ml.tu-berlin.de/toxbenchmark/
inpf = Path(r'datasets\Hansen_2009\raw\smiles_cas_N6512.smi')
tox_data = pd.read_csv(inpf, sep='\t', header=None, encoding='cp1252', names=['smiles', 'cas_number', 'result'])


# create output folder to store the resulting sdf file
outf = Path(r'datasets\Hansen_2009\processed\sdf')
outf.mkdir(parents=True, exist_ok=True)
outf = outf / 'Ames_generic_Hansen_2009.sdf'


# convert the smiles to rdkit molecules and store the molecules in an sdf file
# the assay results are stored as a json dump in the molecule properties
with Chem.SDWriter(outf) as sdf_writer:
    count_success = 0
    count_fail = 0
    count_error_warning = 0
    assay_results = []
    for index, row in tox_data.iterrows():
        print(index)
        smiles = row['smiles']
        mol, error_warning = convert_smiles_to_mol(smiles, sanitize=True)


        if mol is not None:

            # standardise the molecule
            # .. add explicit hydrogen atoms
            mol = Chem.AddHs(mol)

            # set the mol name
            mol_name = 'Hansen 2009 dataset, entry ' + str(index)
            mol.SetProp('_Name', 'Hansen 2009 dataset, entry ' + str(index))

            # set the CAS number
            mol.SetProp('CAS', row['cas_number'].strip())

            # set the smiles
            smiles = Chem.MolToSmiles(mol)
            mol.SetProp('smiles', smiles.strip())

            # set error/warning information
            if error_warning:
                mol.SetProp('error_warning_smiles_to_mol', error_warning)
                count_error_warning += 1
            else:
                mol.SetProp('error_warning_smiles_to_mol', 'no error/warning')

            # set the assay information
            assay_data = {'assay': 'AMES',
                           'assay_notes': 'all strains, with and without metabolic activation',
                           'assay_result': 'positive' if row['result'] == 1 else 'negative'}
            assay_results.append(assay_data['assay_result'])
            # the assay data is stored as a JSON array because for some datasets there is more than one assay, e.g. REACH
            mol.SetProp('assay_data', json.dumps([assay_data]))
            sdf_writer.write(mol)
            log.info(f'stored {mol_name} in {outf}')
            count_success += 1
        else:
            count_fail += 1
log.info(f'Hansen dataset size {len(tox_data)}: {count_success} structures stored in {outf}, out of which {count_error_warning} have errors/warnings' )
log.info(f'assay results: {dict(Counter(assay_results))}')

