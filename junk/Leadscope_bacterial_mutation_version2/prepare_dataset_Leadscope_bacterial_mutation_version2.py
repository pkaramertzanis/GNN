'''
Utility script to convert the Leadscope bacterial mutation dataset to
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
from io import StringIO, BytesIO

from cheminformatics.rdkit_toolkit import convert_smiles_to_mol, Rdkit_operation


# read the raw Leadscope bacterial mutation (version 2) dataset, https://www.leadscope.com/QMRFs/2022/qmrf-Leadscope_QSAR_Microbial_in_vitro_-_Bacterial_Mutation.pdf
inp_sdf = Path(r'datasets/Leadscope_bacterial_mutation_version2/raw/Leadscope_bacterial_mutation_version2.sdf')
mols = []
with open(inp_sdf, 'rt', encoding='cp1252') as inf:
    with Rdkit_operation() as sio:
        with Chem.ForwardSDMolSupplier(BytesIO(inf.read().encode('utf-8'))) as suppl:
            for i_mol, mol in enumerate(suppl):
                if mol is not None:
                    mols.append(mol)
log.info('read ' + str(len(mols)) + ' molecules from ' + str(inp_sdf))




# create output folder to store the resulting sdf file
outf = Path(r'datasets\Leadscope_bacterial_mutation_version2\processed\sdf')
outf.mkdir(parents=True, exist_ok=True)
outf = outf / 'Leadscope_bacterial_mutation_version2.sdf'


# convert the smiles to rdkit molecules and store the molecules in an sdf file
# the assay results are stored as a json dump in the molecule properties
with Chem.SDWriter(outf) as sdf_writer:
    assay_results = []
    count_success = 0
    count_fail = 0
    count_error_warning = 0
    for index, mol in enumerate(mols):

        # standardise the molecule
        # .. add explicit hydrogen atoms
        mol = Chem.AddHs(mol, addCoords=True)

        # fetch the needed properties from the original mol
        mol_name = mol.GetPropsAsDict().get('Substance ID', '-')
        cas_number = mol.GetPropsAsDict().get('CAS.Number', '-')

        bacterial_mutation = mol.GetPropsAsDict().get('Bacterial.Mutation')
        if bacterial_mutation is None:
            log.warning(f'no bacterial mutation data for {mol_name}')
            count_fail += 1
            continue

        # remove the old properties
        for prop in mol.GetPropNames():
            mol.ClearProp(prop)

        # add the needed properties
        mol.SetProp('_Name', mol_name)
        mol.SetProp('CAS', cas_number)

        # set the smiles
        smiles = Chem.MolToSmiles(mol)
        mol.SetProp('smiles', smiles.strip())

        # set the assay information
        assay_data = {'assay': 'AMES',
                      'assay_notes': 'all strains, with and without metabolic activation',
                      'assay_result': 'positive' if float(bacterial_mutation) > 0.5 else 'negative'}
        assay_results.append(assay_data['assay_result'])
        mol.SetProp('assay_data', json.dumps([assay_data]))
        log.info(f'stored {mol_name} in {outf}')

        # write the molecule to the sdf file
        sdf_writer.write(mol)

        count_success += 1
log.info(f'Leadscope dataset size {len(mols)}: {count_success} structures stored in {outf}' )
log.info(f'assay results: {dict(Counter(assay_results))}')