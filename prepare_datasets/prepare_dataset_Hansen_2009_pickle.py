# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_muta_prepare.log')

# setup logging
import numpy as np

import pandas as pd
from pathlib import Path

from tqdm import tqdm
tqdm.pandas()

# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr', False)
# maximum number of columns
pd.set_option("display.max_columns",50)
# maximum number of rows
pd.set_option("display.max_rows",500)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 120)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True


# read in the Hansen 2009 mutagenicity dataset, https://doc.ml.tu-berlin.de/toxbenchmark/
inpf = Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\input\Hansen_2009\smiles_cas_N6512.smi')
tox_data = pd.read_csv(inpf, sep='\t', header=None, encoding='cp1252', names=['smiles', 'cas_number', 'result'])
tox_data['result'] = tox_data['result'].map({1: 'positive', 0: 'negative'})


# Converting smiles to rdkit molecules
log.info('Converting smiles to rdkit molecules')
import sys
import rdkit
from io import StringIO
from rdkit import Chem
class Rdkit_operation:
    def __enter__(self):
        # redirect the standard error to a memory buffer
        rdkit.rdBase.LogToPythonStderr()
        sio = sys.stderr = StringIO()
        return sio
    def __exit__(self, exc_type, exc_value, exc_tb):
        # print(exc_type, exc_value, exc_tb, sep="\n")
        # set the standard error back to the default
        sys.stderr = sys.__stderr__
        return False # this propagates exceptions out of the working context (default)
def convert_smiles_to_mol(smiles: str) -> tuple[str, str]:
    '''
    Converts a smiles string to mol
    :param smiles: input smiles
    :return: tuple with resulting mol string and warnings/errors during the conversion
    '''
    mol = None
    error_warning = None
    with Rdkit_operation() as sio:
        mol = Chem.MolFromSmiles(smiles)
        error_warning = sio.getvalue()
    return (mol, error_warning)
tmp = tox_data['smiles'].apply(convert_smiles_to_mol).apply(pd.Series)
tox_data['mol'] = tmp.iloc[:, 0]
tox_data['error/warning'] = tmp.iloc[:, 1].apply(lambda s: s if s else None)


# keep only the structures without errors and warnings
log.info('Removing structures with errors and warnings')
msk = tox_data['error/warning'].isnull()
tox_data = tox_data.loc[msk].drop('error/warning', axis='columns')
log.info(f'dataset now has {len(tox_data)} records')


# sanitise the molecules
log.info('Sanitising the molecules')
def sanitise_mol(mol: str) -> str:
    # add explicit hydrogen atoms
    mol = Chem.AddHs(mol)
    return mol
tox_data['mol'] = tox_data['mol'].apply(sanitise_mol)



# compute adjacency, node and edge matrices
log.info('Computing the adjacency, atom properties and bond properties matrices')
def compute_adjacency_atom_properties_edge_properties(mol: str) -> str:
    # compute the adjacency matrix
    adjacency_matrix = rdkit.Chem.rdmolops.GetAdjacencyMatrix(mol)
    # compute the atom properties
    atom_properties = []
    for atom in mol.GetAtoms():
        tmp = {'atom number': atom.GetIdx(),
                'atomic number': atom.GetAtomicNum(),
                'atom symbol': atom.GetSymbol(),
                'charge': atom.GetFormalCharge()}
        atom_properties.append(tmp)
    # compute the edge properties
    bond_properties = []
    for bond in mol.GetBonds():
        tmp = {'beginning atom index': bond.GetBeginAtomIdx(),
               'end atom index': bond.GetEndAtomIdx(),
               'bond type': bond.GetBondType().name}
        bond_properties.append(tmp)
    return {'adjacency matrix': adjacency_matrix,
            'atom properties': atom_properties,
            'bond properties': bond_properties}
adjacency_properties = tox_data['mol'].progress_apply(compute_adjacency_atom_properties_edge_properties).rename('adjacency_properties')


# eliminate molecules with fewer than 3 atoms and fewer than 3 bonds
log.info('Eliminating molecules with fewer than 3 atoms and fewer than 3 bonds')
number_atoms = adjacency_properties.apply(lambda matrices: matrices['atom properties']).apply(len)
number_bonds = adjacency_properties.apply(lambda matrices: matrices['bond properties']).apply(len)
msk = (number_atoms >=3) & (number_bonds >= 3)
log.info(f'removed {(~msk).sum()} molecules out of {len(msk)} because of low number of atoms/bonds')
adjacency_properties = adjacency_properties.loc[msk].reset_index(drop=True).rename_axis('mol ID').reset_index()
tox_data = tox_data.loc[msk].reset_index(drop=True).rename_axis('mol ID').reset_index()
log.info(f'dataset now has {len(tox_data)} records')


# preparing the dataset for modeling
# .. labels
log.info('Fetching the labels')
labels = tox_data['result']
# .. adjacency matrix
log.info('Fetching the adjacency matrices')
adjacency_matrices = adjacency_properties.assign(**{'adjacency matrix': lambda df: df['adjacency_properties'].apply(lambda x: x.get('adjacency matrix'))}).drop('adjacency_properties', axis='columns')
# .. one-hot-encoding of atomic numbers
log.info('One-hot-enconding of atomic numbers')
atomic_numbers = adjacency_properties['adjacency_properties'].apply(lambda matrices: matrices['atom properties']).rename_axis('mol ID', axis='index').reset_index().explode('adjacency_properties')
atomic_numbers['atomic number'] = atomic_numbers['adjacency_properties'].apply(lambda x: x['atomic number'])
atomic_numbers = atomic_numbers[['mol ID', 'atomic number']]
atomic_numbers['atomic number'].value_counts()
atomic_numbers = pd.concat([atomic_numbers['mol ID'], pd.get_dummies(atomic_numbers['atomic number'], prefix='AtNum', dtype=int)], axis='columns', sort=False, ignore_index=False)
# .. one-hot-encoding of atomic charges
log.info('One-hot-enconding of atom charges')
charges = adjacency_properties['adjacency_properties'].apply(lambda matrices: matrices['atom properties']).rename_axis('mol ID', axis='index').reset_index().explode('adjacency_properties')
charges['charge'] = charges['adjacency_properties'].apply(lambda x: x['charge'])
charges = charges[['mol ID', 'charge']]
charges['charge'].value_counts()
charges = pd.concat([charges['mol ID'], pd.get_dummies(charges['charge'], prefix='Chg', dtype=int)], axis='columns', sort=False, ignore_index=False)
# .. node matrix
log.info('Compiling the node matrix')
X = pd.concat([atomic_numbers, charges.drop('mol ID', axis='columns')], axis='columns', sort=False, ignore_index=False)
# .. edge matrix
log.info('Compiling the edge matrix')
edges = adjacency_properties['adjacency_properties'].apply(lambda matrices: matrices['bond properties']).rename_axis('mol ID', axis='index').reset_index().explode('adjacency_properties')
edges['atom IDs'] = edges['adjacency_properties'].apply(lambda bond_properties: (bond_properties['beginning atom index'], bond_properties['end atom index']))
edges['bond type'] = edges['adjacency_properties'].apply(lambda bond_properties: bond_properties['bond type'])
edges = pd.concat([edges[['mol ID', 'atom IDs']], pd.get_dummies(edges['bond type'], prefix='BndTyp', dtype=int)], axis='columns', sort=False, ignore_index=False)

# export the training dataset
# the exported dataset assumes that all molecular structure preprocessing has been completed
# the dataset is stored in a pickle file that contains
# tox_data: toxicity data with the columns 'mol ID', 'smiles', 'cas_number', 'result', 'mol'
# X: node matrix
# X: adjacency matrices
# edges: edge matrices
# labels: labels, strings 'positive' and 'negative'
import pickle
outf = Path('training_sets/Hansen_2009.pkl')
with open(outf, 'wb') as file:
    tmp = (tox_data, X, adjacency_matrices, edges, labels)
    pickle.dump(tmp, file)
    log.info(f'dataset stored in {outf}')

arr = adjacency_matrices['adjacency matrix'][0]
indices = np.where(arr == 1)
indices_list = set(zip(indices[0], indices[1]))
indices_list2 = set(edges.loc[edges['mol ID']==0, 'atom IDs'])

indices_list.difference(indices_list2)