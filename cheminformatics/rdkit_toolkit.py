import pandas as pd

import logger
log = logger.get_logger(__name__)

import sys
import rdkit
from io import StringIO
from pathlib import Path
import numpy as np

from typing import Union

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Converting smiles to rdkit molecules
log.info('Converting smiles to rdkit molecules')
class Rdkit_operation:
    '''
    Utility class to redirect the standard error to a memory buffer and capture the warnings and errors during
    cheminformatics operations with RDKit. The utility is used as a context manager, for example:
    with Rdkit_operation() as sio:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        error_warning = sio.getvalue()
    '''
    def __enter__(self):
        # redirect the standard error to a memory buffer
        rdkit.rdBase.LogToPythonStderr()
        sio = sys.stderr = StringIO()
        return sio
    def __exit__(self, exc_type, exc_value, exc_tb):
        # set the standard error back to the default
        sys.stderr = sys.__stderr__
        return False # this propagates exceptions out of the working context (default)


def read_sdf(fpath: Union[str, Path]) -> list[Chem.Mol]:
    """
    Reads an sdf file and returns a list of molecule objects
    :param fpath: string with file path or path object pointing to the sdf file
    :return:
    """
    fpath = Path(fpath)
    if not fpath.exists() or not fpath.is_file() or not fpath.suffix == '.sdf':
        ex = FileNotFoundError(f'Path {fpath} must exist and be and sdf file')
        log.error(ex)
        raise ex
    mols = []
    with open(fpath, 'rb') as inf:
        with Rdkit_operation() as sio:
            with Chem.ForwardSDMolSupplier(inf) as suppl:
                for i_mol, mol in enumerate(suppl):
                    if mol is not None:
                        mols.append(mol)
    log.info('read ' + str(len(mols)) + ' molecules from ' + str(fpath))
    return mols

def convert_smiles_to_mol(smiles: str, sanitize=False) -> tuple[Chem.Mol, str]:
    '''
    Converts a smiles string to mol, without sanitisation by default. The returned tuple contains the mol object and
    any warnings or errors during the conversion. In case of failure, the mol object is None.
    :param smiles: input smiles
    :param sanitize: if True it sanitises the molecule
    :return: tuple with resulting mol object and warnings/errors during the conversion
    '''
    mol = None
    error_warning = None
    with Rdkit_operation() as sio:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        error_warning = sio.getvalue()
        if mol and not error_warning:
            log.info(f'successfully converted smiles {smiles} to mol')
        elif mol and error_warning:
            log.info(f'successfully converted smiles {smiles} to mol, but with the error/warning {error_warning}')
        else:
            log.info(f'Failed to convert smiles {smiles} to mol failed with the error/warning {error_warning}')
    return (mol, error_warning)



def get_adjacency_info(mol: Chem.Mol) -> pd.DataFrame:
    """
    Computes the adjacency matrix and the edge indices in COO format with shape [2, *]. The edges are entered twice
    because the graph is undirected.
    :param mol: rdkit molecule
    :return: pandas dataframe with edge indices with shape [2, *] and dtype int32
    """
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # add the edge twice because the graph is undirected
        edge_indices += [[i, j], [j, i]]
    edge_indices = pd.DataFrame(edge_indices, columns=['atomID_1', 'atomID_2'], dtype='int32').T
    return edge_indices


def get_node_features(mol: Chem.Mol, feats=['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization', 'atom_mass']) -> pd.DataFrame:
    """
    Computes the node features for a molecule
    :param mol: rdkit molecule
    :param feats: list of node feature names to compute, for now it supports
                  'atom_symbol' -> atom symbol (str)
                  'atom_charge' -> atom charge (int64)
                  'atom_degree' -> atom degree (int64)
                  'atom_hybridization' -> atom hybridization (str)
                  'atom_mass' -> atom mass (float64)
    :return: pandas dataframe with node features with shape [number_of_atoms, len(feats)]
    """
    all_node_feats = []
    for atom in mol.GetAtoms():
        node_feats = {}
        for feat in feats:
            try:
                if feat == 'atom_symbol':
                    node_feats[feat] = atom.GetSymbol()
                elif feat == 'atom_charge':
                    node_feats[feat] = atom.GetFormalCharge()
                elif feat == 'atom_degree':
                    node_feats[feat] = atom.GetDegree()
                elif feat == 'atom_hybridization':
                    node_feats[feat] = atom.GetHybridization().name
                elif feat == 'atom_mass':
                    node_feats[feat] = atom.GetMass()
                else:
                    raise ValueError(f'node feature {feat} not recognised')
            except Exception as ex:
                log.error(ex)
                raise ex
        all_node_feats.append(node_feats)
    all_node_feats = pd.DataFrame(all_node_feats)
    return all_node_feats


def get_edge_features(mol: Chem.Mol, feats=['bond_type', 'is_conjugated', 'stereo_type']) -> pd.DataFrame:
    """
    Computes the edge features for a molecule
    :param mol: rdkit molecule
    :param feats: list of edge feature names to compute, for now it supports
                  'bond_type' -> bond type (str)
                  'is_conjugated' -> is the bond conjugated (str)
                  'stereo_type' -> bond stereochemistry (str)
    :return: pandas dataframe with edge features with shape [number_of_bonds, len(feats)]
    """
    all_edge_feats = []
    for bond in mol.GetBonds():
        edge_feats = {}
        for feat in feats:
            try:
                if feat == 'bond_type':
                    edge_feats[feat] = bond.GetBondType().name
                elif feat == 'is_conjugated':
                    edge_feats[feat] = str(bond.GetIsConjugated())
                elif feat == 'stereo_type':
                    edge_feats[feat] = bond.GetStereo().name
                else:
                    raise ValueError(f'edge feature {feat} not recognised')
            except Exception as ex:
                log.error(ex)
                raise ex
        # adding edge features twice because the graph is undirected
        all_edge_feats.extend([edge_feats, edge_feats])
    all_edge_feats = pd.DataFrame(all_edge_feats).astype(str)
    return all_edge_feats


def remove_stereo(mol: Chem.Mol, stereo_types=['cis/trans', 'R/S']) -> Chem.Mol:
    '''
    Utility function to remove stereochemistry from a molecule, including cis/trans and R/S stereochemistry.
    :param mol: input molecule
    :param stereo_types: list of stereochemistry types to remove, for now it supports 'cis/trans' and 'R/S'
    :return: molecule with
    '''
    # check the stereo_types requested are valid
    all_stereo_type = ['cis/trans', 'R/S']
    for stereo_type in stereo_types:
        if stereo_type not in all_stereo_type:
            ex = ValueError(f'stereochemistry type {stereo_type} not recognised')
            log.error(ex)
            raise ex
    # remove cis/trans stereochemistry
    if 'cis/trans' in stereo_types:
        for bond in mol.GetBonds():
            if bond.GetStereo() in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
    # remove R/S stereochemistry
    if 'R/S' in stereo_types:
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() in [Chem.ChiralType.CHI_TETRAHEDRAL, Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_TETRAHEDRAL_CW]:
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    # regenerate computed properties like implicit valence and ring information in case it matters
    mol.UpdatePropertyCache(strict=False)
    return mol

def derive_canonical_tautomer(mol: Chem.Mol) -> Chem.Mol:
    '''
    Derives the canonical tautomer of a molecule. This is not necessarily the most stable tautomer.
    :param mol: input molecule
    :return: tuple with the canonical tautomer, warning and error during the tautomerisation
    '''
    te = rdMolStandardize.TautomerEnumerator()
    mol_can_taut = None
    warning = None
    error = None
    with Rdkit_operation() as sio:
        try:
            # remove cis/trans and R/S stereochemistry
            mol_can_taut = remove_stereo(mol)
            # canonical tautomer (this is not necessarily the most stable)
            mol_can_taut = te.Canonicalize(mol_can_taut)
            # remove cis/trans and R/S stereochemistry, in case tautomerism introduced stereoisomerism
            mol_can_taut = remove_stereo(mol_can_taut)
            # capture any warnings/errors during the tautomerisation
            warning = sio.getvalue()
            if warning:
                log.info(warning)
        except Exception as ex:
            log.error(ex)
            error = str(ex)
    return (mol_can_taut, warning, error)



def standardise_mol(mol: Chem.Mol, ops: list[str]=['cleanup', 'addHs', 'tautomerise', 'remove_stereo'])  -> tuple[Chem.Mol, str]:
    """
    Standardises the RDKit molecule. Exceptions are logged and not raised, instead the returned mol object is None.
    :param mol:
    :param ops: applies the standardisation operations in the order specified in the list, for now it supports
                'cleanup' -> xx
                'addHs' -> adds explicit hydrogen atoms
    :return: tuple with resulting mol and warnings/errors during the standardisation
    """

    all_ops = ['cleanup', 'addHs', 'tautomerise', 'remove_stereo']
    # check the standardisation operations requested are valid
    for op in ops:
        if op not in all_ops:
            ex =  ValueError(f'standardisation operation {op} not recognised')
            log.error(ex)
            raise ex

    mol_std = mol
    error_warning = None
    with Rdkit_operation() as sio:
        try:
            for op in ops:
                if op == 'cleanup':
                    mol_std = rdMolStandardize.Cleanup(mol_std)
                elif op == 'addHs':
                    mol_std = Chem.AddHs(mol_std)
                elif op == 'tautomerise':
                    mol_std, _, _ = derive_canonical_tautomer(mol_std)
                elif op == 'remove_stereo':
                    mol_std = remove_stereo(mol_std)
                    print('removed stereo')
        except Exception as ex:
             log.error(ex)
        error_warning = sio.getvalue()
        if not mol_std:
            log.info(f'failed to standardise mol with the error/warning {error_warning}')
            mol_std = None

    return (mol_std, error_warning)
#
# from rdkit import Chem
# i_mol = 0
# mol = mols[i_mol]
# mol = Chem.AddHs(mol)
# [atom.GetSymbol() for atom in mol.GetAtoms()]