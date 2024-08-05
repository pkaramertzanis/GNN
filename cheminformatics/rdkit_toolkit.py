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
from rdkit.Chem import Descriptors

import logging
from io import StringIO
from rdkit import rdBase
import re


class Rdkit_operation:
    '''
    Utility class to redirect the standard error to a memory buffer and capture the warnings and errors during
    cheminformatics operations with RDKit. The utility is used as a context manager, for example:
    with Rdkit_operation() as sio:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        error_warning = sio
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
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
            error_warning = val if (val:=sio.getvalue()) else None
        except Exception as ex:
            # log.error(ex)
            mol = None
    if mol and not error_warning:
        log.info(f'successfully converted smiles {smiles} to mol')
    elif mol and error_warning:
        log.warning(f'successfully converted smiles {smiles} to mol, but with the error/warning {error_warning}')
    else:
        log.warning(f'Failed to convert smiles {smiles} to mol failed with the error/warning {error_warning}')
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


def get_node_features(mol: Chem.Mol, feats=None) -> pd.DataFrame:
    """
    Computes the node features for a molecule
    :param mol: rdkit molecule
    :param feats: list of node feature names to compute, for now it supports
                  'atom_symbol' -> atom symbol (str)
                  'atom_charge' -> atom charge (int64)
                  'atom_degree' -> atom degree (int64)
                  'atom_hybridization' -> atom hybridization (str)
                  'atom_mass' -> atom mass (float64)
                  'num_rings' -> number of rings the atom is part of (int64)
                  'num_Hs' -> number of hydrogen atoms (int64)
    :return: pandas dataframe with node features with shape [number_of_atoms, len(feats)]
    """

    if feats is None:
        feats = ['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization', 'atom_mass', 'num_rings', 'num_Hs']

    # get the ring information if num_rings is requested
    if'num_rings' in feats:
        ring_info = mol.GetRingInfo()
        # create a list to store the number of rings each atom is part of
        atom_ring_counts = [0] * mol.GetNumAtoms()
        # iterate through the rings and update the atom ring counts
        for ring in ring_info.AtomRings():
            for atom_idx in ring:
                atom_ring_counts[atom_idx] += 1

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
                elif feat == 'num_rings':
                    node_feats[feat] = atom_ring_counts[atom.GetIdx()]
                elif feat == 'num_Hs':
                    node_feats[feat] = atom.GetTotalNumHs()
                else:
                    raise ValueError(f'node feature {feat} not recognised')
            except Exception as ex:
                log.error(ex)
                raise ex
        all_node_feats.append(node_feats)
    all_node_feats = pd.DataFrame(all_node_feats)
    return all_node_feats


def get_edge_features(mol: Chem.Mol, feats=None) -> pd.DataFrame:
    """
    Computes the edge features for a molecule
    :param mol: rdkit molecule
    :param feats: list of edge feature names to compute, for now it supports
                  'bond_type' -> bond type (str)
                  'is_conjugated' -> is the bond conjugated (str)
                  'stereo_type' -> bond stereochemistry (str)
                  'num_rings'- -> number of rings the bond is part of (int64)
    :return: pandas dataframe with edge features with shape [number_of_bonds, len(feats)]
    """

    if feats is None:
        feats = ['bond_type', 'is_conjugated', 'stereo_type', 'num_rings']

    # get the ring information if num_rings is requested
    if'num_rings' in feats:
        ring_info = mol.GetRingInfo()
        # create a list to store the number of rings each bond is part of
        bond_ring_counts = [0] * mol.GetNumBonds()
        # iterate through the rings and update the bond ring counts
        for ring in ring_info.BondRings():
            for bond_idx in ring:
                bond_ring_counts[bond_idx] += 1

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
                elif feat == 'num_rings':
                    edge_feats[feat] = bond_ring_counts[bond.GetIdx()]
                else:
                    raise ValueError(f'edge feature {feat} not recognised')
            except Exception as ex:
                log.error(ex)
                raise ex
        # adding edge features twice because the graph is undirected
        all_edge_feats.extend([edge_feats, edge_feats])
    all_edge_feats = pd.DataFrame(all_edge_feats).astype(str)
    return all_edge_feats


def remove_stereo(mol: Chem.Mol, stereo_types=None) -> Chem.Mol:
    '''
    Utility function to remove stereochemistry from a molecule, including cis/trans and R/S stereochemistry.
    :param mol: input molecule
    :param stereo_types: list of stereochemistry types to remove, for now it supports 'cis/trans' and 'R/S'
    :return: molecule with
    '''


    # check the stereo_types requested are valid
    all_stereo_type = ['cis/trans', 'R/S']

    if stereo_types is None:
        stereo_types = all_stereo_type

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



def standardise_mol(mol: Chem.Mol, ops: list[str]=None)  -> tuple[Chem.Mol, str]:
    """
    Standardises the RDKit molecule. Exceptions are logged and not raised, instead the returned mol object is None.
    :param mol:
    :param ops: applies the standardisation operations in the order specified in the list, for now it supports
                'cleanup' -> applies the rdMolStandardize.Cleanup operation
                'uncharge' -> returns the uncharged molecule (applies only if only one fragment)
                'addHs' -> adds explicit hydrogen atoms
                'remove_stereo' -> removes stereochemistry (R/S and cis/trans)
                'tautomerise' -> derives the canonical tautomer
    :return: tuple with resulting mol and warnings/errors during the standardisation
    """
    if ops is None:
        ops = ['cleanup', 'uncharge', 'addHs', 'remove_stereo', 'tautomerise']

    all_ops = ['cleanup', 'uncharge', 'addHs', 'tautomerise', 'remove_stereo']
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
                    # in case wewish to change thedefault parameters
                    params= rdMolStandardize.CleanupParameters()
                    mol_std = rdMolStandardize.Cleanup(mol_std, params=params)
                elif op == 'uncharge':
                    if len(Chem.GetMolFrags(mol_std)) == 1:
                        mol_std = rdMolStandardize.ChargeParent(mol_std)
                elif op == 'addHs':
                    mol_std = Chem.AddHs(mol_std)
                elif op == 'tautomerise':
                    mol_std, _, _ = derive_canonical_tautomer(mol_std)
                elif op == 'remove_stereo':
                    mol_std = remove_stereo(mol_std)
                    print('removed stereo')
        except Exception as ex:
             log.error(ex)
        error_warning = val if (val:=sio.getvalue()) else None
        if not mol_std:
            log.info(f'failed to standardise mol with the error/warning {error_warning}')
            mol_std = None

        mol_smiles = Chem.MolToSmiles(mol)
        mol_std_smiles = Chem.MolToSmiles(mol_std)
        if (mol_smiles != mol_std_smiles):
            log.info(f'standardised the molecule from {mol_smiles} to {mol_std_smiles}')


    return (mol_std, error_warning)
#
# from rdkit import Chem
# i_mol = 0
# mol = mols[i_mol]
# mol = Chem.AddHs(mol)
# [atom.GetSymbol() for atom in mol.GetAtoms()]


def check_mol(mol: Chem.Mol, ops: dict=None)  -> bool:
    """
    Checks the RDKit molecule. Exceptions are logged and not raised, instead this function returns False.
    This function is used when running the developed models on a new dataset to ensure the molecules do not produce
    node and edge features that are not supported by the model.
    :param mol:
    :param ops: applies the checker operations, for now it supports
                'allowed_atoms': ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'B', 'Si', 'I', 'H'] -> checks that only specified atoms are present
                'min_num_carbon_atoms': 1 -> checks that the structure contains at least so many carbon atoms
                'min_num_bonds': 1 -> checks that the structure contains at least so many bonds
                'max_num_fragments': 1 -> checks that the structure does not contain more than the maximum number of fragments
                'allowed_bonds': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'] -> checks that only specified bonds are present
                'molecular_weight': {'min': 0, 'max': 1000} -> checks that the molecular weight is within the specified range
                'max_number_rings': 5 -> checks that the molecule does not have more than the maximum number of rings
                'allowed_hybridisations': ['UNSPECIFIED', 'SP2', 'SP3', 'SP'] -> checks that only specified hybridisations are present
    :return: True if none of the checker operations fails, and False otherwise
    """
    if ops is None:
        ops = {'allowed_atoms': ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'B', 'Si', 'I', 'H'],
               'min_num_carbon_atoms': 1,
               'min_num_bonds': 1,
               'max_num_fragments': 1,
               'allowed_bonds': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
               'molecular_weight': {'min': 0, 'max': 1000},
               'max_number_rings': 5,
               'allowed_hybridisations': ['UNSPECIFIED', 'SP2', 'SP3', 'SP']
               }

    try:
        # check the allowed atoms
        if 'allowed_atoms' in ops:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ops['allowed_atoms']:
                    log.info(f'atom {atom.GetSymbol()} not in the allowed atoms {ops["allowed_atoms"]}')
                    return False

        # check the minimum number of carbon atoms
        if 'min_num_carbon_atoms' in ops:
            num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            if num_carbons < ops['min_num_carbon_atoms']:
                log.info(f'number of carbon atoms {num_carbons} is less than the minimum allowed {ops["min_num_carbon_atoms"]}')
                return False

        # check the minimum number of bonds
        if 'min_num_bonds' in ops:
            num_bonds = mol.GetNumBonds()
            if num_bonds < ops['min_num_bonds']:
                log.info(f'number of bonds {num_bonds} is less than the minimum allowed {ops["min_num_bonds"]}')
                return False

        # check the maximum number of fragments
        if 'max_num_fragments' in ops:
            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > ops['max_num_fragments']:
                log.info(f'number of fragments {len(frags)} exceeds the maximum allowed {ops["max_num_fragments"]}')
                return False

        # check the allowed bonds
        if 'allowed_bonds' in ops:
            for bond in mol.GetBonds():
                if bond.GetBondType().name not in ops['allowed_bonds']:
                    print(bond.GetBondType().name)
                    log.info(f'bond {bond.GetBondType().name} not in the allowed bonds {ops["allowed_bonds"]}')
                    return False

        # check the molecular weight
        if 'molecular_weight' in ops:
            mol_weight = Descriptors.MolWt(mol)
            if not ops['molecular_weight']['min'] <= mol_weight <= ops['molecular_weight']['max']:
                log.info(f'molecular weight {mol_weight} not in the allowed range {ops["molecular_weight"]}')
                return False

        # check the maximum number of rings
        if 'max_number_rings' in ops:
            ring_info = mol.GetRingInfo()
            if ring_info.NumRings() > ops['max_number_rings']:
                log.info(f'number of rings {ring_info.NumRings()} exceeds the maximum allowed {ops["max_number_rings"]}')
                return False

        # check the allowed hybridisations
        if 'allowed_hybridisations' in ops:
            for atom in mol.GetAtoms():
                if atom.GetHybridization().name not in ops['allowed_hybridisations']:
                    log.info(f'atom {atom.GetHybridization().name} not in the allowed hybridisations {ops["allowed_hybridisations"]}')
                    return False

    except Exception as ex:
        log.error(ex)
        return False

    return True

def normalise_mol(mol: Chem.Mol, tfs: str= None) -> tuple[Chem.Mol, list[str]]:
    """
    Normalises the RDKit molecule. Applies a series of standard transformations to correct functional
    groups and recombine charges. Exceptions are logged and not raised, instead the returned mol object is None.
    The implementation is based on the rdkit blog
    https://greglandrum.github.io/rdkit-blog/posts/2024-02-23-custom-transformations-and-logging.html
    :param mol: input molecule
    :param tfs: list of normalisation transformations to apply, for now it supports
                'None' -> applies the rdMolStandardize.Normalise operation, or (example)
                tfs = '''
                // this should go last, because later transformations will
                // lose the alkali metal
                disconnect_alkali_metals\t[Li,Na,K,Rb:1]-[A:2]>>([*+:1].[*-:2])
                ''' -> applies one transformation to disconnect covalently bonded alkali metals
    :return: mol object with normalised features and name of applied normalisations
    """

    with Rdkit_operation() as sio:
        # create the new Normalizer:
        if tfs:
            cps = rdMolStandardize.CleanupParameters()
            nrm = rdMolStandardize.NormalizerFromData(tfs, cps)
        else:
            nrm = rdMolStandardize.Normalizer()

        match_expr = re.compile(r'Rule applied: (.*?)\n')

        mol_norm = nrm.normalize(mol)
        text = val if (val:=sio.getvalue()) else None
        tfs_applied = match_expr.findall(text)

        mol_smiles = Chem.MolToSmiles(mol)
        mol_norm_smiles = Chem.MolToSmiles(mol_norm)
        if (mol_smiles != mol_norm_smiles):
            log.info(f'normalised the molecule from {mol_smiles} to {mol_norm_smiles}')

    return mol_norm, tfs_applied if tfs_applied else []

def remove_fragments(mol: Chem.Mol, frags_to_remove: list[str]=None) -> tuple[Chem.Mol, list[str]]:
    """
    Removes the specified fragments from the molecule
    :param mol: input molecule
    :param frags: list of fragments to remove (as SMARTS), if None a standard set of fragments is removed
    :return: tuple with mol object with the specified fragments removed, and list with fragments removed (can be empty)
    """
    if frags_to_remove is None:
        frags_to_remove = [r'[H,H+]', r'[Na,Na+1]', r'[K,K+1]',
                           r'[F,F-1]', r'[Cl,Cl-1]', r'[Br,Br-1]', r'[I,I-1]',
                           r'[O,O-2]',
                           r'O=S(=O)([O,O-1])[O,O-1]',
                           r'[NH4+,NX3H3,NX0]',
                           r'[Ca,Ca+2]', r'[Mg,Mg+2]',
                           r'[OH-]',
                           r'O=[N+]([O,O-1])[O,O-1]',
                           r'CC(=O)[O,O-1]',
                           r'C(=O)[O,O-1]',
                           r'O=C([O,O-1])C(=O)[O,O-1]',
                           r'O=C([O,O-1])[O,O-1]',
                           r'O=P([O,O-1])([O,O-1])[O,O-1]'
                           ]
    frags_to_remove = [Chem.MolFromSmarts(frag) for frag in frags_to_remove]

    frags = Chem.GetMolFrags(mol, asMols=True)
    frags_kept = []
    frags_removed = []
    for frag in frags:
        removed = False
        for frag_to_remove in frags_to_remove:
            matches = frag.GetSubstructMatches(frag_to_remove)
            full_match = any(len(match) == frag.GetNumAtoms() for match in matches)
            if full_match:
                print(Chem.MolToSmarts(frag_to_remove))
                frags_removed.append(frag)
                removed = True
                break
        if not removed:
            frags_kept.append(frag)
    if frags_removed:
        log.info(f"removed the {', '.join([Chem.MolToSmiles(frag) for frag in frags_removed])} fragment(s) from the structure")
    if len(frags_kept) > 1:
        mol = frags_kept[0]
        for frag in frags_kept[1:]:
            mol = Chem.CombineMols(mol, frag)
    elif len(frags_kept) == 0:
        mol = None
    else:
        mol = frags_kept[0]
    return mol, [Chem.MolToSmiles(frag) for frag in frags_removed]