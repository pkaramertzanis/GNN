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
from rdkit.Chem import rdDepictor
from rdkit.Chem import AllChem

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
    Reads an sdf file and returns a list of molecule objects. It does not do sanitisation or removal of Hs. This should
    be done explicitly if needed.
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


def remove_stereo(mol: Chem.Mol, stereo_types=None, set_coord_zero=False) -> Chem.Mol:
    '''
    Utility function to remove stereochemistry from a molecule, including cis/trans and R/S stereochemistry.
    :param mol: input molecule
    :param stereo_types: list of stereochemistry types to remove, for now it supports 'cis/trans' and 'R/S'; if not specified
    :param set_coord_zero: if True, the atomic coordinates are set to zero
    all stereoisomerism is removed (this includes ring stereochemistry)
    :return: molecule with
    '''

    if stereo_types is not None:
        for stereo_type in stereo_types:
            if stereo_type not in ['cis/trans', 'R/S']:
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
    else:
        Chem.RemoveStereochemistry(mol)

    # regenerate computed properties like implicit valence and ring information in case it matters
    # Chem.SanitizeMol(mol)
    mol.UpdatePropertyCache(strict=False)

    # set the coordinates to zero so that we do not leave evidence of stereoisomerism in the structure
    if set_coord_zero:
        mol = Chem.AddHs(mol)
        # .. compute the atomic coordinates
        #  enforce chirality, do not use basic knowledge, torsion preferences and use random cooridnates eliminates failures and we do not need the stereochemistry as we will set the coordinates to zero
        AllChem.EmbedMolecule(mol, enforceChirality=False, useRandomCoords=True, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (0.0, 0.0, 0.0))
        mol = Chem.RemoveHs(mol)

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


def apply_reaction(smirks: str, mol: Chem.Mol, maxIterations=5) -> tuple[Chem.Mol, Union[str, None]]:
    '''
    Applies a reaction SMIRKS to a molecule and returns the product. It is important to have the aromatised molecular
    structure otherwise we may destroy aromatic rings
    :param smirks: the reaction SMIRKS
    :param mol: the molecule
    :param maxIterations: maximum number of reaction steps (do not use recursion for stability)
    :return: the reaction product and a message explaining how many times the reaction was applied (can be None)
    '''
    rxn = AllChem.ReactionFromSmarts(smirks)
    # add explicit hydrogen atoms as there are needed by the reaction SMIRKS
    reactant = mol
    for atom in reactant.GetAtoms():
        implicit_valence = atom.GetImplicitValence()
    reactant = Chem.AddHs(reactant)
    Chem.SanitizeMol(reactant)

    for i_reaction in range(maxIterations):
        products = rxn.RunReactants((reactant,), maxProducts=1)
        if len(products) == 0:
            product = reactant
            break
        else:
            reactant = products[0][0]
            Chem.SanitizeMol(reactant) # sanitisation may be needed, see https://www.rdkit.org/docs/GettingStartedInPython.html#chemical-reactions
    product = Chem.RemoveHs(product)
    Chem.SanitizeMol(product)
    return product, f'reaction "{smirks}" applied {i_reaction} time(s)' if i_reaction else None

def apply_tautomerisation_transformations(mol: Chem.Mol) -> tuple[Chem.Mol, list[str]]:
    '''
    Applies the tautomerisation transformations to a molecule. The transformations are based on the RDKit blog. There is no
    way for the function to fail. If the molecule is not transformed, the original molecule is returned.
    :param mol: input molecule
    :return: tuple with the transformed molecule, a message with the list of transformations applied and a message with errors
    '''

    # transformation reactions taken from https://hub.knime.com/kmansouri/spaces/Public/QSAR-ready_2.5.8~5TRvnGfMJsgTkcZu/current-state
    # the original RXN SMARTS is commented out and replaced with the RDKit SMARTS that contains the atom mapping, including for hydrogens
    tautomerism_reactions = [
        # nitro group
        r'[*:4]-[n:3](:[o:2]):[o:1]>>[#8-:2]-[#7+:3](-[*:4])=[O:1]',  # r'[*]N(:O):O>>[O-][N+]([*])=O',
        r'[H][#8:1]-[#7:3](-[*:4])=[O:2]>>[#8-:1]-[#7+:3](-[*:4])=[O:2]',  # r'[H]ON([*])=O>>[O-][N+]([*])=O',
        r'[*:4][N:3](=[O:2])=[O:1]>>[#8-:1]-[#7+:3](-[*:4])=[O:2]',  # r'[*]N(=O)=O>>[O-][N+]([*])=O',
        r'[H][#8:1]-[#7:3](-[*:4])-[#8:2][H]>>[#8-:1]-[#7+:3](-[*:4])=[O:2]',  # r'[H]ON(O)[*]>>[O-][N+]([*])=O',
        r'[H][N:3]([*:4])(=[O:2])=[O:1]>>[#8-:2]-[#7+:3](-[*:4])=[O:1]',  # r'[H][N]([*])(=O)=O>>[O-][N+]([*])=O',

        # imine - enamine, the nitrogen is not allowed to be in a three member ring (aziridine), removed cis/trans isomerism from the reaction
        r'[H:8][#7;!r3:2](-[$([#1,*]):1])-[#6:3](-[$([#1,*]):7])=[#6:4](-[$([#1,*]):5])-[$([#1,*]):6]>>[H:8][C:4]([$([#1,*]):6])([$([#1,*]):5])[#6:3](-[$([#1,*]):7])=[#7:2]-[$([#1,*]):1]', # r'[H][#7;!R0](-[$([#1,*])])-[#6;!R0](\[$([#1,*])])=[#6;!R0](\[$([#1,*])])-[$([#1,*])]>>[$([#1,*])]\[#6](=[#7]\[$([#1,*])])-[#6](-[$([#1,*])])-[$([#1,*])]',

        # third transformation block
        r'[H:7][#7:2](-[*,#1:1])-[#7:3]=[#6:4](-[*,#1:5])-[*,#1:6]>>[H:7][C:4]([*,#1:5])([*,#1:6])[#7:3]=[#7:2]-[*,#1:1]', # r'[H][#7](-[$([#1,*])])\[#7]=[#6](\[$([#1,*])])-[$([#1,*])]>>[$([#1,*])]-[#6](-[$([#1,*])])\[#7]=[#7]\[$([#1,*])]',
        r'[H:7][#8:2]-[#6:1](-[*,#1:6])=[#6:5](-[*,#1:4])-[*,#1:3]>>[H:7][C:5]([*,#1:3])([*,#1:4])[#6:1](-[*,#1:6])=[O:2]', # r'[H][#8]\[#6](-[$([#1,*])])=[#6](/[$([#1,*])])-[$([#1,*])]>>[$([#1,*])]-[#6](-[$([#1,*])])-[#6](-[$([#1,*])])=O',
        r'[H:4][#8:1][C:2]#[*,#1:3]>>[H:4][*,#1:3]=[C:2]=[O:1]',  # r'[H][#8]C#[$([#1,*])]>>O=C=[$([#1,*])]',
        r'[H:8][#8:6]-[#6:3](-[#8:4][H:7])-[#6:2](-[*,#1:1])=[O:5]>>[H:8][#8:5]-[#6:2](-[*,#1:1])-[#6:3](=[O:6])-[#8:4][H:7]', # r'[H][#8]-[#6](-[#8][H])-[#6](-[$([#1,*])])=O>>[#8]-[#6](-[$([#1,*])])-[#6](-[#8])=O',
        r'[*,#1:1]-[#7-:2][N+:3]#[N:4]>>[*,#1:1]-[#7:2]=[N+:3]=[#7-:4]',  # r'[$([#1,*])]-[#7-][N+]#N>>[$([#1,*])]-[#7]=[N+]=[#7-]',

        # fourth transformation block
        r'[H:15][C:3]1([#6:7](-[*,#1:9])=[O:8])[#6:2](-[*,#1:10])=[#7:1]-[#6:6](-[*,#1:11])=[#6:5](-[*,#1:12])[C:4]1([*,#1:13])[*,#1:14]>>[H:15][#7:1]-1-[#6:6](-[*,#1:11])=[#6:5](-[*,#1:12])[C:4]([*,#1:13])([*,#1:14])[#6:3](-[#6:7](-[*,#1:9])=[O:8])=[#6:2]-1-[*,#1:10]', # r'[H]C1([#6](-[*,#1])=O)[#6](-[*,#1])=[#7]-[#6](-[*,#1])=[#6](-[*,#1])C1([*,#1])[*,#1]>>[H][#7]-1-[#6](-[*,#1])=[#6](-[*,#1])C([*,#1])([*,#1])[#6](-[#6](-[*,#1])=O)=[#6]-1-[*,#1]',
        r'[H:6][C:3]([*,#1:4])([*,#1:5])[#7:2]=[O:1]>>[H:6][#8:1]\[#7:2]=[#6:3](\[*,#1:5])-[*,#1:4]', # r'[H]C([$([#1,*])])([$([#1,*])])[#7]=O>>[H][#8]\[#7]=[#6](\[$([#1,*])])-[$([#1,*])]',
        r'[H:7][#8:4][S:3](=[O:5])([#8:6][H])=[#7:2]-[*,#1:1]>>[H:7][#8:4][S:3](=[O:6])(=[O:5])[#7:2]=[*,#1:1]', # r'[H][#8]S(=O)([#8][H])=[#7]-[$([#1,*])]>>[H][#8]S(=O)(=O)[#7]=[$([#1,*])]',
        r'[H][#7:2](-[*,#1:1])[N+:3]#[N:4]>>[*,#1:1]-[#7:2]=[N:3]#[N:4]', # r'[H][#7](-[$([#1,*])])[N+]#N>>[$([#1,*])]-[#7]=N#N',
        r'[H][#7:4]=[N+:3]=[#7:2]-[*,#1:1]>>[*,#1:1]-[#7:2]=[N:3]#[N:4]', # r'[H][#7]=[N+]=[#7]-[$([#1,*])]>>[$([#1,*])]-[#7]=N#N',
        r'[H][N:1]([H])([*,#1:2])([*,#1:4])[*,#1:3]>>[*,#1:2]-[#7:1](-[*,#1:4])-[*,#1:3]', # r'[H][N]([H])([$([#1,*])])([$([#1,*])])[$([#1,*])]>>[$([#1,*])]-[#7](-[$([#1,*])])-[$([#1,*])]',
        r'[H][N:1]([H])([*,#1:2])=[*,#1:3]>>[*,#1:2]-[#7:1]=[*,#1:3]', # r'[H][N]([H])([$([#1,*])])=[$([#1,*])]>>[$([#1,*])]-[#7]=[$([#1,*])]',
        r'[H:4][N-:3]#[N+:2][*,#1:1]>>[H:4]\[#7:3]=[#7:2]/[*,#1:1]', # r'[H][N-]#[N+][$([#1,*])]>>[$([#1,*])]-[#7]=[#7]'
    ]

    messages = []
    with Rdkit_operation() as sio:
        try:
            product = mol
            for smirks in tautomerism_reactions:
                product, message = apply_reaction(smirks, product)
                if message:
                    messages.append(message)
            if messages:
                messages = ', '.join(messages)
            else:
                messages = None
            error = None
        except Exception as ex:
            log.error(ex)
            error = str(ex)
            product = mol
    return product, messages, error



def standardise_mol(mol: Chem.Mol, ops: list[str]=None)  -> tuple[Chem.Mol, str]:
    """
    Standardises the RDKit molecule. Exceptions are logged and not raised, instead the returned mol object is None.
    :param mol:
    :param ops: applies the standardisation operations in the order specified in the list, for now it supports
                'cleanup' -> applies the rdMolStandardize.Cleanup operation
                'uncharge' -> returns the uncharged molecule (applies only if only one fragment)
                'addHs' -> adds explicit hydrogen atoms
    :return: tuple with resulting mol and warnings/errors during the standardisation
    """
    if ops is None:
        ops = ['cleanup', 'uncharge', 'addHs', ]

    all_ops = ['cleanup', 'uncharge', 'addHs']
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
                'allowed_total_charge' -> checks that the total charge is one of the permitted values, typically 0
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
               'allowed_hybridisations': ['UNSPECIFIED', 'SP2', 'SP3', 'SP'],
               'allowed_total_charge': [0]
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

        # check the allowed total charge
        if 'allowed_total_charge' in ops:
            total_charge = Chem.GetFormalCharge(mol)
            if total_charge not in ops['allowed_total_charge']:
                log.info(f'total charge {total_charge} not in the allowed charges {ops["allowed_total_charge"]}')
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