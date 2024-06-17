import logger
log = logger.get_logger(__name__)

from collections import Counter
import pandas as pd
from tqdm import tqdm
import json

from cheminformatics.rdkit_toolkit import get_node_features, get_edge_features, get_adjacency_info
from cheminformatics.rdkit_toolkit import read_sdf, standardise_mol

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class PyG_Dataset(InMemoryDataset):
    '''
    PyTorch Geometric dataset class. Modified from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
    to allow setting the following parameters in the initialisation:
    - root: location where the input sdf file can be found
    - target_level: "genotoxicity_assay_level" or "genotoxicity_endpoint_level" corresponding to the assay (most granular) or endpoint level (least granular)
    - target_assay_endpoint: the target assay or endpoint name
    - ambiguous_outcomes: 'ignore' (default), 'set_negative', 'set_positive' to handle ambiguous outcomes
    - force_reload: boolean to force re-processing of the sdf file even if PyTorch Geometric can find the processed file 'data.pt'
      in the root folder
      - node_feats: list of node features to be used in the graph, for now it supports 'atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization'
      - edge_feats: list of edge features to be used in the graph, for now it supports 'bond_type'
    - checker_ops: dictionary with checker operations
    - standardiser_ops: list of standardiser operations
    '''
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                 target_level=None, target_assay_endpoint=None, ambiguous_outcomes='ignore',
                 force_reload=False,
                 node_feats=['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization'],
                 edge_feats=['bond_type'],
                 checker_ops={'allowed_atoms': ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'B','Si', 'I', 'H']},
                 standardiser_ops=['cleanup', 'addHs']
                 ):

        # set the target assay or endpoint
        self.target_level = target_level
        self.target_assay_endpoint = target_assay_endpoint
        self.ambiguous_outcomes = ambiguous_outcomes

        # set the node and edge features
        self.node_feats = node_feats
        log.info('node features used: ' + str(self.node_feats))
        self.edge_feats = edge_feats
        log.info('edge features used: ' + str(self.edge_feats))

        # set the checker and standardiser operations
        self.checker_ops = checker_ops
        log.info('checker operations: ' + str(self.checker_ops))
        self.standardiser_ops = standardiser_ops
        log.info('standardiser operations: ' + str(self.standardiser_ops))

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        '''
        Returns the names of the raw file. Download is not implemented, so the raw file must be present. The convention
        is that the raw dataset is a single sdf file at the location self.root/../sdf
        :return:
        '''
        sdf_folder = self.root.parent / 'sdf'
        sdf_files = list(sdf_folder.glob('*.sdf'))
        try:
            if len(sdf_files) == 1:
                log.info('PyTorch-Geometric raw file to be used is ' + str(sdf_files[0]))
                return sdf_files
            else:
                raise IOError(f'There must be exactly one sdf file in the folder {sdf_folder}')
        except IOError as ex:
            log.error(ex)
            raise

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        '''Not implemented, a single raw sdf file must be present in folder  self.root/../sdf'''
        pass

    def process(self):
        '''
        Processes the raw data from the sdp file to prepare a PyTorch Geometric list of Data objects
        '''

        # read the sdf file with the raw data
        mols = read_sdf(self.raw_file_names[0])

        # check and standarddise molecules, some molecules will be removed
        mols_std = []
        for i_mol in range(len(mols)):
            # filter out molecules with no edges, this is a requirement for using graphs and is applied by default
            if not mols[i_mol].GetNumBonds():
                log.info(f'skipping molecule {i_mol} because it has no bonds')
                continue
            # filter out molecules with rare atoms
            if 'allowed_atoms' in self.checker_ops:
                not_allowed_atoms = [atom.GetSymbol() for atom in mols[i_mol].GetAtoms() if atom.GetSymbol() not in self.checker_ops['allowed_atoms']]
                if not_allowed_atoms:
                    log.info(f'skipping molecule {i_mol} because it contains not allowed atoms: {dict(Counter(not_allowed_atoms))}')
                    continue
            # standardise the molecule
            mol_std, info_error_warning = standardise_mol(mols[i_mol], ops=self.standardiser_ops)
            if mol_std is not None:
                mols_std.append(mol_std)
            else:
                log.info(f'skipping molecule {i_mol} because it could not be standardised ({info_error_warning})')
                continue
        log.info(f'following checking and standardisation {len(mols_std)} molecules remain out of the starting {len(mols)} molecules')
        mols = mols_std


        # collect the adjacency information, the node features, the edge features and the assay_results
        adjacency_info = []
        node_features = []
        edge_features = []
        # assay_results = []
        for i_mol, mol in tqdm(enumerate(mols), total=len(mols)):
            # compute the adjacency information, the node features and the edge features
            adjacency_info.append(get_adjacency_info(mol))
            node_features.append(get_node_features(mol, feats=self.node_feats))
            edge_features.append(get_edge_features(mol, feats=self.edge_feats))
            # assay_results.append(pd.DataFrame(json.loads(mols[i_mol].GetProp('assay_data'))))

        # categorical node features and their counts
        tmp = pd.concat(node_features, axis='index', ignore_index=True)
        one_hot_encode_node_cols = tmp.select_dtypes(include=['object', 'int']).columns.to_list()
        one_hot_encode_node_cols = {col: dict(Counter(tmp[col].to_list()).most_common()) for col in one_hot_encode_node_cols}
        for col in one_hot_encode_node_cols:
            log.info(f'categorical node feature {col} counts: {one_hot_encode_node_cols[col]}')

        # categorical edge features and their counts
        tmp = pd.concat(edge_features, axis='index', ignore_index=True)
        one_hot_encode_edge_cols = tmp.select_dtypes(include=['object', 'int']).columns.to_list()
        one_hot_encode_edge_cols = {col: dict(Counter(tmp[col].to_list()).most_common()) for col in one_hot_encode_edge_cols}
        for col in one_hot_encode_edge_cols:
            log.info(f'categorical edge feature {col} counts: {one_hot_encode_edge_cols[col]}')

        # Read data into huge `Data` list.
        data_list = []
        for i_mol in range(len(mols)):

            # adjacency information
            edge_index = torch.tensor(adjacency_info[i_mol].to_numpy()).to(torch.long)

            # node features
            # .. categorical node features (type string and int)
            x = pd.DataFrame()
            prefix_sep = '_'
            for key in one_hot_encode_node_cols.keys():
                all_cols = [key+prefix_sep+str(val) for val in list(one_hot_encode_node_cols[key].keys())]
                x = pd.concat([x,
                                   pd.get_dummies(node_features[i_mol], prefix_sep='_', columns=[key]).reindex(all_cols, axis='columns')
                                   ], axis='columns')
            # .. numerical node features (type float)
            x = pd.concat([x, node_features[i_mol].drop(one_hot_encode_node_cols.keys(), axis='columns')], axis='columns')
            x = x.astype('float32').fillna(0.)
            x = torch.tensor(x.to_numpy(), dtype=torch.float)

            # edge features
            # .. categorical edge features (type string and int)
            edge_attr = pd.DataFrame()
            prefix_sep = '_'
            for key in one_hot_encode_edge_cols.keys():
                all_cols = [key+prefix_sep+str(val) for val in list(one_hot_encode_edge_cols[key].keys())]
                edge_attr = pd.concat([edge_attr,
                                   pd.get_dummies(edge_features[i_mol], prefix_sep='_', columns=[key]).reindex(all_cols, axis='columns')
                                   ], axis='columns')
            # .. numerical edge features (type float)
            edge_attr = pd.concat([edge_attr, edge_features[i_mol].drop(one_hot_encode_edge_cols.keys(), axis='columns')], axis='columns')
            edge_attr = edge_attr.astype('float32').fillna(0.)
            edge_attr = torch.tensor(edge_attr.to_numpy(), dtype=torch.float)

            # obtain the genotoxicity outcome
            if self.target_level == 'genotoxicity_assay_level':
                tmp = mols[i_mol].GetProp(self.target_level)
                tmp = json.loads(tmp)
                tmp = pd.DataFrame(tmp)
                msk = tmp['assay'] == self.target_assay_endpoint
                assay_data = tmp.loc[msk, 'genotoxicity'].iloc[0]
                molecule_ID = tmp.loc[msk, 'molecule ID'].iloc[0]
            elif self.target_level == 'genotoxicity_endpoint_level':
                tmp = mols[i_mol].GetProp(self.target_level)
                tmp = json.loads(tmp)
                tmp = pd.DataFrame(tmp)
                msk = tmp['endpoint'] == self.target_assay_endpoint
                assay_data = tmp.loc[msk, 'genotoxicity'].iloc[0]
                molecule_ID = tmp.loc[msk, 'molecule ID'].iloc[0]
            else:
                ex = ValueError(f'target_level can only be "genotoxicity_assay_level", or "genotoxicity_endpoint_level", but was set to {self.target_level}')
                log.error(ex)
                raise ex
            if assay_data == 'ambiguous':
                if self.ambiguous_outcomes == 'ignore':
                    log.info(f'skipping molecule {i_mol} because of ambiguous outcome')
                    continue
                elif self.ambiguous_outcomes == 'set_negative':
                    assay_data = 'negative'
                elif self.ambiguous_outcomes == 'set_positive':
                    assay_data = 'positive'
                else:
                    ex = ValueError(f'ambiguous_outcomes can only be "ignore", "set_negative", or "set_positive", but was set to {self.ambiguous_outcomes}')
                    log.error(ex)
                    raise ex
            elif assay_data == 'not available':
                continue

            # create the data object and set the molecule ID for ease of use (not necessary for the model)
            # we set the assay data as a property of the data object and leave y to be None at this stage
            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=None,
                        assay_data=assay_data,
                        molecule_id=molecule_ID,
                        )

            data_list.append(data)
        log.info(f'added {len(data_list)} structures in the dataset')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])


