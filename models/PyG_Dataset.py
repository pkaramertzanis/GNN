import logger
log = logger.get_logger(__name__)

from collections import Counter
import pandas as pd
from tqdm import tqdm
import json

from cheminformatics.rdkit_toolkit import get_node_features, get_edge_features, get_adjacency_info
from cheminformatics.rdkit_toolkit import read_sdf, check_mol, standardise_mol

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class PyG_Dataset(InMemoryDataset):
    '''
    PyTorch Geometric dataset class. Modified from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
    to allow setting the following parameters in the initialisation:
    - root: location where the input sdf file can be found
    - task: the task to model
    - ambiguous_outcomes: 'ignore' (default), 'set_negative', 'set_positive' to handle ambiguous outcomes
    - force_reload: boolean to force re-processing of the sdf file even if PyTorch Geometric can find the processed file 'data.pt'
      in the root folder
    - node_feats: list of node features to be used in the graph
    - edge_feats: list of edge features to be used in the graph

    The sdf file should be in the expected format that is ensured by the data.combine.create_sdf function. In particular, the mol
    block should contain the field genotoxocity, an example of which is

    [{"smiles_std":"C#CC1(O)CCC2C3CCc4cc(O)ccc4C3CCC21C", "task aggregation":"in vitro\/in vivo, endpoint, assay",
      "task":"in vitro,.. aberration test", "CAS number":"57-63-6",
      "source record ID":"QSAR Toolbox, Genotoxicity & Carcinogenicity ECVAM database 89339",
      "genotoxicity":"ambiguous"},

      {"smiles_std":"C#CC1(O)CCC2C3CCc4cc(O)ccc4C3CCC21C", "task aggregation":"in vitro\/in vivo, endpoint, assay",
       "task":"in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay", "CAS number":"57-63-6",
       "source record ID":"Hansen 2009 4416, QSAR Toolbox, Bacterial mutagenicity ISSSTY database 27486",
       "genotoxicity":"negative"}]

    As the example shows, the same mol block could contribute to more than one task. The task to model is set in the task parameter.
    Genotoxicity can only be "positive", "negative" and "ambiguous".
    '''
    def __init__(self, root,
                 task,
                 node_feats,
                 edge_feats,
                 transform=None, pre_transform=None, pre_filter=None,
                 ambiguous_outcomes='ignore',
                 force_reload=False
                 ):

        self.task = task
        self.ambiguous_outcomes = ambiguous_outcomes

        # set the node and edge features
        self.node_feats = node_feats
        log.info('node features used: ' + str(self.node_feats))
        self.edge_feats = edge_feats
        log.info('edge features used: ' + str(self.edge_feats))

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

        # Read data into Data list
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
            tmp = mols[i_mol].GetProp('genotoxicity')
            tmp = json.loads(tmp)
            tmp = pd.DataFrame(tmp)
            msk = tmp['task'] == self.task

            if msk.sum() == 0:
                log.info(f'skipping molecule {i_mol} because of missing task {self.task}')
                continue
            assay_data = tmp.loc[msk, 'genotoxicity'].iloc[0]
            molecule_ID = tmp.loc[msk, 'source record ID'].iloc[0]

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


