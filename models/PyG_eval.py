import numpy as np

import logger
log = logger.get_logger(__name__)

import pandas as pd

from collections import Counter

import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader

from rdkit import Chem
from tqdm import tqdm
from cheminformatics.rdkit_toolkit import get_adjacency_info, get_node_features, get_edge_features
import torch.nn.functional as F

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(mols: list[Chem.Mol],
         net,
         tasks: list[str],
         node_feats: list[str],
         edge_feats: list[str],
         node_feature_names: list[str],
         edge_attr_names: list[str],
         embeddings: bool = False) -> pd.DataFrame:
    '''
    Generate predictions for a list of molecules using a trained model. This function does not examine if
    the molecules are suitable for running the model. It also does not standardise the molecular structures. These
    operations need to be performed before calling this function. The last two arguments are needed to ensure
    that the PyG model is fed with tensors of the same size as the training data. For example, if the molecules
    to run predictions for do not contain nitrogen, but the training set had molecules with nitrogen, the node feature
    matrix will contain a column with the one-hot-embedding for nitrogen before running the predictions.
    :param mols: list of RDKit molecules to run predictions or
    :param net: trained PyG model
    :param tasks: list with task names that the model was trained to predict
    :param node_feats: list of node features to generate
    :param edge_feats: list of edge features to generate
    :param node_feature_names: column names in the node feature matrix for the molecules in the training set
    :param edge_attr_names: column names in edge feature matrix for the molecules in the training set
    :param embeddings: if True then the embeddings are returned, otherwise the genotoxicity positive and negative probabilities
    :return: pandas dataframe with the predictions or embeddings
    '''
    # collect the adjacency information, the node features, the edge features and the assay_results
    adjacency_info = []
    node_features = []
    edge_features = []
    # assay_results = []
    for i_mol, mol in tqdm(enumerate(mols), total=len(mols)):
        # compute the adjacency information, the node features and the edge features
        adjacency_info.append(get_adjacency_info(mol))
        node_features.append(get_node_features(mol, feats=node_feats))
        edge_features.append(get_edge_features(mol, feats=edge_feats))
        # assay_results.append(pd.DataFrame(json.loads(mols[i_mol].GetProp('assay_data'))))

    # categorical node features and their counts
    tmp = pd.concat(node_features, axis='index', ignore_index=True)
    one_hot_encode_node_cols = tmp.select_dtypes(include=['object', 'int']).columns.to_list()
    one_hot_encode_node_cols = {col: dict(Counter(tmp[col].to_list()).most_common()) for col in
                                one_hot_encode_node_cols}
    for col in one_hot_encode_node_cols:
        log.info(f'categorical node feature {col} counts: {one_hot_encode_node_cols[col]}')

    # categorical edge features and their counts
    tmp = pd.concat(edge_features, axis='index', ignore_index=True)
    one_hot_encode_edge_cols = tmp.select_dtypes(include=['object', 'int']).columns.to_list()
    one_hot_encode_edge_cols = {col: dict(Counter(tmp[col].to_list()).most_common()) for col in
                                one_hot_encode_edge_cols}
    for col in one_hot_encode_edge_cols:
        log.info(f'categorical edge feature {col} counts: {one_hot_encode_edge_cols[col]}')

    data_list = []
    for i_mol in range(len(mols)):

        # adjacency information
        edge_index = torch.tensor(adjacency_info[i_mol].to_numpy()).to(torch.long)

        # node features
        # .. categorical node features (type string and int)
        x = pd.DataFrame()
        prefix_sep = '_'
        for key in one_hot_encode_node_cols.keys():
            all_cols = [key + prefix_sep + str(val) for val in list(one_hot_encode_node_cols[key].keys())]
            x = pd.concat([x,
                           pd.get_dummies(node_features[i_mol], prefix_sep='_', columns=[key]).reindex(all_cols,
                                                                                                       axis='columns')
                           ], axis='columns')
        # .. numerical node features (type float)
        x = pd.concat([x, node_features[i_mol].drop(one_hot_encode_node_cols.keys(), axis='columns')], axis='columns')
        x = x.reindex(node_feature_names, axis='columns')
        x = x.astype('float32').fillna(0.)
        x = torch.tensor(x.to_numpy(), dtype=torch.float)

        # edge features
        # .. categorical edge features (type string and int)
        edge_attr = pd.DataFrame()
        prefix_sep = '_'
        for key in one_hot_encode_edge_cols.keys():
            all_cols = [key + prefix_sep + str(val) for val in list(one_hot_encode_edge_cols[key].keys())]
            edge_attr = pd.concat([edge_attr,
                                   pd.get_dummies(edge_features[i_mol], prefix_sep='_', columns=[key]).reindex(all_cols,
                                                                                                               axis='columns')
                                   ], axis='columns')
        # .. numerical edge features (type float)
        edge_attr = pd.concat([edge_attr, edge_features[i_mol].drop(one_hot_encode_edge_cols.keys(), axis='columns')],
                              axis='columns')
        edge_attr = edge_attr.reindex(edge_attr_names, axis='columns')
        edge_attr = edge_attr.astype('float32').fillna(0.)
        edge_attr = torch.tensor(edge_attr.to_numpy(), dtype=torch.float)

        # create the data object and set the molecule ID for ease of use (not necessary for the model)
        # we set the assay data as a property of the data object and leave y to be None at this stage
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=None,
                    assay_data=None,
                    molecule_id=None,
                    )

        data_list.append(data)
    batch_size = 1024
    prediction_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, drop_last=False)

    net.eval()

    if not embeddings:
        # generate genotoxocity positive and negative probabilities for each task
        predictions = []
        i_mol_start = 0
        for i_batch, batch in tqdm(enumerate(prediction_loader)):
            batch.to(device)
            for i_task, task in enumerate(tasks):
                with torch.no_grad():
                    pred = net(batch.x, batch.edge_index, batch.edge_attr, batch.batch, task_id=i_task)
                pred = pd.DataFrame(F.softmax(pred, dim=1).detach().cpu().numpy(),
                                    columns=['negative (probability)', 'positive (probability)'])
                pred['genotoxicity call'] = np.where(pred['positive (probability)'] >= 0.5, 'positive', 'negative')
                pred['task'] = task
                pred['i mol'] = range(i_mol_start, i_mol_start + len(batch))
                predictions.append(pred)
            i_mol_start = i_mol_start + len(batch)
        predictions = pd.concat(predictions, axis='index', ignore_index=True, sort=False)
        return predictions
    else:
        # generate embeddings for each molecule
        embeddings = []
        i_mol_start = 0
        for i_batch, batch in tqdm(enumerate(prediction_loader)):
            batch.to(device)
            with torch.no_grad():
                embedding = net(batch.x, batch.edge_index, batch.edge_attr, batch.batch, task_id=0, embeddings=True)
            embedding = pd.DataFrame(embedding.detach().cpu().numpy(), columns=[f'embedding_{i}' for i in range(embedding.shape[1])])
            embedding['i mol'] = range(i_mol_start, i_mol_start + len(batch))
            embeddings.append(embedding)
            i_mol_start = i_mol_start + len(batch)
        embeddings = pd.concat(embeddings, axis='index', sort=False, ignore_index=True)
        return embeddings
