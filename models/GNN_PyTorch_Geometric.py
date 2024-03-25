import logger
log = logger.get_logger(__name__)

from typing import Callable
from collections import Counter
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

from cheminformatics.rdkit_toolkit import get_node_features, get_edge_features, get_adjacency_info
from cheminformatics.rdkit_toolkit import read_sdf, standardise_mol

import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import NNConv, global_add_pool
from torch_geometric.data import Data, DataLoader

import matplotlib.pyplot as plt

class GNNDatasetPyG(InMemoryDataset):
    '''
    PyTorch Geometric dataset class. Modified from https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
    to allow setting the following parameters in the initialisation:
    - input_sdf: input sdf file
    - target_assay_name: target assay name, the result of which is expected to be found in
      the assay_data property of the sdf file, .e.g.
      [{"assay": "AMES", "assay_notes": "all strains, with and without metabolic activation", "assay_result": "negative"}]
    - force_reload: boolean to force re-processing of the sdf file even if PyTorch Geometric can find the processed file 'data.pt'
      in the root folder
      - node_feats: list of node features to be used in the graph, for now it supports 'atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization'
      - edge_feats: list of edge features to be used in the graph, for now it supports 'bond_type'
    - checker_ops: dictionary with checker operations
    - standardiser_ops: list of standardiser operations
    '''
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                 input_sdf=None, target_assay_name=None, force_reload=False,
                 node_feats=['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization'],
                 edge_feats=['bond_type'],
                 checker_ops={'allowed_atoms': ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'B','Si', 'I', 'H']},
                 standardiser_ops=['cleanup', 'addHs']
                 ):

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
        assay_results = []
        for i_mol, mol in tqdm(enumerate(mols), total=len(mols)):
            # compute the adjacency information, the node features and the edge features
            adjacency_info.append(get_adjacency_info(mol))
            node_features.append(get_node_features(mol, feats=self.node_feats))
            edge_features.append(get_edge_features(mol, feats=self.edge_feats))
            assay_results.append(pd.DataFrame(json.loads(mols[i_mol].GetProp('assay_data'))))

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

            # create the data object and set the CAS and smiles for ease of use (not necessary for the model)
            # we set the assay data as a property of the data object and leave y to be None at this stage
            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=None,
                        assay_data=mols[i_mol].GetProp('assay_data'),
                        smiles=mols[i_mol].GetProp('smiles'),
                        cas=mols[i_mol].GetProp('CAS'),
                        )

            data_list.append(data)
        log.info(f'added {len(data_list)} structures in the dataaset')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])




class GNNModelPyG(torch.nn.Module):
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 n_conv: int,
                 n_edge_NN: int,
                 n_conv_hidden: int,
                 n_lin: int,
                 n_lin_hidden: int,
                 dropout: float,
                 activation_function: Callable,
                 n_classes: int):
        """
        Initialises the PyTorch Geometric model
        :param num_node_features: number of node features
        :param num_edge_features: number of edge features
        :param n_conv: number of convolutional layers
        :param n_edge_NN: number of neurons in the edge FNN
        :param n_conv_hidden: number of hidden features in the convolutional layers
        :param n_lin: number of linear layers
        :param n_lin_hidden: number of hidden features in the linear layers
        :param dropout: dropout rate
        :param activation_function: PyTorch activation function, e.g. torch.nn.functional.relu or torch.nn.functional.leaky_relu
        :param n_classes: number of output classes
        """
        super().__init__()

        # general parameters
        self.dropout = dropout
        self.activation_function = activation_function
        self.n_classes = n_classes

        # convolutional layers' parameters
        self.n_conv = n_conv
        self.n_edge_NN = n_edge_NN
        self.n_conv_hidden = n_conv_hidden

        # linear layers' parameters
        self.n_lin = n_lin
        self.n_lin_hidden = n_lin_hidden

        # set up the convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        # first convolutional layer
        conv_net = nn.Sequential(nn.Linear(num_edge_features, self.n_edge_NN),
                                 nn.LeakyReLU(),
                                 nn.Linear(self.n_edge_NN, num_node_features * self.n_conv_hidden))
        self.conv_layers.append(NNConv(num_node_features, self.n_conv_hidden, conv_net))
        # remaining of the convolutional layers
        for i_conv in range(self.n_conv-1):
            conv_net = nn.Sequential(nn.Linear(num_edge_features, self.n_edge_NN),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.n_edge_NN, self.n_conv_hidden * self.n_conv_hidden))
            self.conv_layers.append(NNConv(self.n_conv_hidden, self.n_conv_hidden, conv_net))

        # set up the linear layers
        self.lin_layers = torch.nn.ModuleList()
        self.lin_layers.append(torch.nn.Linear(self.n_conv_hidden, self.n_lin_hidden))
        self.lin_layers.extend(
            [torch.nn.Linear(self.n_lin_hidden, self.n_lin_hidden)
             for i in range(self.n_lin-1)])

        # set up the output layer
        self.out_layer = nn.Linear(self.n_lin_hidden, self.n_classes)

        # dropout layer
        self.dropout = torch.nn.Dropout(0.4)

        # initialise the weights and biases
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.)

    def forward(self, data):
        batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr

        # apply the convolutional layers
        for module in self.conv_layers:
            x = self.activation_function(module(x, edge_index, edge_attr))
            x = self.dropout(x)

        # pooling
        x = global_add_pool(x, batch)

        # apply the linear layers, dropout after activation, https://sebastianraschka.com/faq/docs/dropout-activation.html)
        for module in self.lin_layers:
            x = self.activation_function(module(x))
            x = self.dropout(x)

        # apply the output layer (classification, produces logits)
        output = self.out_layer(x)
        return output





def compute_metrics(tp, tn, fp, fn) -> dict:
    """
    Compute metrics (accuracy, precision, recall, f1 score) from true positive, true negative, false positive, and false negative counts
    :param tp: true positive
    :param tn: true negative
    :param fp: false positive
    :param fn: false negative
    :return: dictionary with accuracy, precision, recall, and f1 score in addition to the input tp, tn, fp and fn
    """
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    metrics = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1 score': f1_score}
    return metrics

def train_GNNModelPyG (net: GNNModelPyG,
                       trainloader: DataLoader,
                       testloader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       loss_fn: torch.nn.modules.loss._Loss,
                       scheduler: torch.optim.lr_scheduler.LRScheduler,
                       num_epochs: int,
                       metrics_history = None):
    """
    Train the GNN model
    :param net: PyTorch Geometric model
    :param trainloader: PyTorch Geometric DataLoader with the training data
    :param testloader: PyTorch Geometric DataLoader with the test data
    :param optimizer: PyTorch optimizer
    :param loss_fn: PyTorch loss function
    :param scheduler: PyTorch learning rate scheduler
    :param num_epochs: number of epochs
    :param metrics_history: list with metrics history
    :return:
    """
    # metrics history, continue from the previous run if not None
    if metrics_history is None:
        metrics_history = []

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train
    for total_epochs in range(num_epochs):
        epoch_loss = 0
        net.train()
        tp, tn, fp, fn = 0, 0, 0, 0
        for batch in trainloader:
            # batch.to(device)

            y = [1 if json.loads(assay_data)[0]['assay_result']=='positive' else 0 for assay_data in batch.assay_data]
            y = torch.tensor(y, dtype=torch.long).to(device)

            optimizer.zero_grad()
            pred = net(batch)
            tp += ((torch.argmax(pred, dim=1) == y) & (y == 1)).int().sum()
            tn += ((torch.argmax(pred, dim=1) == y) & (y == 0)).int().sum()
            fp += ((torch.argmax(pred, dim=1) != y) & (y == 0)).int().sum()
            fn += ((torch.argmax(pred, dim=1) != y) & (y == 1)).int().sum()
            loss = loss_fn(pred, y)
            loss.backward()
            epoch_loss += loss.item()*len(batch)
            optimizer.step()

        scheduler.step()
        train_avg_loss = epoch_loss / len(trainloader.dataset)

        # training metrics
        metrics = {'training': compute_metrics(tp.item(), tn.item(), fp.item(), fn.item())}
        metrics['training'].update({'loss': train_avg_loss})


        val_loss = 0
        net.eval()
        tp, tn, fp, fn = 0, 0, 0, 0
        for batch in testloader:
            # batch.to(device)
            y = [1 if json.loads(assay_data)[0]['assay_result']=='positive' else 0 for assay_data in batch.assay_data]
            y = torch.tensor(y, dtype=torch.long).to(device)

            pred = net(batch)
            tp += ((torch.argmax(pred, dim=1) == y) & (y == 1)).int().sum()
            tn += ((torch.argmax(pred, dim=1) == y) & (y == 0)).int().sum()
            fp += ((torch.argmax(pred, dim=1) != y) & (y == 0)).int().sum()
            fn += ((torch.argmax(pred, dim=1) != y) & (y == 1)).int().sum()
            loss = loss_fn(pred, y)
            val_loss += loss.item()*len(batch)

        val_avg_loss = val_loss / len(testloader.dataset)

        # validation metrics
        metrics.update({'validation': compute_metrics(tp.item(), tn.item(), fp.item(), fn.item())})
        metrics['validation'].update({'loss': val_avg_loss})

        log.info(f"epoch [{total_epochs+1}/{num_epochs}] learning rate {optimizer.param_groups[0]['lr']:.3e} | training: loss {metrics['training']['loss']:.5f} tp {metrics['training']['tp']} tn {metrics['training']['tn']} fp {metrics['training']['fp']} fn {metrics['training']['fn']} f1 {metrics['training']['f1 score']:.5f}"
                 f" | validation: loss {metrics['validation']['loss']:.5f} tp {metrics['validation']['tp']} tn {metrics['validation']['tn']} fp {metrics['validation']['fp']} fn {metrics['validation']['fn']} f1 {metrics['validation']['f1 score']:.5f}")
        metrics_history.append(metrics)

    return metrics_history



def plot_metrics(metrics_history: dict, output: Path):
    """
    Plot the metrics history
    :param metrics_history: list with metrics history dictionaries for training and validation for each epoch
    :param output: output file path
    :return:
    """
    # plot the metrics history
    # plot accuracy, precision, recall and F1-score
    plt.interactive(False)
    fig = plt.figure(figsize=(10, 6))
    axs = fig.subplots(3, 1, sharex=True)
    # .. loss
    loss_train = [metrics['training']['loss'] for metrics in metrics_history]
    loss_val = [metrics['validation']['loss'] for metrics in metrics_history]
    axs[0].plot(loss_train, label='training', c='k', linewidth=0.5)
    axs[0].plot(loss_val, label='validation', c='k', linestyle='dashed', linewidth=0.5)
    n_last_epochs = 10
    train_loss_converged = sum(loss_train[-n_last_epochs:]) / n_last_epochs
    val_loss_converged = sum(loss_val[-n_last_epochs:]) / n_last_epochs
    title = f'loss: training {train_loss_converged:.5f}, validation {val_loss_converged:.5f}'
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].set_ylim(0, 1.1*max(loss_val))
    axs[0].set_ylabel('loss')

    # training metrics
    accuracy_train = [metrics['training']['accuracy'] for metrics in metrics_history]
    precision_train = [metrics['training']['precision'] for metrics in metrics_history]
    recall_train = [metrics['training']['recall'] for metrics in metrics_history]
    f1_score_train = [metrics['training']['f1 score'] for metrics in metrics_history]
    style = '-'
    axs[1].set_ylim(0.5, 1.)
    axs[1].hlines(0.8, 0, len(metrics_history), colors='k', linestyles='dashed', label='80%')
    axs[1].plot(accuracy_train, label=f'accuracy', linestyle=style, c='k', linewidth=0.5)
    axs[1].plot(precision_train, label=f'precision', linestyle=style, c='r', linewidth=0.5)
    axs[1].plot(recall_train, label=f'recall', linestyle=style, c='g', linewidth=0.5)
    axs[1].plot(f1_score_train, label=f'F1 score', linestyle=style, c='y', linewidth=0.5)
    train_f1_converged = sum(f1_score_train[-n_last_epochs:]) / n_last_epochs
    title = f'F1 score: training {train_f1_converged:.5f}'
    axs[1].set_title(title)
    axs[1].legend()

    # validation metrics
    accuracy_val = [metrics['validation']['accuracy'] for metrics in metrics_history]
    precision_val = [metrics['validation']['precision'] for metrics in metrics_history]
    recall_val = [metrics['validation']['recall'] for metrics in metrics_history]
    f1_score_val = [metrics['validation']['f1 score'] for metrics in metrics_history]
    style = '-'
    axs[2].set_ylim(0.5, 1.)
    axs[2].hlines(0.8, 0, len(metrics_history), colors='k', linestyles='dashed', label='80%')
    axs[2].plot(accuracy_val, label=f'accuracy', linestyle=style, c='k', linewidth=0.5)
    axs[2].plot(precision_val, label=f'precision', linestyle=style, c='r', linewidth=0.5)
    axs[2].plot(recall_val, label=f'recall', linestyle=style, c='g', linewidth=0.5)
    axs[2].plot(f1_score_val, label=f'F1 score', linestyle=style, c='y', linewidth=0.5)
    val_f1_converged = sum(f1_score_val[-n_last_epochs:]) / n_last_epochs
    title = f'F1 score: validation {val_f1_converged:.5f}'
    axs[2].set_title(title)
    axs[2].legend()
    fig.tight_layout()
    # save the figure
    fig.savefig(output, dpi=300)
    plt.interactive(True)

