import logger
log = logger.get_logger(__name__)

# setup logging
import numpy as np



import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Callable

class BasicGraphConvolutionLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W2 = Parameter(torch.rand(
            (in_channels, out_channels), dtype=torch.float32))
        self.W1 = Parameter(torch.rand(
            (in_channels, out_channels), dtype=torch.float32))

        # initialize weights
        torch.nn.init.xavier_uniform_(self.W2)
        torch.nn.init.xavier_uniform_(self.W1)

        self.bias = Parameter(torch.zeros(
            out_channels, dtype=torch.float32))

    def forward(self, X, A):
        potential_msgs = torch.mm(X, self.W2)
        propagated_msgs = torch.mm(A, potential_msgs)
        # normalise with the number of nodes
        propagated_msgs = propagated_msgs / A.sum(dim=1, keepdims=True)
        # update
        root_update = torch.mm(X, self.W1)
        output = propagated_msgs + root_update + self.bias
        return output

def global_sum_pool(X, batch_mat):
    if batch_mat is None or batch_mat.dim() == 1:
        return torch.sum(X, dim=0).unsqueeze(0)
    else:
        return torch.mm(batch_mat, X)


class GNNModel(torch.nn.Module):

    def __init__(self, n_input: int,
                 n_conv: int, n_conv_hidden: int,
                 n_lin: int, n_lin_hidden: int,
                 dropout: float,
                 activation_function: Callable,
                 n_classes: int):
        '''Initialize the GNN model. The following parameters are required:
        :param n_input: number of input features
        :param n_conv: number of convolutional layers
        :param n_conv_hidden: number of hidden features in the convolutional layers
        :param n_lin: number of linear layers
        :param n_lin_hidden: number of hidden features in the linear layers
        :param dropout: dropout rate
        :param activation_function: PyTorch activation function, e.g. torch.nn.functional.relu or torch.nn.functional.leaky_relu
        :param n_classes: number of output classes
        '''
        super().__init__()

        # convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(BasicGraphConvolutionLayer(n_input, n_conv_hidden))
        self.conv_layers.extend(
            [BasicGraphConvolutionLayer(n_conv_hidden, n_conv_hidden)
             for i in range(n_conv-1)])

        # linear layers
        self.lin_layers = torch.nn.ModuleList()
        self.lin_layers.append(torch.nn.Linear(n_conv_hidden, n_lin_hidden))
        self.lin_layers.extend(
            [torch.nn.Linear(n_lin_hidden, n_lin_hidden)
             for i in range(n_lin-1)])

        # dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        # activation function
        self.activation_function = activation_function

        # output layer
        self.out_layer = torch.nn.Linear(n_lin_hidden, n_classes)

        # initialize weights for the linear and output layers
        for module in self.lin_layers:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.)
        torch.nn.init.xavier_uniform_(self.out_layer.weight)
        torch.nn.init.constant_(self.out_layer.bias, 0.)

    def forward(self, X, A, batch_mat):

        # apply the convolutional layers
        x = X
        for module in self.conv_layers:
            x = self.activation_function(module(x, A))

        # pooling
        output = global_sum_pool(x, batch_mat)
        # do we need activation here? output = self.activation_function(output)

        # apply the linear layers (is the dropout before or after activation? https://sebastianraschka.com/faq/docs/dropout-activation.html)
        for module in self.lin_layers:
            output = self.activation_function(module(output))
            output = self.dropout(output)

        # apply the output layer (classification, produces logits)
        output = self.out_layer(output)
        return output
        # return F.softmax(output, dim=1)

    def fit(self, graph_list_training, graph_list_validation, n_epocs=100, learning_rate=0.004, batch_size=200):
        '''
        Fits a GNN model to the data for a predefined number of epocs
        :param graph_list_training: list of graphs (dictionaries) to be used for training
        :param graph_list_validation: list of graphs (dictionaries) to be used for validation
        :param n_epocs: number of epochs
        :param learning_rate: learning rate
        :param batch_size: batch size
        :return:
        '''

        dset_training = GNNDataset(graph_list_training)
        loader_training = DataLoader(dset_training, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs,
                                     drop_last=False)
        dset_validation = GNNDataset(graph_list_validation)
        loader_validation = DataLoader(dset_validation, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs,
                                     drop_last=False)

        # train the model (multiclass classification problem)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_hist_training, loss_hist_validation = [0] * n_epocs, [0] * n_epocs
        self.fp_history_training, self.fp_history_validation = [0] * n_epocs, [0] * n_epocs
        self.fn_history_training, self.fn_history_validation = [0] * n_epocs, [0] * n_epocs
        self.tp_history_training, self.tp_history_validation = [0] * n_epocs, [0] * n_epocs
        self.tn_history_training, self.tn_history_validation = [0] * n_epocs, [0] * n_epocs
        for epoch in range(n_epocs):
            # train model
            self.train()
            for batch in loader_training:
                A = batch['A']
                X = batch['X']
                batch_mat = batch['batch']

                pred = self.forward(X, A, batch_mat)
                loss = loss_fn(pred, batch['y'])
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                loss_hist_training[epoch] += loss.item() * batch['y'].size(0)
                self.tp_history_training[epoch] += (
                            (torch.argmax(pred, dim=1) == batch['y']) & (batch['y'] == 1)).int().sum()
                self.tn_history_training[epoch] += (
                            (torch.argmax(pred, dim=1) == batch['y']) & (batch['y'] == 0)).int().sum()
                self.fp_history_training[epoch] += (
                            (torch.argmax(pred, dim=1) != batch['y']) & (batch['y'] == 0)).int().sum()
                self.fn_history_training[epoch] += (
                            (torch.argmax(pred, dim=1) != batch['y']) & (batch['y'] == 1)).int().sum()
            loss_hist_training[epoch] /= len(loader_training.dataset)
            # evaluate model
            self.eval()
            with torch.no_grad():
                for batch in loader_validation:
                    A = batch['A']
                    X = batch['X']
                    batch_mat = batch['batch']

                    pred = self.forward(X, A, batch_mat)
                    loss = loss_fn(pred, batch['y'])

                    loss_hist_validation[epoch] += loss.item() * batch['y'].size(0)
                    self.tp_history_validation[epoch] += (
                                (torch.argmax(pred, dim=1) == batch['y']) & (batch['y'] == 1)).int().sum()
                    self.tn_history_validation[epoch] += (
                                (torch.argmax(pred, dim=1) == batch['y']) & (batch['y'] == 0)).int().sum()
                    self.fp_history_validation[epoch] += (
                                (torch.argmax(pred, dim=1) != batch['y']) & (batch['y'] == 0)).int().sum()
                    self.fn_history_validation[epoch] += (
                                (torch.argmax(pred, dim=1) != batch['y']) & (batch['y'] == 1)).int().sum()
                loss_hist_validation[epoch] /= len(loader_validation.dataset)

            # compute metrics for the epoch
            training_metrics = compute_metrics(self.tp_history_training[epoch], self.tn_history_training[epoch], self.fp_history_training[epoch], self.fn_history_training[epoch])
            validation_metrics = compute_metrics(self.tp_history_validation[epoch], self.tn_history_validation[epoch], self.fp_history_validation[epoch], self.fn_history_validation[epoch])

            # log the epoch
            log.info(f'Epoch {epoch:5d} | train: loss {loss_hist_training[epoch]:.4f} precision {training_metrics["precision"]:.4f} recall {training_metrics["recall"]:.4f} F1 {training_metrics["f1 score"]:.4f} '
                     f'tp {self.tp_history_training[epoch]:4d} tn: {self.tn_history_training[epoch]:4d} fp: {self.fp_history_training[epoch]:4d} fn: {self.fn_history_training[epoch]:4d} | '
                     f'valid: loss {loss_hist_validation[epoch]:.4f} precision {validation_metrics["precision"]:.4f} recall {validation_metrics["recall"]:.4f} F1 {validation_metrics["f1 score"]:.4f} '
                     f'tp {self.tp_history_validation[epoch]:4d} tn: {self.tn_history_validation[epoch]:4d} fp: {self.fp_history_validation[epoch]:4d} fn: {self.fn_history_validation[epoch]:4d}')


    def predict_proba(self, graph_list, batch_size=200) -> np.ndarray:
        '''
        Predict probabilities for a list of graphs
        :param graph_list: list of graphs
        :param batch_size: batch size to use for model inference
        :return: numpy array with predicted probabilities
        '''
        dset = GNNDataset(graph_list)
        loader = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs,
                                     drop_last=False)
        self.eval()
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                A = batch['A']
                X = batch['X']
                batch_mat = batch['batch']
                pred = self.forward(X, A, batch_mat)
                prob = F.softmax(pred, dim=1).numpy()
                y_pred.append(prob)
        y_pred = np.concatenate(y_pred, axis=0)
        return y_pred

def get_batch_tensor(graph_sizes):
    starts = [sum(graph_sizes[:idx]) for idx in range(len(graph_sizes))]
    stops = [starts[idx]+graph_sizes[idx] for idx in range(len(graph_sizes))]
    tot_len = sum(graph_sizes)
    batch_size = len(graph_sizes)
    batch_mat = torch.zeros([batch_size, tot_len]).float()
    for idx, starts_and_stops in enumerate(zip(starts, stops)):
        start = starts_and_stops[0]
        stop = starts_and_stops[1]
        batch_mat[idx, start:stop] = 1
    return batch_mat




def collate_graphs(batch):
    adj_mats = [graph['A'] for graph in batch]
    sizes = [A.size(0) for A in adj_mats]
    tot_size = sum(sizes)
    # create batch matrix
    batch_mat = get_batch_tensor(sizes)
    # combine feature matrices
    feat_mats = torch.cat([graph['X'] for graph in batch],dim=0)
    # combine labels
    labels = torch.cat([graph['y'] for graph in batch], dim=0)
    # combine adjacency matrices
    batch_adj = torch.zeros([tot_size, tot_size], dtype=torch.float32)
    accum = 0
    for adj in adj_mats:
        g_size = adj.shape[0]
        batch_adj[accum:accum+g_size, accum:accum+g_size] = adj
        accum = accum + g_size
    repr_and_label = {
            'A': batch_adj,
            'X': feat_mats,
            'y': labels,
            'batch' : batch_mat}

    return repr_and_label


class GNNDataset(Dataset):
    '''
    Simple PyTorch dataset that will use our list of graphs
    '''
    def __init__(self, graph_list):
        self.graphs = graph_list
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        mol_rep = self.graphs[idx]
        return mol_rep


def compute_metrics(tp, tn, fp, fn) -> dict:
    '''Compute metrics (accuracy, precision, recall, f1 score) from true positive, true negative, false positive, and false negative counts'''
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1 score': f1_score}


