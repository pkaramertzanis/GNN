import logger
log = logger.get_logger(__name__)

from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_add_pool

class MPNN_GNN(torch.nn.Module):
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
                 n_classes: [int]):
        """
        Implements the GCN multitask classifier in PyTorch Geometric
        :param num_node_features: number of node features
        :param num_edge_features: number of edge features
        :param n_conv: number of convolutional layers
        :param n_edge_NN: number of neurons in the edge FNN
        :param n_conv_hidden: number of hidden features in the convolutional layers
        :param n_lin: number of linear layers
        :param n_lin_hidden: number of hidden features in the linear layers
        :param dropout: dropout rate
        :param activation_function: PyTorch activation function, e.g. torch.nn.functional.relu or torch.nn.functional.leaky_relu
        :param n_classes: array with the number of output classes in each classification task
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

        # set up the output layers, one for each task
        self.out_layers = torch.nn.ModuleList()
        for n_class in self.n_classes:
            self.out_layers.append(torch.nn.Linear(self.n_lin_hidden, n_class))

        # dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        # initialise the weights and biases
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.)

    # def forward(self, data):
    #     batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor, task_id: int):

        # apply the convolutional layers
        for module in self.conv_layers:
            x = self.activation_function(module(x, edge_index, edge_attr))
            # x = self.dropout(x)

        # pooling
        x = global_add_pool(x, batch)

        # apply the linear layers, dropout after activation, https://sebastianraschka.com/faq/docs/dropout-activation.html)
        for module in self.lin_layers:
            x = self.activation_function(module(x))
            x = self.dropout(x)

        # apply the output layer (classification, produces logits)
        output = self.out_layers[task_id](x)
        return output
