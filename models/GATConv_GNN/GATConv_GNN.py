import logger
log = logger.get_logger(__name__)

from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv, global_add_pool, global_mean_pool

class GATConv_GNN(torch.nn.Module):
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 n_conv: int,
                 n_heads: int,
                 n_conv_hidden: int,
                 v2: bool,
                 n_lin: int,
                 n_lin_hidden: int,
                 dropout_lin: float,
                 dropout_conv: float,
                 pool: str,
                 n_classes: [int],
                 activation_function: Callable = torch.nn.functional.leaky_relu
                 ):


        """
        Implements the GAT multitask classifier in PyTorch Geometric
        :param num_node_features: number of node features
        :param num_edge_features: number of edge features
        :param n_conv: number of message passing (convolutional) layers
        :param n_conv_hidden: number of hidden features in the convolutional layers per head
        :param n_heads: number of multi-head-attentions
        :param v2: if True, use the GATv2 implementation, otherwise use the original GAT implementation
        :param n_lin: number of linear layers
        :param n_lin_hidden: number of hidden features in the linear layers
        :param dropout_lin: dropout rate in linear layers
        :param dropout_conv: dropout rate in convolutional layers
        :param pool: "mean" or "add" pooling
        :param n_classes: array with the number of output classes in each classification task
        :param activation_function: PyTorch activation function, e.g. torch.nn.functional.relu or torch.nn.functional.leaky_relu
        See
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv
        https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
        """
        super().__init__()

        # number of node and edge features
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        # number of message passing layers
        self.n_conv = n_conv

        # number of hidden features in the convolutional layers per head
        self.n_conv_hidden = n_conv_hidden

        # number of heads in the multi-head-attention
        self.n_heads = n_heads

        # GATconv version
        self.v2 = v2

        # linear layers' parameters
        self.n_lin = n_lin
        self.n_lin_hidden = n_lin_hidden

        # general parameters
        self.dropout_lin = dropout_lin
        self.dropout_conv = dropout_conv
        self.pool = pool
        self.activation_function = activation_function
        self.n_classes = n_classes

        # set up the GAT convolutional layer version
        if self.v2:
            gat = GATv2Conv
        else:
            gat = GATConv

        # set up the GAT convolutions
        self.convs = torch.nn.ModuleList()
        # first layer
        self.convs.append(gat(in_channels=self.num_node_features, out_channels=self.n_conv_hidden, heads=self.n_heads, edge_dim=self.num_edge_features, dropout=self.dropout_conv))
        # intermediate layers
        for _ in range(self.n_conv-2):
            self.convs.append(gat(in_channels=self.n_conv_hidden*self.n_heads, out_channels=self.n_conv_hidden, heads=self.n_heads, edge_dim=self.num_edge_features, dropout=self.dropout_conv))
        # last layer
        self.convs.append(gat(in_channels=self.n_conv_hidden*self.n_heads, out_channels=self.n_conv_hidden, heads=1, edge_dim=self.num_edge_features, dropout=self.dropout_conv))


        # set up the linear layers (if any)
        self.lin_layers = torch.nn.ModuleList()
        if self.n_lin > 0:
            self.lin_layers.append(torch.nn.Linear(self.n_conv_hidden, self.n_lin_hidden))
            self.lin_layers.extend(
                [torch.nn.Linear(self.n_lin_hidden, self.n_lin_hidden)
                 for i in range(self.n_lin-1)])

        # set up the output layers, one for each task
        self.out_layers = torch.nn.ModuleList()
        for n_class in self.n_classes:
            if self.n_lin > 0:
                self.out_layers.append(torch.nn.Linear(self.n_lin_hidden, n_class))
            else:
                self.out_layers.append(torch.nn.Linear(self.n_conv_hidden, n_class))

        # dropout layer for the linear layers
        self.dropout_lin = torch.nn.Dropout(dropout_lin)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor, task_id: int):
        '''
        Inference function
        :param x: node features
        :param edge_index: edge index
        :param edge_attr: edge features
        :param batch: batch vector
        :param task_id: task to model
        :return: logits for the different classes of the task
        '''

        # apply the convolutional layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs)-1:
                x = self.activation_function(x)

        # pooling
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        else:
            x = global_add_pool(x, batch)
        x = self.dropout_lin(x)

        # apply the linear layers, dropout after activation, https://sebastianraschka.com/faq/docs/dropout-activation.html)
        for module in self.lin_layers:
            x = self.activation_function(module(x))
            x = self.dropout_lin(x)

        # apply the output layer (classification, produces logits)
        output = self.out_layers[task_id](x)
        return output

