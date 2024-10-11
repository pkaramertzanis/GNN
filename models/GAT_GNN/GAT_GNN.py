import logger
log = logger.get_logger(__name__)

from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.nn import GAT, global_add_pool

class GAT_GNN(torch.nn.Module):
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 n_conv: int,
                 n_heads: int,
                 n_conv_hidden_per_head: int,
                 v2: bool,
                 n_lin: int,
                 n_lin_hidden: int,
                 dropout: float,
                 n_classes: [int],
                 activation_function: Callable = torch.nn.functional.leaky_relu,
                 ):
        """
        Implements the GAT multitask classifier in PyTorch Geometric
        :param num_node_features: number of node features
        :param num_edge_features: number of edge features
        :param n_conv: number of message passing (convolutional) layers
        :param n_conv_hidden_per_head: number of hidden features in the convolutional layers per head, i.e. n_conv_hidden = n_heads * n_conv_hidden_per_head
        :param n_heads: number of multi-head-attentions
        :param v2: if True, use the GATv2 implementation, otherwise use the original GAT implementation
        :param n_lin: number of linear layers
        :param n_lin_hidden: number of hidden features in the linear layers
        :param dropout: dropout rate
        :param n_classes: array with the number of output classes in each classification task
        :param activation_function: PyTorch activation function, e.g. torch.nn.functional.relu or torch.nn.functional.leaky_relu
        See
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GAT.html#torch_geometric.nn.models.GAT (and its base class)
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv
        """
        super().__init__()

        # number of node and edge features
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        # number of message passing layers
        self.n_conv = n_conv

        # number of hidden features in the convolutional layers
        self.n_conv_hidden_per_head = n_conv_hidden_per_head
        self.n_conv_hidden = self.n_conv_hidden_per_head * n_heads

        # number of heads in the multi-head-attention
        self.n_heads = n_heads

        # GATconv version
        self.v2 = v2

        # linear layers' parameters
        self.n_lin = n_lin
        self.n_lin_hidden = n_lin_hidden

        # general parameters
        self.dropout = dropout
        self.activation_function = activation_function
        self.n_classes = n_classes

        # set up the base GAT model
        # because we do not set out_channels the output size will be self.n_conv_hidden
        self.gat = GAT(num_node_features, self.n_conv_hidden, num_layers=self.n_conv, out_channels=None,
                       v2=self.v2, dropout=self.dropout, heads=self.n_heads)

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

        # dropout layer
        self.dropout = torch.nn.Dropout(dropout)

        # initialise the weights and biases of the output layers, the other modules are initialised by the GAT model
        # for output_layer in self.out_layers:
        #     torch.nn.init.xavier_uniform_(output_layer.weight)
        #     torch.nn.init.constant_(output_layer.bias, 0.)
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         torch.nn.init.xavier_uniform_(param)
        #     if 'bias' in name:
        #         torch.nn.init.constant_(param, 0.)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor, task_id: int):

        # apply the convolutional layers
        x = self.gat(x, edge_index, edge_attr)
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

