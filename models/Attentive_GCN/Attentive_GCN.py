import logger
log = logger.get_logger(__name__)

from typing import Callable

import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_add_pool

class Attentive_GCN(torch.nn.Module):
    def __init__(self,

                 n_classes: [int]):
        """
        Implements the Attentive multitask classifier in PyTorch Geometric
        :param n_classes: array with the number of output classes in each classification task
        """
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor, task_id: int):
        pass


# implement the below, using a wrapper
# Attentive FP model
# num_node_features = dset.num_node_features
# num_edge_features = dset.num_edge_features
# n_classes = 2
# from torch_geometric.nn.models import AttentiveFP
# net = AttentiveFP(in_channels=num_node_features, hidden_channels=200, out_channels=n_classes,
#                     edge_dim=num_edge_features, num_layers=2, num_timesteps=2,
#                     dropout=0.1)
# net.to(device)
#
# # number of epochs
# num_epochs = 200
#
# # optimiser
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=10**-3)
#
# # loos function
# loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
#
# # scheduler
# lambda_group1 = lambda epoch: 0.98 ** epoch
# scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group1])
#
# metrics_history = train_GNNModelPyG(net, trainloader, testloader, optimizer, loss_fn, scheduler, num_epochs)
#
# # plot the metrics history
# plot_metrics(metrics_history, Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\metrics_history.png'))
#
