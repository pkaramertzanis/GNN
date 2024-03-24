# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_muta_model.log')

from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR

from torch_geometric.loader import DataLoader
from models.GNN_PyTorch_Geometric import GNNDatasetPyG, GNNModelPyG, train_GNNModelPyG, plot_metrics


# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr',False)
# maximum number of columns
pd.set_option("display.max_columns",50)
# maximum number of rows
pd.set_option("display.max_rows",500)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True



# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Hansen 2009 dataset
dset = GNNDatasetPyG(root=Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\datasets/Hansen_2009/processed/PyTorch_Geometric'),
                     input_sdf=r'D:\myApplications\local\2024_01_21_GCN_Muta\datasets\Hansen_2009\processed\sdf\Ames_generic_Hansen_2009.sdf',
                     target_assay_name='AMES',
                     force_reload=False,
                     node_feats=['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization'],
                     edge_feats=['bond_type', 'is_conjugated'],
                     checker_ops = {'allowed_atoms': ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'B','Si', 'I', 'H']},
                     standardiser_ops = ['cleanup', 'addHs']
                     )

# Leadscope bacterial mutation dataset (version 2)
dset = GNNDatasetPyG(root=Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\datasets/Leadscope_bacterial_mutation_version2/processed/PyTorch_Geometric'),
                     input_sdf=r'D:\myApplications\local\2024_01_21_GCN_Muta\datasets\Leadscope_bacterial_mutation_version2\processed\sdf\Leadscope_bacterial_mutation_version2.sdf',
                     target_assay_name='AMES', force_reload=True,
                     node_feats=['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization'],
                     edge_feats=['bond_type', 'is_conjugated'],
                     checker_ops = {'allowed_atoms': ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'B','Si', 'I', 'H']},
                     standardiser_ops = ['cleanup', 'addHs']
                     )

dset.to(device)

# split into training and test set
generator=torch.Generator().manual_seed(1)
train_set, test_set = random_split(dset,[int(len(dset)*0.8), len(dset)-int(len(dset)*0.8)], generator=generator)
# we drop the last batch to have stable gradients
trainloader = DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True)
# we do not drop the last batch to have stable metrics (without oscillations for the test set), the same is needed for both test and validation in cross validation
testloader = DataLoader(test_set, batch_size=256, shuffle=True, drop_last=False)



# set up the model
num_node_features = dset.num_node_features
num_edge_features = dset.num_edge_features
n_conv = 4
n_edge_NN = 32
n_conv_hidden = 48
n_lin = 2
n_lin_hidden = 128
dropout = 0.4
activation_function = torch.nn.functional.leaky_relu
n_classes = 2
net = GNNModelPyG(num_node_features, num_edge_features,
                 n_conv, n_edge_NN, n_conv_hidden,
                 n_lin, n_lin_hidden,
                 dropout, activation_function, n_classes)
net.to(device)


# optimiser
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# loos function
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))

# scheduler
lambda_group1 = lambda epoch: 0.98 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group1])

# number of epochs
num_epochs = 1000

# train the model
metrics_history = train_GNNModelPyG(net, trainloader, testloader, optimizer, loss_fn, scheduler, num_epochs)

# plot the metrics history
plot_metrics(metrics_history, Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\metrics_history.png'))





