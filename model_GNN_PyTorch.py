# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_muta_model.log')

from pathlib import Path
import pickle

import sys
import networkx as nx
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from models.GNN_PyTorch import GNNModel, collate_graphs


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# import the training dataset
inpf = Path('training_sets/Hansen_2009.pkl')
with open(inpf, 'rb') as file:
    (tox_data, X, adjacency_matrices, edges, labels) = pickle.load(file)
    log.info(f'dataset read from {inpf}; it contains {len(tox_data)} records')
num_features = X.drop('mol ID', axis='columns').shape[1]

# create graph list
graph_list = []
n_mols = tox_data.shape[0]
for i_mol in range(n_mols):
    graph = {'A': torch.from_numpy(np.array(adjacency_matrices.iloc[i_mol]['adjacency matrix'])).float(),
             'X': torch.from_numpy(np.array(X.loc[X['mol ID']==i_mol].drop('mol ID', axis='columns'))).float(),
             'y': torch.tensor([1 if (labels.iloc[i_mol]=='positive') else 0]), 'batch': None}
    graph_list.append(graph)
graph_list_training, graph_list_validation = train_test_split(graph_list, test_size=0.2, random_state=2, stratify=labels)

# set the PyTorch model
# from models.GNN_PyTorch_simple import GNNModel
# model = GNNModel(num_features)
model = GNNModel(n_input=num_features,
                 n_conv=4, n_conv_hidden=32,
                 n_lin=3, n_lin_hidden=16,
                 dropout=0.3,
                 activation_function=torch.nn.functional.leaky_relu,
                 n_classes=2)


model.fit(graph_list_training, graph_list_validation, n_epocs=100, learning_rate=0.004, batch_size=200)


# single model prediction (returns logits in tensor form)
graph = graph_list_training[0]
pred = model(graph['X'], graph['A'], batch_mat=None).detach()
print(F.softmax(pred, dim=1).numpy())

# single model prediction (returns probabilities in numpy array)
graph_list = graph_list_training[0:1]
print(model.predict_proba(graph_list))

# batch model prediction (returns probabilities in numpy array)
graph_list = graph_list_training[:1]
res = model.predict_proba(graph_list)






# plot accuracy, precision, recall and F1-score
fig = plt.figure()
axs = fig.subplots(2, 1)
for part in ['training', 'validation']:
    if part == 'training':
        tp_history = model.tp_history_training
        tn_history = model.tn_history_training
        fp_history = model.fp_history_training
        fn_history = model.fn_history_training
        ax = axs[0]
        ax.set_title('training')
    else:
        tp_history = model.tp_history_validation
        tn_history = model.tn_history_validation
        fp_history = model.fp_history_validation
        fn_history = model.fn_history_validation
        ax = axs[1]
        ax.set_title('validation')
    accuracy = [(tp.item()+tn.item())/(tp.item()+tn.item()+fp.item()+fn.item()) for tp, tn, fp, fn in zip(tp_history, tn_history, fp_history, fn_history)]
    precision = [tp.item()/(tp.item()+fp.item()) if (tp.item()+fp.item())>0 else 0 for tp, fp in zip(tp_history, fp_history)]
    recall = [tp.item()/(tp.item()+fn.item()) for tp, fn in zip(tp_history, fn_history)]
    f1_score = [2*precision*recall/(precision+recall) if (precision+recall)>0 else 0 for precision, recall in zip(precision, recall)]
    style = '-'
    ax.plot(accuracy, label=f'accuracy ({part})', linestyle=style, c='k', linewidth=0.5)
    ax.plot(precision, label=f'precision ({part})', linestyle=style, c='r', linewidth=0.5)
    ax.plot(recall, label=f'recall ({part})', linestyle=style, c='g', linewidth=0.5)
    ax.plot(f1_score, label=f'F1 score ({part})', linestyle=style, c='y', linewidth=0.5)
    ax.legend()
plt.show()


# compute the AUC curve
from sklearn.metrics import RocCurveDisplay
fig = plt.figure()
ax = fig.subplots()
for part in ['training', 'validation']:
    if part == 'training':
        prob = model.predict_proba(graph_list_training)[:,1]
        y_true = [graph['y'].item() for graph in graph_list_training]
        RocCurveDisplay.from_predictions(y_true, prob, ax=ax, name='training')
    else:
        prob = model.predict_proba(graph_list_validation)[:,1]
        y_true = [graph['y'].item() for graph in graph_list_validation]
        RocCurveDisplay.from_predictions(y_true, prob, ax=ax, name='validation')
plt.show()





