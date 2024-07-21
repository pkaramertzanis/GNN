# setup logging
import glob
import random

import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_muta_model.log')

# import and configure pandas globally
import pandas as pd
import pandas_config

from pathlib import Path
import numpy as np

import torch
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR

from torch_geometric.loader import DataLoader
from models.PyG_Dataset import PyG_Dataset

from sklearn.model_selection import train_test_split

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import Counter
from itertools import product
import random
import math
from datetime import datetime


from sklearn.model_selection import StratifiedKFold

from data.combine import create_sdf

from models.DMPNN_GCN.DMPNN_GCN import DMPNN_GCN
from models.Attentive_GCN.Attentive_GCN import Attentive_GCN

from models.metrics import plot_metrics
from models.train import train_eval

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set up the dataset
flat_datasets = [
                # r'data/Hansen_2009/tabular/Hansen_2009_genotoxicity.xlsx',
                r'data/Leadscope/tabular/Leadscope_genotoxicity.xlsx',
                # r'data/QSARToolbox/tabular/QSARToolbox_genotoxicity.xlsx'
                ]
task_aggregation_cols = ['in vitro/in vivo', 'endpoint', 'assay']
outp_sdf = Path(r'data/combined/sdf/genotoxicity_dataset.sdf')
outp_tab = Path(r'data/combined/tabular/genotoxicity_dataset.sdf')
tasks = create_sdf(flat_datasets = flat_datasets,
                   filter_unknown = False,
                   task_aggregation_cols = task_aggregation_cols,
                   outp_sdf = outp_sdf,
                   outp_tab = outp_tab)
# ensure task names can be used in paths
# tasks = [task.replace('/','slash') for task in tasks]

# set general parameters
PYTORCH_SEED = 1 # seed for PyTorch random number generator, it is also used for splits and shuffling to ensure reproducibility
MINIMUM_TASK_DATASET = 256 # minimum number of data points for a task
BATCH_SIZE_MAX = 512 # maximum batch size (largest task, the smaller tasks are scaled accordingly so the number of batches is the same)
K_FOLD_INNER = 5 # number of folds for the inner cross-validation
K_FOLD_OUTER = 10 # number of folds for the outer cross-validation
NUM_EPOCHS = 50 # number of epochs
MODEL_NAME = 'Attentive_GCN' # name of the model, can be 'DMPNN_GCN' or 'Attentive_GCN'
SCALE_LOSS = 'equal task' # how to scale the loss function, can be 'equal task' or None
HANDLE_AMBIGUOUS = 'ignore' # how to handle ambiguous outcomes, can be 'keep', 'set_positive', 'set_negative' or 'ignore', but the model fitting does not support 'keep'
LOG_EPOCH_FREQUENCY = 10 # frequency to log the metrics during training



# location to store the metrics logs
metrics_history_path = Path(rf'D:\myApplications\local\2024_01_21_GCN_Muta\output\iteration34')/MODEL_NAME
metrics_history_path.mkdir(parents=True, exist_ok=True)

# features, checkers and standardisers
NODE_FEATS = ['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization', 'num_rings']
EDGE_FEATS = ['bond_type', 'is_conjugated', 'num_rings'] # ['bond_type', 'is_conjugated', 'stereo_type']

# select model
if MODEL_NAME == 'DMPNN_GCN':
    model = DMPNN_GCN
    model_parameters = {'n_conv': [5], # [1, 2, 3, 4, 5, 6]
                        'n_lin': [2], # [1, 2, 3, 4]
                        'n_conv_hidden': [128], # [32, 64, 128, 256]
                        'n_edge_NN': [64], # [32, 64, 128, 256]
                        'n_lin_hidden': [64], # [32, 64, 128, 256, 512]
                        'dropout': [0.5], # [0.5, 0.6, 0.7, 0.8]
                        'activation_function': [torch.nn.functional.leaky_relu],
                        'learning_rate': [0.01],  # [0.001, 0.005, 0.01]
                        'weight_decay': [1.e-5],  # [1.e-5, 1e-4, 1e-3]
                        }
elif MODEL_NAME == 'Attentive_GCN':
    model = Attentive_GCN
    model_parameters = {'hidden_channels': [128], # [64, 128, 256]
                        'num_layers': [2], # [1, 2, 3, 4]
                        'num_timesteps': [2], # [1, 2, 3, 4]
                        'dropout': [0.0], # [0.5, 0.6, 0.7, 0.8]
                        'learning_rate': [0.005], # [0.001, 0.005, 0.01]
                        'weight_decay': [1e-3],  # [1.e-5, 1e-4, 1e-3]
                        }


# build the PyG datasets, no split at this stage
dsets = {}
for i_task, task in enumerate(tasks):
    log.info(f'preparing PyG dataset for task: {task}')
    entry = {}
    dset = PyG_Dataset(root=Path(outp_sdf.parent.parent/f'PyTorch_Geometric_{i_task}'),
                       task=task,
                       node_feats=NODE_FEATS,
                       edge_feats=EDGE_FEATS,
                       ambiguous_outcomes=HANDLE_AMBIGUOUS,
                       force_reload=True,
                       )
    dset.to(device) # all datasets are moved to the device
    # store the dataset in the dset dictionary
    entry['dset'] = dset
    if len(dset) >= MINIMUM_TASK_DATASET:
       dsets[task] = entry
    else:
        log.warning(f'task {task} has less than {MINIMUM_TASK_DATASET} data points, skipping')





# outer and inner cross-validation splits (in the interest of reproducibility, the order is deterministic for all splits)
cv_outer = StratifiedKFold(n_splits=K_FOLD_OUTER, random_state=PYTORCH_SEED, shuffle=True)
cv_inner = StratifiedKFold(n_splits=K_FOLD_INNER, random_state=PYTORCH_SEED, shuffle=True)
splits = []
for task in dsets:
    dset = dsets[task]['dset']
    y_task = [d.assay_data for d in dset]
    y_task_dist = dict(Counter(y_task))
    per_pos_task = 100 * y_task_dist['positive'] / len(y_task)
    log.info(f"task {task}: {len(y_task):4d} data points, positives: {y_task_dist['positive']:4d} ({per_pos_task:.2f}%)")

    for i_outer, (tmp_indices, test_indices) in enumerate(cv_outer.split(range(len(dset)), y=y_task)):
        y_outer_test = [d.assay_data for i, d in enumerate(dset) if i in test_indices]
        y_outer_test_dist = dict(Counter(y_outer_test))
        per_pos_test = 100 * y_outer_test_dist['positive'] / len(y_outer_test)
        log.info(f"task {task}, outer fold {i_outer}, test set: {len(y_outer_test):4d} data points, positives: {y_outer_test_dist['positive']:4d} ({per_pos_test:.2f}%)")

        y_inner = [d.assay_data for i, d in enumerate(dset) if i in tmp_indices]
        for i_inner, (train_indices, evaluate_indices) in enumerate(cv_inner.split(range(len(tmp_indices)), y=y_inner)):
            train_indices = np.array([tmp_indices[i] for i in train_indices])
            evaluate_indices = np.array([tmp_indices[i] for i in evaluate_indices])
            y_inner_train = [d.assay_data for i, d in enumerate(dset) if i in train_indices]
            y_inner_train_dist = dict(Counter(y_inner_train))
            per_pos_inner_train = 100 * y_inner_train_dist['positive'] / len(y_inner_train)
            log.info(f"task {task}, outer fold {i_outer}, inner fold {i_inner}, train set: {len(y_inner_train):4d} data points, positives: {y_inner_train_dist['positive']:4d} ({per_pos_inner_train:.2f}%)")
            entry = {'task': task, 'outer fold': i_outer, 'inner fold': i_inner,
                     'train indices': train_indices, 'eval indices': evaluate_indices, 'test indices': test_indices,
                     'task # data points': len(dset),
                     'test # data points': len(test_indices),
                     'train # data points': len(train_indices),
                     'task %positives': per_pos_task,
                     'test %positives': per_pos_test,
                     'train %positives': per_pos_inner_train,
                     }
            splits.append(entry)
splits = pd.DataFrame(splits)
for task in dsets:
    tmp = splits.loc[splits['task']==task, ['task %positives', 'test %positives', 'train %positives']].describe().loc[['min', 'max']].to_markdown()
    log.info(f'task {task}, %positives in different splits\n{tmp}')



# set up the model configurations
configuration_ID = 0
configurations = []
for model_parameter_values in product(*model_parameters.values()):
    configuration = {'configuration_ID': configuration_ID}
    configuration.update(dict(zip(model_parameters.keys(), model_parameter_values)))
    configurations.append(configuration)
    configuration_ID += 1
log.info(f'number of configurations: {len(configurations)}')
# .. shuffle to sample the configurations randomly
random.seed(PYTORCH_SEED)
random.shuffle(configurations)


# outer loop of the nested cross-validation
for i_outer in range(K_FOLD_OUTER):
    log.info(f'Initiating outer iteration {i_outer}')

    # loop over the model configurations
    for i_configuration, configuration in enumerate(configurations, 0):
        configuration_ID = configuration['configuration_ID']
        log.info(f'Trialing model/optimiser configuration {configuration_ID} ({i_configuration+1} out of {len(configurations)})')
        model_parameters = {k: v for k, v in configuration.items() if k not in ['configuration_ID', 'learning_rate', 'weight_decay']}
        optimiser_parameters = {k: v for k, v in configuration.items() if k in ['configuration_ID', 'learning_rate', 'weight_decay']}

        # inner loop of the nested cross-validation
        metrics_history_configuration = []
        for i_inner in range(K_FOLD_INNER):
            log.info(f'Initiating inner iteration {i_inner}')
            # .. create the train and eval set loaders
            train_loaders, eval_loaders = [], []
            msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == i_inner)
            train_set_size_max = max(len(idxs) for idxs in splits.loc[msk, 'train indices'])  # largest train set size among tasks
            eval_set_size_max = max(len(idxs) for idxs in splits.loc[msk, 'eval indices'])  # largest eval set size among tasks
            for task in dsets:
                msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == i_inner) & (splits['task'] == task)
                train_set = dsets[task]['dset'].index_select(splits.loc[msk, 'train indices'].iloc[0].tolist())
                batch_size = round(BATCH_SIZE_MAX * len(train_set) / float(train_set_size_max))
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True) # .. we drop the last to have stable gradients
                train_loaders.append(train_loader)
                eval_set = dsets[task]['dset'].index_select(splits.loc[msk, 'eval indices'].iloc[0].tolist())
                batch_size = round(BATCH_SIZE_MAX * len(eval_set) / float(eval_set_size_max))
                eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=False)
                eval_loaders.append(eval_loader)
                log.info(f'task {task}, train set: {len(train_set):4d} data points in {len(train_loader)} batches, eval set: {len(eval_set):4d} data points in {len(eval_loader)} batches')

            # set up the model
            num_node_features = (train_loaders[0].dataset).num_node_features
            num_edge_features = (train_loaders[0].dataset).num_edge_features
            n_classes = [2] * len(dsets)

            # --------------


            torch.manual_seed(PYTORCH_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(PYTORCH_SEED)
            # set the model
            net = model(num_node_features=num_node_features, num_edge_features=num_edge_features,
                            **model_parameters,
                            n_classes=n_classes)
            net.to(device)


            # optimiser
            optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'], betas=[0.9, 0.999], eps=1e-08, weight_decay=optimiser_parameters['weight_decay'], amsgrad=False)
            # loss function, default reduction is mean, we treat the binary classification as a multi-class classification for generalisation
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
            # scheduler
            lambda_group = lambda epoch: 0.97 ** epoch
            scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])

            # train the model
            outp = metrics_history_path/f'outer_fold_{i_outer}_configuration_ID_{configuration_ID}_inner_fold_{i_inner}'
            outp.mkdir(parents=True, exist_ok=True)
            metrics_history = train_eval(net, train_loaders, eval_loaders, optimizer, loss_fn, scheduler, NUM_EPOCHS, outp/'model_weights_diff_quantiles.tsv', log_epoch_frequency=LOG_EPOCH_FREQUENCY, scale_loss=SCALE_LOSS)

# -------------

            # plot the metrics
            plot_metrics(metrics_history, outp)


            # log the metrics for the training set and evaluation set
            metrics_history = pd.DataFrame(metrics_history)
            cols = {'time': datetime.now(), 'outer fold': i_outer}
            cols.update(configuration)
            cols.update({'inner fold': i_inner})
            for i_col, (col_name, col_value) in enumerate(cols.items()):
                metrics_history.insert(i_col, col_name, col_value)

            # append the results to the metric history log
            with open(metrics_history_path/'metrics_history.tsv', mode='at', encoding='utf-8', buffering=1, newline='') as f:
                metrics_history.to_csv(f, header=f.tell()==0, index=False, sep='\t', lineterminator='\n')

    # find the optimal configuration by using the average eval f1 score over the inner folds (avoid reading the whole file in memory)
    chunk_iterator = pd.read_csv(metrics_history_path/'metrics_history.tsv', chunksize=10_000, sep='\t')
    metrics_history_configuration = []
    for chunk in chunk_iterator:
        # .. select the eval rows that provide the aggregate metrics across tasks and batches
        msk = (chunk['batch'].isnull()) & (chunk['task'].isnull()) & (chunk['stage'] == 'eval') & (chunk['type'] == 'aggregate (epoch)')
        metrics_history_configuration.append(chunk.loc[msk])
    metrics_history_configuration = pd.concat(metrics_history_configuration, axis=0, sort=False, ignore_index=True)
    # .. keep the last three epochs
    msk = (metrics_history_configuration['epoch'] >= metrics_history_configuration['epoch'].max() - 3)
    metrics_history_configuration = metrics_history_configuration.loc[msk]
    f1_eval_inner_folds = metrics_history_configuration.groupby('configuration_ID')['f1 score'].mean()
    best_configuration_ID = f1_eval_inner_folds.idxmax()
    log.info(f'outer fold {i_outer}, best configuration ID: {best_configuration_ID} with f1 score: {f1_eval_inner_folds.max():.4} (range: {f1_eval_inner_folds.min():.4} - {f1_eval_inner_folds.max():.4})')

    # refit using the whole train + eval sets and evaluate in the test set
    msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == 0)
    train_eval_set_size_max = max(len(idxs_train.tolist()+idxs_eval.tolist()) for idxs_train, idxs_eval in zip(splits.loc[msk, 'train indices'], splits.loc[msk, 'eval indices']))
    test_set_size_max = max(len(idxs) for idxs in splits.loc[msk, 'test indices'])  # largest test set size among tasks
    train_eval_loaders, test_loaders = [], []
    for task in dsets:
        msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == 0) & (splits['task'] == task)
        train_eval_set = dsets[task]['dset'].index_select(splits.loc[msk, 'train indices'].iloc[0].tolist()+splits.loc[msk, 'eval indices'].iloc[0].tolist())
        batch_size = math.ceil(BATCH_SIZE_MAX * len(train_eval_set) / float(train_eval_set_size_max))
        train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size, shuffle=True, drop_last=True) # .. we drop the last to have stable gradients
        train_eval_loaders.append(train_eval_loader)
        test_set = dsets[task]['dset'].index_select(splits.loc[msk, 'test indices'].iloc[0].tolist())
        batch_size = math.ceil(BATCH_SIZE_MAX * len(test_set) / float(test_set_size_max))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loaders.append(test_loader)
    # .. train the model
    # .. set the seed for reproducibility
    torch.manual_seed(PYTORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PYTORCH_SEED)
    # set the model
    configuration = [configuration for configuration in configurations if configuration['configuration_ID'] == best_configuration_ID][0]
    configuration_ID = configuration['configuration_ID']
    model_parameters = {k: v for k, v in configuration.items() if k not in ['configuration_ID', 'learning_rate', 'weight_decay']}
    optimiser_parameters = {k: v for k, v in configuration.items() if k in ['configuration_ID', 'learning_rate', 'weight_decay']}

    net = model(num_node_features=num_node_features, num_edge_features=num_edge_features,
                    **model_parameters,
                    n_classes=n_classes)
    net.to(device)
    # optimiser
    optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'], betas=[0.9, 0.999], eps=1e-08, weight_decay=optimiser_parameters['weight_decay'],
                                 amsgrad=False)
    # loss function, default reduction is mean, we treat the binary classification as a multi-class classification for generalisation
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
    # scheduler
    lambda_group = lambda epoch: 0.97 ** epoch
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])

    # train the model, double the number of epochs for the final training
    outp = metrics_history_path/f'outer_fold_{i_outer}_configuration_ID_{configuration_ID}'
    outp.mkdir(parents=True, exist_ok=True)
    metrics_history = train_eval(net, train_eval_loaders, test_loaders, optimizer, loss_fn, scheduler, 2*NUM_EPOCHS, outp=None, log_epoch_frequency=LOG_EPOCH_FREQUENCY, scale_loss=SCALE_LOSS)

    # plot the metrics
    plot_metrics(metrics_history, outp)

    # save the model
    torch.save(net, outp/'model.pth')

    # log the metrics for the training set and evaluation set
    metrics_history = pd.DataFrame(metrics_history)
    cols = {'time': datetime.now(), 'outer fold': i_outer}
    cols.update(configuration)
    cols.update({'inner fold': None})
    for i_col, (col_name, col_value) in enumerate(cols.items()):
        metrics_history.insert(i_col, col_name, col_value)
    metrics_history['stage'] = np.where(metrics_history['stage']=='train', 'train+eval', 'test')

    # append the results to the metric history log
    with open(metrics_history_path/'metrics_history.tsv', mode='at', encoding='utf-8', buffering=1, newline='') as f:
        metrics_history.to_csv(f, header=f.tell() == 0, index=False, sep='\t', lineterminator='\n')


# retrieve the metrics for each outer iteration and list the optimal configuration for each outer iteration
chunk_iterator = pd.read_csv(metrics_history_path/'metrics_history.tsv', chunksize=10_000, sep='\t')
metrics_history_outer = []
for chunk in chunk_iterator:
    # .. select the train+eval and test rows
    msk = (chunk['batch'].isnull()) & (chunk['task'].isnull()) & chunk['stage'].isin(['train+eval', 'test']) & (chunk['type'] == 'aggregate (epoch)')
    metrics_history_outer.append(chunk.loc[msk])
metrics_history_outer = pd.concat(metrics_history_outer, axis=0, sort=False, ignore_index=True)
# keep only the last three epochs (or averaging)
msk = metrics_history_outer['epoch'] >= metrics_history_outer['epoch'].max()-3
metrics_history_outer = metrics_history_outer.loc[msk]
res = metrics_history_outer.pivot_table(index='outer fold', columns='stage', values=['accuracy', 'precision', 'recall', 'f1 score'], aggfunc='mean', margins=True)
res.columns = ['_'.join(col).strip() for col in res.columns.values]
res = res.drop([col for col in res.columns if col.endswith('_All')], axis='columns')
res = res.merge(metrics_history_outer[['outer fold', 'configuration_ID']+list(model_parameters.keys())].drop_duplicates().set_index('outer fold'), left_index=True, right_index=True, how='left')
res.to_excel(metrics_history_path/'metrics_history_outer.xlsx')


