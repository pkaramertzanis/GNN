# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_muta_model_baseline.log')

# import and configure pandas globally
import pandas_config
import pandas as pd
import numpy as np

from pathlib import Path
import glob
import random


import torch
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
# from models.PyG_Dataset import PyG_Dataset

from sklearn.model_selection import train_test_split

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import Counter
from itertools import product
import random
import math
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from sklearn.model_selection import StratifiedKFold

from data.combine import create_sdf

from models.FFNN.FFNN import FFNNModel

from models.metrics import (plot_metrics_convergence, consolidate_metrics_outer,
                            plot_metrics_convergence_outer_average, plot_roc_curve_outer_average)
from models.PyTorch_train import train_eval
from models.PyTorch_nested_cross_validation import nested_cross_validation

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# set up the dataset
flat_datasets = [
                #r'data/Hansen_2009/tabular/Hansen_2009_genotoxicity.xlsx',
                # r'data/Leadscope/tabular/Leadscope_genotoxicity.xlsx',
                r'data/QSARToolbox/tabular/QSARToolbox_genotoxicity.xlsx'
                ]
task_specifications = [
    {'filters': {'assay': ['bacterial reverse mutation assay'], 'cell line/species': ['Salmonella typhimurium (TA 100)', 'Salmonella typhimurium (TA 98)', 'Salmonella typhimurium (TA 1535)'], 'metabolic activation': ['yes', 'no']},
     'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation']},

    {'filters': {'assay': ['in vitro mammalian cell gene mutation test using the Hprt and xprt genes']},
     'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

    {'filters': {'assay': ['in vitro mammalian chromosome aberration test']},
     'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},
    ]
outp_sdf = Path(r'data/combined_dev/sdf/genotoxicity_dataset.sdf')
outp_tab = Path(r'data/combined_dev/tabular/genotoxicity_dataset.xlsx')
tasks = create_sdf(flat_datasets = flat_datasets,
                   task_specifications = task_specifications,
                   outp_sdf = outp_sdf,
                   outp_tab = outp_tab)


# set general parameters
PYTORCH_SEED = 1 # seed for PyTorch random number generator, it is also used for splits and shuffling to ensure reproducibility
MINIMUM_TASK_DATASET = 512 # minimum number of data points for a task
BATCH_SIZE_MAX = 512 # maximum batch size (largest task, the smaller tasks are scaled accordingly so the number of batches is the same)
K_FOLD_INNER = 5 # number of folds for the inner cross-validation
K_FOLD_OUTER = 10 # number of folds for the outer cross-validation
NUM_EPOCHS = 80 # number of epochs
HANDLE_AMBIGUOUS = 'ignore' # how to handle ambiguous outcomes, can be 'keep', 'set_positive', 'set_negative' or 'ignore', but the model fitting does not support 'keep'
MODEL_NAME = 'FFNN'
LOG_EPOCH_FREQUENCY = 10 # frequency to log the metrics during training
SCALE_LOSS_TASK_SIZE = None # how to scale the loss function, can be 'equal task' or None
SCALE_LOSS_CLASS_SIZE = 'equal class (task)' # how to scale the loss function, can be 'equal class (task)', 'equal class (global)' or None

# location to store the metrics logs
metrics_history_path = Path(rf'D:\myApplications\local\2024_01_21_GCN_Muta\output\iteration_dev')/MODEL_NAME
metrics_history_path.mkdir(parents=True, exist_ok=True)

fingerprint_parameters = {'radius': 2,
                          'fpSize': 2048,
                          'type': 'binary' # 'binary', 'count'
                          }

model_parameters = {'hidden_layers': [[512, 512], [256, 256]],  # [64, 128, 256]
                    'dropout': [0.5],  # [0.5, 0.6, 0.7, 0.8],
                    'activation_function': [torch.nn.functional.leaky_relu],
                    'learning_rate': [0.002],  # [0.001, 0.005, 0.01]
                    'weight_decay': [2.e-3],  # [1.e-5, 1e-4, 1e-3]
                    }


# read in the molecular structures from the SDF file, and compute fingerprints
import json
from cheminformatics.rdkit_toolkit import read_sdf
from rdkit.Chem import AllChem
from tqdm import tqdm
fpgen = AllChem.GetMorganGenerator(radius=fingerprint_parameters['radius'], fpSize=fingerprint_parameters['fpSize'],
                                   countSimulation=False, includeChirality=False)
mols = read_sdf(outp_sdf)
data = []
for i_mol, mol in tqdm(enumerate(mols)):
    tox_data = mol.GetProp('genotoxicity')
    tox_data = pd.DataFrame(json.loads(tox_data))
    # keep only positive and negative calls
    tox_data = tox_data.loc[tox_data['genotoxicity'].isin(['positive', 'negative'])]
    if not tox_data.empty:
        fg = fpgen.GetFingerprint(mol)
        fg_count = fpgen.GetCountFingerprint(mol)
        if fingerprint_parameters['type'] == 'binary':
            tox_data['fingerprint'] = fg.ToBitString()
        elif fingerprint_parameters['type'] == 'count':
            tox_data['fingerprint'] = ''.join([str(bit) for bit in fg_count.ToList()])
        else:
            ex = ValueError(f"unknown fingerprint type: {fingerprint_parameters}")
            log.error(ex)
            raise ex
        tox_data['molecule ID'] = i_mol
        data.append(tox_data)
data = pd.concat(data, axis='index', ignore_index=True, sort=False)

# features (fingerprints)
X = data.groupby('molecule ID')['fingerprint'].first()
X = np.array([[float(bit) for bit in list(fg)] for fg in X.to_list()], dtype=np.float32)
# genotoxicity outcomes for each task
task_outcomes = data.pivot(index='molecule ID',columns='task',values='genotoxicity')



# outer and inner cross-validation splits (in the interest of reproducibility, the order is deterministic for all splits)
# the indices stored aare the molecules IDs
cv_outer = StratifiedKFold(n_splits=K_FOLD_OUTER, random_state=PYTORCH_SEED, shuffle=True)
cv_inner = StratifiedKFold(n_splits=K_FOLD_INNER, random_state=PYTORCH_SEED, shuffle=True)
splits = []
for task in task_outcomes.columns:
    task_molecule_ids = pd.Series(task_outcomes[task].dropna().index)
    y_task = task_outcomes[task].dropna().to_list()
    y_task_dist = dict(Counter(y_task))
    per_pos_task = 100 * y_task_dist['positive'] / len(y_task)
    log.info(f"task {task}: {len(y_task):4d} data points, positives: {y_task_dist['positive']:4d} ({per_pos_task:.2f}%)")

    for i_outer, (tmp_indices, test_indices) in enumerate(cv_outer.split(range(len(task_molecule_ids)), y=y_task)):
        y_outer_test = [y_task[i_mol] for i_mol in test_indices]
        y_outer_test_dist = dict(Counter(y_outer_test))
        per_pos_test = 100 * y_outer_test_dist['positive'] / len(y_outer_test)
        log.info(f"task {task}, outer fold {i_outer}, test set: {len(y_outer_test):4d} data points, positives: {y_outer_test_dist['positive']:4d} ({per_pos_test:.2f}%)")

        y_inner = [y_task[i_mol] for i_mol in tmp_indices]
        for i_inner, (train_indices, evaluate_indices) in enumerate(cv_inner.split(range(len(tmp_indices)), y=y_inner)):
            train_indices = [tmp_indices[i] for i in train_indices]
            evaluate_indices = [tmp_indices[i] for i in evaluate_indices]
            y_inner_train = [y_task[i] for i in train_indices]
            y_inner_train_dist = dict(Counter(y_inner_train))
            per_pos_inner_train = 100 * y_inner_train_dist['positive'] / len(y_inner_train)
            log.info(f"task {task}, outer fold {i_outer}, inner fold {i_inner}, train set: {len(y_inner_train):4d} data points, positives: {y_inner_train_dist['positive']:4d} ({per_pos_inner_train:.2f}%)")
            entry = {'task': task, 'outer fold': i_outer, 'inner fold': i_inner,
                     'train indices': pd.Series(train_indices), # task_molecule_ids.iloc[train_indices],
                     'eval indices': pd.Series(evaluate_indices), # task_molecule_ids.iloc[evaluate_indices],
                     'test indices': pd.Series(test_indices), # task_molecule_ids.iloc[test_indices],
                     'task # data points': len(y_task),
                     'test # data points': len(test_indices),
                     'train # data points': len(train_indices),
                     'task %positives': per_pos_task,
                     'test %positives': per_pos_test,
                     'train %positives': per_pos_inner_train,
                     }
            splits.append(entry)
splits = pd.DataFrame(splits)
for task in task_outcomes.columns:
    tmp = splits.loc[splits['task']==task, ['task %positives', 'test %positives', 'train %positives']].describe().loc[['min', 'max']].to_markdown()
    log.info(f'task {task}, %positives in different splits\n{tmp}')


# build the PyTorch datasets, no split at this stage
dsets = {}
for i_task, task in enumerate(task_outcomes.columns):
    log.info(f'preparing PyTorch dataset for task: {task}')
    entry = {}
    if HANDLE_AMBIGUOUS == 'ignore':
        msk = task_outcomes[task].isin(['positive', 'negative'])
        task_molecule_ids = task_outcomes.loc[msk].index.to_list()
        X_task = X[task_outcomes.index.isin(task_molecule_ids), :]
        y_task = task_outcomes.loc[msk, task].apply(lambda genotoxicity: 1 if genotoxicity == 'positive' else 0).values
    elif HANDLE_AMBIGUOUS == 'set_positive':
        msk = task_outcomes[task].isin(['positive', 'negative', 'ambiguous'])
        task_molecule_ids = task_outcomes.loc[msk].index.to_list()
        X_task = X[task_outcomes.index.isin(task_molecule_ids), :]
        y_task = task_outcomes.loc[msk, task].apply(lambda genotoxicity: 1 if genotoxicity == 'positive' or genotoxicity == 'ambiguous' else 0).values
    elif HANDLE_AMBIGUOUS == 'set_negative':
        msk = task_outcomes[task].isin(['positive', 'negative', 'ambiguous'])
        task_molecule_ids = task_outcomes.loc[msk].index.to_list()
        X_task = X[task_outcomes.index.isin(task_molecule_ids), :]
        y_task = task_outcomes.loc[msk, task].apply(lambda genotoxicity: 1 if genotoxicity == 'positive' else 0).values
    elif HANDLE_AMBIGUOUS == 'keep':
        msk = task_outcomes[task].isin(['positive', 'negative', 'ambiguous'])
        task_molecule_ids = task_outcomes.loc[msk].index.to_list()
        X_task = X[task_outcomes.index.isin(task_molecule_ids), :]
        y_task = task_outcomes.loc[msk, task].apply(lambda genotoxicity: 1 if genotoxicity == 'positive' else 0 if genotoxicity == 'negative' else 2).values
    else:
        ex = ValueError(f"unknown handling of ambiguous outcomes: {HANDLE_AMBIGUOUS}")
        log.error(ex)
        raise ex
    dset = TensorDataset(torch.tensor(X_task), torch.tensor(y_task))
    # dset.to(device) # all datasets are moved to the device
    # store the dataset in the dset dictionary
    entry['dset'] = dset
    if len(dset) >= MINIMUM_TASK_DATASET:
       dsets[task] = entry
    else:
        log.warning(f'task {task} has less than {MINIMUM_TASK_DATASET} data points, skipping')



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
# .. output the configurations
pd.DataFrame(configurations).to_excel(metrics_history_path/'configurations.xlsx', index=False)

# set the model
model = FFNNModel


# nested cross-validation
nested_cross_validation(model,
                        dsets,
                        splits,
                        configurations,
                        fingerprint_parameters,
                        PYTORCH_SEED,
                        BATCH_SIZE_MAX,
                        NUM_EPOCHS,
                        SCALE_LOSS_CLASS_SIZE,
                        SCALE_LOSS_TASK_SIZE,
                        LOG_EPOCH_FREQUENCY,
                        device,
                        metrics_history_path)



# consolidate the metrics for each outer iteration and list the optimal configuration for each outer iteration
consolidate_metrics_outer(metrics_history_path/'metrics_history.tsv',
                          metrics_history_path/'metrics_history_outer.xlsx',
                          task_names=list(dsets.keys()),
                          configuration_parameters=list(configurations[0].keys()))


# plot the average metrics for all outer iterations as a function of epoch (range is shown as a shaded area)
task_names = [f'task {i_task}' for i_task in range(len(tasks))]
plot_metrics_convergence_outer_average(metrics_history_path/'metrics_history.tsv',
                                       metrics_history_path/'metrics_convergence_outer_average.png',
                                       task_names=task_names)


# plot the average ROC for all outer iterations (range is shown as a shaded area)
roc_curve_outer_path = metrics_history_path/'roc_outer_average.png'
plot_roc_curve_outer_average(metrics_history_path, roc_curve_outer_path)


























#  ------------------- delete after this point
#
#
# # compute the overall fraction of positives (for all tasks)
# y_all = []
# for task in dsets:
#     dest = dsets[task]['dset']
#     y_all.extend(dset.tensors[1].cpu().numpy().tolist())
# fraction_positives = sum([1 for y in y_all if y == 1]) / len(y_all)
#
#
# # outer loop of the nested cross-validation
# for i_outer in range(K_FOLD_OUTER):
# # for i_outer in [0,1,2]: # DO NOT FORGET TO UNCOMMENT for limited runs !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     log.info(f'Initiating outer iteration {i_outer}')
#
#     if len(configurations) == 1:
#         best_configuration_ID = 0
#     else:
#         # loop over the model configurations
#         for i_configuration, configuration in enumerate(configurations, 0):
#             configuration_ID = configuration['configuration_ID']
#             log.info(f'Trialing model/optimiser configuration {configuration_ID} ({i_configuration+1} out of {len(configurations)})')
#             model_parameters = {k: v for k, v in configuration.items() if k not in ['configuration_ID', 'learning_rate', 'weight_decay']}
#             optimiser_parameters = {k: v for k, v in configuration.items() if k in ['configuration_ID', 'learning_rate', 'weight_decay']}
#
#             # inner loop of the nested cross-validation
#             metrics_history_configuration = []
#             for i_inner in range(K_FOLD_INNER):
#                 log.info(f'Initiating inner iteration {i_inner}')
#                 # .. create the train and eval set loaders
#                 train_loaders, eval_loaders = [], []
#                 msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == i_inner)
#                 train_set_size_max = max(len(idxs) for idxs in splits.loc[msk, 'train indices'])  # largest train set size among tasks
#                 eval_set_size_max = max(len(idxs) for idxs in splits.loc[msk, 'eval indices'])  # largest eval set size among tasks
#                 for task in dsets:
#                     msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == i_inner) & (splits['task'] == task)
#                     train_set = [rec for idx, rec in enumerate(dsets[task]['dset']) if idx in splits.loc[msk, 'train indices'].iloc[0].tolist()]
#                     batch_size = round(BATCH_SIZE_MAX * len(train_set) / float(train_set_size_max))
#                     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True) # .. we drop the last to have stable gradients
#                     train_loaders.append(train_loader)
#                     eval_set = [rec for idx, rec in enumerate(dsets[task]['dset']) if idx in splits.loc[msk, 'eval indices'].iloc[0].tolist()]
#                     batch_size = round(BATCH_SIZE_MAX * len(eval_set) / float(eval_set_size_max))
#                     eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=False)
#                     eval_loaders.append(eval_loader)
#                     log.info(f'task {task}, train set: {len(train_set):4d} data points in {len(train_loader)} batches, eval set: {len(eval_set):4d} data points in {len(eval_loader)} batches')
#
#                 torch.manual_seed(PYTORCH_SEED)
#                 if torch.cuda.is_available():
#                     torch.cuda.manual_seed_all(PYTORCH_SEED)
#
#                 # set the model
#                 n_classes = [2] * len(dsets)
#                 net = FFNNModel(n_input=fingerprint_parameters['fpSize'], **model_parameters, n_classes=n_classes)
#                 net.to(device)
#
#                 # if specified, scale the loss so that each class contributes according to its size or equally
#                 # default reduction is mean
#                 if SCALE_LOSS_CLASS_SIZE is None:
#                     global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
#                 elif SCALE_LOSS_CLASS_SIZE == 'equal class (global)':
#                     global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([fraction_positives, 1.-fraction_positives]))
#                 elif SCALE_LOSS_CLASS_SIZE == 'equal class (task)':
#                     # in this case we define a separate loss function per task in the train_eval function
#                     global_loss_fn = None
#
#                 # optimiser
#                 optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'], betas=[0.9, 0.999], eps=1e-08, weight_decay=optimiser_parameters['weight_decay'], amsgrad=False)
#
#                 # scheduler
#                 lambda_group = lambda epoch: 0.97 ** epoch
#                 scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])
#
#                 # train the model
#                 outp = metrics_history_path/f'outer_fold_{i_outer}_configuration_ID_{configuration_ID}_inner_fold_{i_inner}'
#                 outp.mkdir(parents=True, exist_ok=True)
#                 metrics_history = train_eval(net, train_loaders, eval_loaders, global_loss_fn, optimizer, scheduler, NUM_EPOCHS, outp/'model_weights_diff_quantiles.tsv', log_epoch_frequency=LOG_EPOCH_FREQUENCY, scale_loss_task_size=SCALE_LOSS_TASK_SIZE)
#
#                 # log the metrics for the training set and evaluation set
#                 metrics_history = pd.DataFrame(metrics_history)
#                 # remove the roc columns so that we can store the metrics as a dataframe
#                 metrics_history = metrics_history.drop(columns=['roc'], axis='columns')
#                 cols = {'time': datetime.now(), 'outer fold': i_outer}
#                 cols.update(configuration)
#                 cols.update({'inner fold': i_inner})
#                 for i_col, (col_name, col_value) in enumerate(cols.items()):
#                     metrics_history.insert(i_col, col_name, str(col_value))
#                 with open(metrics_history_path/'metrics_history.tsv', mode='at', encoding='utf-8', buffering=1, newline='') as f:
#                     metrics_history.to_csv(f, header=f.tell()==0, index=False, sep='\t', lineterminator='\n')
#
#                 # plot the metrics
#                 task_names = list(dsets.keys())
#                 plot_metrics_convergence(metrics_history, task_names=task_names, stages=['train', 'eval'], output=outp)
#
#
#         # find the optimal configuration by using the average eval balanced accuracy over the inner folds (avoid reading the whole file in memory)
#         chunk_iterator = pd.read_csv(metrics_history_path/'metrics_history.tsv', chunksize=10_000, sep='\t')
#         metrics_history_configuration = []
#         for chunk in chunk_iterator:
#             # .. select the eval rows that provide the aggregate metrics across tasks and batches
#             msk = (chunk['batch'].isnull()) & (chunk['task'].isnull()) & (chunk['stage'] == 'eval') & (chunk['type'] == 'aggregate (epoch)')
#             metrics_history_configuration.append(chunk.loc[msk])
#         metrics_history_configuration = pd.concat(metrics_history_configuration, axis=0, sort=False, ignore_index=True)
#         # .. keep the last three epochs
#         msk = (metrics_history_configuration['epoch'] >= metrics_history_configuration['epoch'].max() - 3)
#         metrics_history_configuration = metrics_history_configuration.loc[msk]
#         balanced_accuracy_eval_inner_folds = metrics_history_configuration.groupby('configuration_ID')['balanced accuracy'].mean()
#         best_configuration_ID = balanced_accuracy_eval_inner_folds.idxmax()
#         log.info(f'outer fold {i_outer}, best configuration ID: {best_configuration_ID} with balanced accuracy: {balanced_accuracy_eval_inner_folds.max():.4} (range: {balanced_accuracy_eval_inner_folds.min():.4} - {balanced_accuracy_eval_inner_folds.max():.4})')
#
#
#
#     # refit using the whole train + eval sets and evaluate in the test set
#     msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == 0)
#     train_eval_set_size_max = max(len(idxs_train.tolist()+idxs_eval.tolist()) for idxs_train, idxs_eval in zip(splits.loc[msk, 'train indices'], splits.loc[msk, 'eval indices']))
#     test_set_size_max = max(len(idxs) for idxs in splits.loc[msk, 'test indices'])  # largest test set size among tasks
#     train_eval_loaders, test_loaders = [], []
#     for task in dsets:
#         msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == 0) & (splits['task'] == task)
#         train_eval_set = [rec for idx, rec in enumerate(dsets[task]['dset']) if idx in splits.loc[msk, 'train indices'].iloc[0].tolist()+splits.loc[msk, 'eval indices'].iloc[0].tolist()]
#         batch_size = math.ceil(BATCH_SIZE_MAX * len(train_eval_set) / float(train_eval_set_size_max))
#         train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size, shuffle=True, drop_last=True) # .. we drop the last to have stable gradients
#         train_eval_loaders.append(train_eval_loader)
#         test_set = [rec for idx, rec in enumerate(dsets[task]['dset']) if idx in splits.loc[msk, 'test indices'].iloc[0].tolist()]
#         batch_size = math.ceil(BATCH_SIZE_MAX * len(test_set) / float(test_set_size_max))
#         test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)
#         test_loaders.append(test_loader)
#     # .. train the model
#     # .. set the seed for reproducibility
#     torch.manual_seed(PYTORCH_SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(PYTORCH_SEED)
#
#     # set the model
#     configuration = [configuration for configuration in configurations if configuration['configuration_ID'] == best_configuration_ID][0]
#     configuration_ID = configuration['configuration_ID']
#     model_parameters = {k: v for k, v in configuration.items() if k not in ['configuration_ID', 'learning_rate', 'weight_decay']}
#     optimiser_parameters = {k: v for k, v in configuration.items() if k in ['configuration_ID', 'learning_rate', 'weight_decay']}
#     n_classes = [2] * len(dsets)
#     net = FFNNModel(n_input=fingerprint_parameters['fpSize'], **model_parameters, n_classes=n_classes)
#     net.to(device)
#
#     # if specified, scale the loss so that each class contributes according to its size or equally
#     # default reduction is mean
#     if SCALE_LOSS_CLASS_SIZE is None:
#         global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
#     elif SCALE_LOSS_CLASS_SIZE == 'equal class (global)':
#         global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([fraction_positives, 1. - fraction_positives]))
#     elif SCALE_LOSS_CLASS_SIZE == 'equal class (task)':
#         # in this case we define a separate loss function per task in the train_eval function
#         global_loss_fn = None
#
#     # optimiser
#     optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'], betas=[0.9, 0.999], eps=1e-08, weight_decay=optimiser_parameters['weight_decay'],
#                                  amsgrad=False)
#
#     # scheduler
#     lambda_group = lambda epoch: 0.97 ** epoch
#     scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])
#
#     # train the model, double the number of epochs for the final training
#     outp = metrics_history_path/f'outer_fold_{i_outer}_configuration_ID_{configuration_ID}'
#     outp.mkdir(parents=True, exist_ok=True)
#     metrics_history = train_eval(net, train_eval_loaders, test_loaders, global_loss_fn, optimizer, scheduler, 2*NUM_EPOCHS, outp=None, log_epoch_frequency=LOG_EPOCH_FREQUENCY, scale_loss_task_size=SCALE_LOSS_TASK_SIZE)
#
#     # log the metrics for the training set and evaluation set
#     metrics_history = pd.DataFrame(metrics_history)
#     # remove the roc columns so that we can store the metrics as a dataframe and store the roc for further processing
#     roc = (metrics_history
#            .dropna(subset=['roc'])
#            .filter(['epoch', 'stage', 'task', 'roc'], axis='columns')
#            .pipe(lambda df: df.assign(**{'stage': np.where(df['stage']=='train', 'train+eval', 'test')}))
#            )
#     roc.to_pickle(outp/'roc_outer.pkl')
#
#     metrics_history = metrics_history.drop(columns=['roc'], axis='columns')
#     cols = {'time': datetime.now(), 'outer fold': i_outer}
#     cols.update(configuration)
#     cols.update({'inner fold': None})
#     for i_col, (col_name, col_value) in enumerate(cols.items()):
#         metrics_history.insert(i_col, col_name, str(col_value))
#     metrics_history['stage'] = np.where(metrics_history['stage']=='train', 'train+eval', 'test')
#     with open(metrics_history_path/'metrics_history.tsv', mode='at', encoding='utf-8', buffering=1, newline='') as f:
#         metrics_history.to_csv(f, header=f.tell() == 0, index=False, sep='\t', lineterminator='\n')
#
#     # plot the metrics
#     task_names = list(dsets.keys())
#     plot_metrics_convergence(metrics_history, task_names=task_names, stages=['train+eval', 'test'], output=outp)
#
#     # save the model
#     torch.save(net, outp/'model.pth')
#
#
#
#
# # consolidate the metrics for each outer iteration and list the optimal configuration for each outer iteration
# consolidate_metrics_outer(metrics_history_path/'metrics_history.tsv',
#                           metrics_history_path/'metrics_history_outer.xlsx',
#                           task_names=list(dsets.keys()),
#                           configuration_parameters=list(configurations[0].keys()))
#
#
# # plot the average metrics for all outer iterations as a function of epoch (range is shown as a shaded area)
# task_names = [f'task {i_task}' for i_task in range(6)]
# plot_metrics_convergence_outer_average(metrics_history_path/'metrics_history.tsv',
#                                        metrics_history_path/'metrics_convergence_outer_average.png',
#                                        task_names=task_names)
#
#
# # plot the average ROC for all outer iterations (range is shown as a shaded area)
# roc_curve_outer_path = metrics_history_path/'roc_outer_average.png'
# plot_roc_curve_outer_average(metrics_history_path, roc_curve_outer_path)





