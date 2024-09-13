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
                r'data/REACH/tabular/REACH_genotoxicity.xlsx',
                r'data/QSARToolbox/tabular/QSARToolbox_genotoxicity.xlsx'
                ]
task_specifications = [
    {'filters': {'assay': ['bacterial reverse mutation assay'], 'cell line/species': [#'Escherichia coli (WP2 Uvr A)',
                                                                                      #'Salmonella typhimurium (TA 102)',
                                                                                      'Salmonella typhimurium (TA 100)',
                                                                                      #'Salmonella typhimurium (TA 1535)',
                                                                                      'Salmonella typhimurium (TA 98)',
                                                                                      #'Salmonella typhimurium (TA 1537)'
                                                                                      ], 'metabolic activation': ['yes', 'no']},
     'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation']},

    {'filters': {'assay': ['in vitro mammalian cell micronucleus test']},
     'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

    {'filters': {'assay': ['in vitro mammalian chromosome aberration test']},
     'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

    # {'filters': {'assay': ['in vitro mammalian cell gene mutation test using the Hprt and xprt genes']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},
    #
    # {'filters': {'assay': ['in vitro mammalian cell gene mutation test using the thymidine kinase gene']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

    {'filters': {'endpoint': ['in vitro gene mutation study in mammalian cells']},
     'task aggregation columns': ['in vitro/in vivo', 'endpoint']},

]
task_specifications = [
    {'filters': {'assay': ['bacterial reverse mutation assay'], },
     'task aggregation columns': ['in vitro/in vivo', 'endpoint']},
]
outp_sdf = Path(r'data/combined/sdf/genotoxicity_dataset.sdf')
outp_tab = Path(r'data/combined/tabular/genotoxicity_dataset.xlsx')
tasks = create_sdf(flat_datasets = flat_datasets,
                   task_specifications = task_specifications,
                   outp_sdf = outp_sdf,
                   outp_tab = outp_tab)


# set general parameters
PYTORCH_SEED = 1 # seed for PyTorch random number generator, it is also used for splits and shuffling to ensure reproducibility
MINIMUM_TASK_DATASET = 512 # minimum number of data points for a task
BATCH_SIZE_MAX = 1024 # maximum batch size (largest task, the smaller tasks are scaled accordingly so the number of batches is the same)
K_FOLD_INNER = 5 # number of folds for the inner cross-validation
K_FOLD_OUTER = 10 # number of folds for the outer cross-validation
NUM_EPOCHS = 80 # number of epochs
HANDLE_AMBIGUOUS = 'ignore' # how to handle ambiguous outcomes, can be 'keep', 'set_positive', 'set_negative' or 'ignore', but the model fitting does not support 'keep'
MODEL_NAME = 'FFNN'

SCALE_LOSS_TASK_SIZE = None # how to scale the loss function, can be 'equal task' or None
SCALE_LOSS_CLASS_SIZE = 'equal class (task)' # how to scale the loss function, can be 'equal class (task)', 'equal class (global)' or None

# location to store the metrics logs
metrics_history_path = Path(rf'D:\myApplications\local\2024_01_21_GCN_Muta\output\iteration103')/MODEL_NAME
metrics_history_path.mkdir(parents=True, exist_ok=True)

fingerprint_parameters = {'radius': 2,
                          'fpSize': 1024,
                          'type': 'binary' # 'binary', 'count'
                          }

model_parameters = {'hidden_layers': [[256, 256]],  # [64, 128, 256]
                    'dropout': [0.6, 0.7],  # [0.5, 0.6, 0.7, 0.8],
                    'activation_function': [torch.nn.functional.leaky_relu],
                    'learning_rate': [0.005],  # [0.001, 0.005, 0.01]
                    'weight_decay': [1.e-3, 5.e-5],  # [1.e-5, 1e-4, 1e-3]
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
            tox_data['fingerprint'] = ''.join([str(bit) for bit in fg.ToList()]) #fg.ToBitString()
        elif fingerprint_parameters['type'] == 'count':
            tox_data['fingerprint'] = ''.join([str(min(bit, 9)) for bit in fg_count.ToList()])
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
# .. output the tasks
splits.to_excel(metrics_history_path/'splits.xlsx', index=False)

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


# report the balanced accuracy for the test set
metrics_history_outer = pd.read_excel(metrics_history_path/'metrics_history_outer.xlsx', sheet_name='task')
cols = ['outer fold'] + [col for col in metrics_history_outer.columns if 'test_balanced accuracy' in col]
metrics_history_outer_summary = metrics_history_outer.loc[metrics_history_outer['outer fold']!='All', cols].melt(id_vars='outer fold', var_name='task', value_name='balanced accuracy (test)')
log.info(metrics_history_outer_summary.groupby('task')['balanced accuracy (test)'].agg(['mean', 'min', 'max']))













