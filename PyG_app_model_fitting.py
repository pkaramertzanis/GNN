# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_muta_model.log')

# import and configure pandas globally (this needs to be imported first as other modules import pandas too)
import pandas_config
import pandas as pd

from pathlib import Path
import numpy as np
import torch

from models.PyG_Dataset import PyG_Dataset

from collections import Counter
from itertools import product
import random

from sklearn.model_selection import StratifiedKFold

from data.combine import create_sdf

from models.MPNN_GNN.MPNN_GNN import MPNN_GNN
from models.AttentiveFP_GNN.AttentiveFP_GNN import AttentiveFP_GNN
from models.GAT_GNN.GAT_GNN import GAT_GNN

from models.metrics import (consolidate_metrics_outer,
                            plot_metrics_convergence_outer_average, plot_roc_curve_outer_average)
from models.PyG_nested_cross_validation import nested_cross_validation

from visualisations.task_concordance import visualise_task_concordance

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set up the dataset
flat_datasets = [
                # r'data/Hansen_2009/tabular/Hansen_2009_genotoxicity.xlsx',
                # r'data/Leadscope/tabular/Leadscope_genotoxicity.xlsx',
                r'data/QSARToolbox/tabular/QSARToolbox_genotoxicity.xlsx',
                r'data/REACH/tabular/REACH_genotoxicity.xlsx',
]
task_specifications = [
    {'filters': {'assay': ['bacterial reverse mutation assay'], 'cell line/species': [#'Escherichia coli (WP2 Uvr A)',
                                                                                      #'Salmonella typhimurium (TA 102)',
                                                                                      'Salmonella typhimurium (TA 100)',
                                                                                      #'Salmonella typhimurium (TA 1535)',
                                                                                      # 'Salmonella typhimurium (TA 98)',
                                                                                      #'Salmonella typhimurium (TA 1537)'
                                                                                      ], 'metabolic activation': ['yes']},
     'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation']},

    # {'filters': {'assay': ['in vitro mammalian cell micronucleus test']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},
    #
    # {'filters': {'assay': ['in vitro mammalian chromosome aberration test']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

    # {'filters': {'assay': ['in vitro mammalian cell gene mutation test using the Hprt and xprt genes']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},
    #
    # {'filters': {'assay': ['in vitro mammalian cell gene mutation test using the thymidine kinase gene']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

    # {'filters': {'endpoint': ['in vitro gene mutation study in mammalian cells']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint']},

]
# task_specifications = [
#     {'filters': {'assay': ['bacterial reverse mutation assay'], },
#      'task aggregation columns': ['in vitro/in vivo', 'endpoint']},
# ]
# task_specifications = [
#     {'filters': {'assay': ['bacterial reverse mutation assay'], 'cell line/species': [#'Escherichia coli (WP2 Uvr A)',
#                                                                                       #'Salmonella typhimurium (TA 102)',
#                                                                                       'Salmonella typhimurium (TA 100)',
#                                                                                       #'Salmonella typhimurium (TA 1535)',
#                                                                                       'Salmonella typhimurium (TA 98)',
#                                                                                       #'Salmonella typhimurium (TA 1537)'
#                                                                                       ], 'metabolic activation': ['yes', 'no']},
#      'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay', 'cell line/species']},
#
# ]
outp_sdf = Path(r'data/combined/sdf/genotoxicity_dataset.sdf')
outp_tab = Path(r'data/combined/tabular/genotoxicity_dataset.xlsx')
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
NUM_EPOCHS = 500 # number of epochs
MODEL_NAME = 'AttentiveFP_GNN' # name of the model, can be 'MPNN_GNN', 'AttentiveFP_GNN' or 'GAT_GNN'
SCALE_LOSS_TASK_SIZE = None # how to scale the loss function, can be 'equal task' or None
SCALE_LOSS_CLASS_SIZE = 'equal class (task)' # how to scale the loss function, can be 'equal class (task)', 'equal class (global)' or None

HANDLE_AMBIGUOUS = 'ignore' # how to handle ambiguous outcomes, can be 'keep', 'set_positive', 'set_negative' or 'ignore', but the model fitting does not support 'keep'


# location to store the metrics logs
metrics_history_path = Path(rf'D:\myApplications\local\2024_01_21_GCN_Muta\output\iteration104')/MODEL_NAME
metrics_history_path.mkdir(parents=True, exist_ok=True)


# visualise the task concordance (if more than one task)
if len(tasks) > 1:
    visualise_task_concordance(outp_sdf, [metrics_history_path/'task_concordance.png',
                                                metrics_history_path / 'task_co-occurrence.png'])


# features, checkers and standardisers
NODE_FEATS = ['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization', 'num_rings', 'num_Hs']
EDGE_FEATS = ['bond_type', 'is_conjugated', 'num_rings'] # ['bond_type', 'is_conjugated', 'stereo_type']

# select model
if MODEL_NAME == 'MPNN_GNN':
    model = MPNN_GNN
    model_parameters = {'n_conv': [3], # [1, 2, 3, 4, 5, 6]
                        'n_lin': [1], # 1, 2, 3, 4]
                        'n_conv_hidden': [64], # [32, 64, 128, 256]
                        'n_edge_NN': [64], # [32, 64, 128, 256]
                        'n_lin_hidden': [64], # [32, 64, 128, 256, 512]
                        'dropout': [0.6], # [0.5, 0.6, 0.7, 0.8]
                        'activation_function': [torch.nn.functional.leaky_relu],
                        'learning_rate': [1.e-3],  # [0.001, 0.005, 0.01]
                        'weight_decay': [1.e-3],  # [1.e-5, 1e-4, 1e-3]
                        }
elif MODEL_NAME == 'AttentiveFP_GNN':
    model = AttentiveFP_GNN
    model_parameters = {'hidden_channels': [200], # [64, 128, 256]
                        'num_layers': [3], # [1, 2, 3, 4]
                        'num_timesteps': [3], # [1, 2, 3, 4]
                        'dropout': [0.5], # [0.5, 0.6, 0.7, 0.8]
                        'learning_rate': [10**(-3)], # [0.001, 0.005, 0.01]
                        'weight_decay': [10**(-3.5)],  # [1.e-5, 1e-4, 1.e-3]
                        }
elif MODEL_NAME == 'GAT_GNN':
    model = GAT_GNN
    model_parameters = {'n_conv': [6],
                        'n_lin': [1],  # 1, 2, 3, 4]
                        'n_heads': [3, 4],
                        'n_conv_hidden': [120, 240],
                        'n_lin_hidden': [64, 128],  # [32, 64, 128, 256, 512]
                        'v2': [True], # [True, False]
                        'dropout': [0.2],
                        'activation_function': [torch.nn.functional.leaky_relu],
                        'learning_rate': [0.005],  # [0.001, 0.005, 0.01]
                        'weight_decay': [1.e-4],  # [1.e-5, 1e-4, 1e-3]
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
    # move the dataset to device immediately after creation
    if dset.x.device != device:
        dset.to(device)
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
for i_task, task in enumerate(dsets):
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
            entry = {'task ID': i_task,
                     'task': task, 'outer fold': i_outer, 'inner fold': i_inner,
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
# .. output the tasks
splits.to_excel(metrics_history_path/'splits.xlsx', index=False)

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


# # --------------------------------------
# # export the dataset for the first outer iteration to build a model with deep chem
# msk = (splits['outer fold']==0) & (splits['inner fold']==0)
# train_indices = splits.loc[msk, 'train indices'].iloc[0]
# eval_indices = splits.loc[msk, 'eval indices'].iloc[0]
# test_indices = splits.loc[msk, 'test indices'].iloc[0]
# molecule_ids = dsets['in vitro, in vitro gene mutation study in bacteria']['dset'].data.molecule_id
# genotox_data = pd.read_excel('data/combined/tabular/genotoxicity_dataset.xlsx')
# # .. training/eval set
# molecule_ids_train_eval = [molecule_ids[i] for i in np.concatenate([train_indices, eval_indices])]
# msk = genotox_data['source record ID'].isin(molecule_ids_train_eval)
# train_eval_smiles = genotox_data.loc[msk, 'smiles_std']
# train_eval_y = [1 if gentox=='positive' else 0 for gentox in genotox_data.loc[msk, 'genotoxicity'].to_list()]
# # .. training/eval set
# molecule_ids_test = [molecule_ids[i] for i in test_indices]
# msk = genotox_data['source record ID'].isin(molecule_ids_test)
# test_smiles = genotox_data.loc[msk, 'smiles_std']
# test_y = [1 if gentox=='positive' else 0 for gentox in genotox_data.loc[msk, 'genotoxicity'].to_list()]
# # store train_eval_smiles, train_eval_y, test_smiles, test_y in a pickle
# import pickle
# with open('junk/deepchem_data.pickle', 'wb') as f:
#     pickle.dump((train_eval_smiles, train_eval_y, test_smiles, test_y), f)
# # --------------------------------------


# nested cross-validation
nested_cross_validation(model,
                        dsets,
                        splits,
                        configurations,
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

# ----------  delete after this line




