# setup logging
from optuna.trial import TrialState

import logger
import logging
log = logger.setup_applevel_logger(file_name ='logs/FFNN_model_fit.log', level_stream=logging.DEBUG, level_file=logging.DEBUG)

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
import pickle

import torch
from torch.utils.data import TensorDataset

from sklearn.model_selection import StratifiedKFold

from data.combine import create_sdf

from models.FFNN.FFNN import FFNNModel

from models.metrics import plot_metrics_convergence

from models.PyTorch_train import train_eval

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

import json
from cheminformatics.rdkit_toolkit import read_sdf
from rdkit.Chem import AllChem
from tqdm import tqdm

# set the model architecture
MODEL_NAME = 'FFNN'
STUDY_NAME = "Ames_5_strains" # name of the study in the Optuna sqlit database

# location to store the results
output_path = Path(rf'output/{MODEL_NAME}/{STUDY_NAME}')
# output_path = Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\inference\baderna_2020\training_eval_dataset')
output_path.mkdir(parents=True, exist_ok=True)

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set up the dataset
flat_datasets = [
                # r'data/Hansen_2009/tabular/Hansen_2009_genotoxicity.xlsx',
                # r'data/Leadscope/tabular/Leadscope_genotoxicity.xlsx',
                r'data/ECVAM_AmesNegative/tabular/ECVAM_Ames_negative_genotoxicity.xlsx',
                r'data/ECVAM_AmesPositive/tabular/ECVAM_Ames_positive_genotoxicity.xlsx',
                r'data/QSARChallengeProject/tabular/QSARChallengeProject.xlsx',
                r'data/QSARToolbox/tabular/QSARToolbox_genotoxicity.xlsx',
                r'data/REACH/tabular/REACH_genotoxicity.xlsx',
                # r'data/Baderna_2020/tabular/Baderna_2020_genotoxicity.xlsx',
]
task_specifications = [
       {'filters': {'assay': ['bacterial reverse mutation assay'], 'cell line/species': ['Escherichia coli (WP2 Uvr A)',
                                                                                         # 'Salmonella typhimurium (TA 102)',
                                                                                          'Salmonella typhimurium (TA 100)',
                                                                                          'Salmonella typhimurium (TA 1535)',
                                                                                         'Salmonella typhimurium (TA 98)',
                                                                                          'Salmonella typhimurium (TA 1537)'
                                                                                         ], 'metabolic activation': ['yes',
                                                                                                                      'no'
                                                                                                                     ]},
        'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation']},

     # {'filters': {'assay': ['bacterial reverse mutation assay']},
     #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

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
#      {'filters': {'assay': ['bacterial reverse mutation assay'], },
#       'task aggregation columns': ['in vitro/in vivo', 'endpoint']},
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
training_eval_dataset_path_tabular = Path(output_path / 'training_eval_dataset/tabular')
training_eval_dataset_path_tabular.mkdir(parents=True, exist_ok=True)
training_eval_dataset_path_sdf = Path(output_path / 'training_eval_dataset/sdf')
training_eval_dataset_path_sdf.mkdir(parents=True, exist_ok=True)
outp_sdf = Path(training_eval_dataset_path_sdf/'genotoxicity_dataset.sdf')
outp_tab = Path(training_eval_dataset_path_tabular/'genotoxicity_dataset.xlsx')
tasks = create_sdf(flat_datasets = flat_datasets,
                   task_specifications = task_specifications,
                   outp_sdf = outp_sdf,
                   outp_tab = outp_tab)

# set general parameters
N_TRIALS = 200 # number of trials to be attempted by the Optuna optimiser
PYTORCH_SEED = 2 # seed for PyTorch random number generator, it is also used for splits and shuffling to ensure reproducibility
MINIMUM_TASK_DATASET = 300 # minimum number of data points for a task
BATCH_SIZE_MAX = 1024 # maximum batch size (largest task, the smaller tasks are scaled accordingly so the number of batches is the same)
K_FOLD = 5 # number of folds for the cross-validation
MAX_NUM_EPOCHS = 500 # maximum number of epochs
SCALE_LOSS_TASK_SIZE = None # how to scale the loss function, can be 'equal task' or None
SCALE_LOSS_CLASS_SIZE = 'equal class (task)' # how to scale the loss function, can be 'equal class (task)', 'equal class (global)' or None
HANDLE_AMBIGUOUS = 'ignore' # how to handle ambiguous outcomes, can be 'keep', 'set_positive', 'set_negative' or 'ignore', but the model fitting does not support 'keep'
DROP_LAST_TRAINING = True # we can drop the last to have stable gradients and possibly NAN loss function due to lack of positives
LOG_EPOCH_FREQUENCY = 10
EARLY_STOPPING_LOSS_EVAL = 20
EARLY_STOPPING_ROC_EVAL = 10
EARLY_STOPPING_THRESHOLD = 1.e-3
NUMBER_MODEL_FITS = 3  # number of model fits in each fold




# fingerprint parameters
fingerprint_parameters = {'radius': 2,
                          'fpSize': 2048,
                          'type': 'count' # 'binary', 'count'
                          }

# hyperparameter search space
hyperparameters = {
    'model parameters': {
        'size_hidden_layers': IntDistribution(low=50, high=400, log=False, step=25),
        'number_hidden_layers': IntDistribution(low=1, high=4, log=False, step=1),
        'dropout': FloatDistribution(low=0.0, high=0.8, step=None, log=False),
    },
    'optimiser parameters': {
        'learning_rate': FloatDistribution(low=1.e-5, high=1.e-2, step=None, log=True),
        'weight_decay': FloatDistribution(low=1.e-7, high=1.e-2, step=None, log=True),
        'scheduler_decay': FloatDistribution(low=0.94, high=0.99, step=None, log=False)
    }
}



# read in the molecular structures from the SDF file, and compute fingerprints
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

# store the fingerprint parameters and task names
with open(output_path/'fingerprint_tasks.json', 'w') as f:
    fingerprint_tasks = {'fingerprint parameters': fingerprint_parameters,
                         'tasks': list(task_outcomes.columns)}
    json.dump(fingerprint_tasks, f)

# set up the cross-validation splits
cv = StratifiedKFold(n_splits=K_FOLD, random_state=PYTORCH_SEED, shuffle=True)
splits = []
for task in task_outcomes.columns:
    task_molecule_ids = pd.Series(task_outcomes[task].dropna().index)
    y_task = task_outcomes[task].dropna().to_list()
    y_task_dist = dict(Counter(y_task))
    per_pos_task = 100 * y_task_dist['positive'] / len(y_task)
    log.info(f"task {task}: {len(y_task):4d} data points, positives: {y_task_dist['positive']:4d} ({per_pos_task:.2f}%)")

    for i_fold, (train_indices, eval_indices) in enumerate(cv.split(range(len(task_molecule_ids)), y=y_task)):
        # train set
        y_train = [y_task[i_mol] for i_mol in train_indices]
        y_train_dist = dict(Counter(y_train))
        per_pos_train = 100 * y_train_dist['positive'] / len(y_train)
        log.info(f"task {task}, fold {i_fold}, train set: {len(y_train):4d} data points, positives: {y_train_dist['positive']:4d} ({per_pos_train:.2f}%)")
        # eval set
        y_eval = [y_task[i_mol] for i_mol in eval_indices]
        y_eval_dist = dict(Counter(y_eval))
        per_pos_eval = 100 * y_eval_dist['positive'] / len(y_eval)
        log.info(f"task {task}, fold {i_fold}, eval set: {len(y_eval):4d} data points, positives: {y_eval_dist['positive']:4d} ({per_pos_eval:.2f}%)")
        # store the split
        entry = {'task': task, 'fold': i_fold,
                 'train indices': pd.Series(train_indices),  # task_molecule_ids.iloc[train_indices],
                 'eval indices': pd.Series(eval_indices),  # task_molecule_ids.iloc[evaluate_indices],
                 'task # data points': len(y_task),
                 'eval # data points': len(eval_indices),
                 'train # data points': len(train_indices),
                 'task %positives': per_pos_task,
                 'eval %positives': per_pos_eval,
                 'train %positives': per_pos_train,
                 }
        splits.append(entry)
splits = pd.DataFrame(splits)




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


# compute the overall fraction of positives (for all tasks)
y_all = []
for task in dsets:
    dset = dsets[task]['dset']
    y_all.extend(dset.tensors[1].cpu().numpy().tolist())
fraction_positives = sum([1 for y in y_all if y == 1]) / len(y_all)



def objective(trial) -> float:
    '''
    Optuna objective function to optimise the hyperparameters by minimising the balanced accuracy
    :param trial: Optuna trial
    :return: balanced accuracy for the eval set
    '''

    try:

        # fetch the model and optimiser parameters
        model_parameters = dict()
        for parameter in hyperparameters['model parameters']:
            model_parameters[parameter] = trial._suggest(parameter, hyperparameters['model parameters'][parameter])

        optimiser_parameters = dict()
        for parameter in hyperparameters['optimiser parameters']:
            # optimiser_parameters[parameter.name] = getattr(trial, parameter.type)(parameter.name, parameter.lower_bound, parameter.upper_bound, log=parameter.log)
            optimiser_parameters[parameter] = trial._suggest(parameter, hyperparameters['optimiser parameters'][parameter])
        log.info(f'attempting configuration with the following parameters: {trial.params}')


        metrics_history_trial = []
        for i_fold in tqdm(range(K_FOLD), desc='inner loop', total=K_FOLD):
            log.info(f'Initiating fold {i_fold}')
            # .. create the train and eval set loaders
            train_loaders, eval_loaders = [], []
            msk = splits['fold'] == i_fold
            train_set_size_max = max(
                len(idxs) for idxs in splits.loc[msk, 'train indices'])  # largest train set size among tasks
            eval_set_size_max = max(
                len(idxs) for idxs in splits.loc[msk, 'eval indices'])  # largest eval set size among tasks
            for task in dsets:
                msk = (splits['fold'] == i_fold) & (splits['task'] == task)
                train_set = [rec for idx, rec in enumerate(dsets[task]['dset']) if idx in splits.loc[msk, 'train indices'].iloc[0].tolist()]
                batch_size = round(BATCH_SIZE_MAX * len(train_set) / float(train_set_size_max))
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=DROP_LAST_TRAINING if len(train_set) > batch_size else False)
                train_loaders.append(train_loader)
                eval_set = [rec for idx, rec in enumerate(dsets[task]['dset']) if idx in splits.loc[msk, 'eval indices'].iloc[0].tolist()]
                batch_size = round(BATCH_SIZE_MAX * len(eval_set) / float(eval_set_size_max))
                eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=False)
                eval_loaders.append(eval_loader)
                log.info(f'task {task}, train set: {len(train_set):4d} data points in {len(train_loader)} batches, eval set: {len(eval_set):4d} data points in {len(eval_loader)} batches')

            # number of classes
            n_classes = [2] * len(dsets)

            # if specified, scale the loss so that each class contributes according to its size or equally
            # default reduction is mean
            if SCALE_LOSS_CLASS_SIZE is None:
                global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]).to(device))
            elif SCALE_LOSS_CLASS_SIZE == 'equal class (global)':
                global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([fraction_positives, 1. - fraction_positives]).to(device))
            elif SCALE_LOSS_CLASS_SIZE == 'equal class (task)':
                # in this case we define a separate loss function per task in the train_eval function
                global_loss_fn = None


            # reset the seed for deterministic behaviour
            torch.manual_seed(PYTORCH_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(PYTORCH_SEED)

            # repeat model fits with different initial seeds
            for i_model_fit in range(NUMBER_MODEL_FITS):
                log.info(f'Initiating model fit {i_model_fit}')

                # set up the model
                hidden_layers = [model_parameters['size_hidden_layers']] * model_parameters['number_hidden_layers']
                net = FFNNModel(n_input=fingerprint_parameters['fpSize'], hidden_layers=hidden_layers, dropout=model_parameters['dropout'], n_classes=n_classes)
                net.to(device)
                # set up the optimiser
                optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'],
                                             betas=[0.9, 0.999], eps=1e-08,
                                             weight_decay=optimiser_parameters['weight_decay'], amsgrad=False)
                # set up the scheduler
                lambda_group = lambda epoch: optimiser_parameters['scheduler_decay'] ** epoch
                scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])

                # train the model
                metrics_history = train_eval(net, train_loaders, eval_loaders, global_loss_fn,
                                             optimizer, scheduler, MAX_NUM_EPOCHS,
                                             weight_converge_path=None,
                                             early_stopping={'loss_eval': EARLY_STOPPING_LOSS_EVAL,
                                                             'roc_eval': EARLY_STOPPING_ROC_EVAL,
                                                             'threshold': EARLY_STOPPING_THRESHOLD},
                                             log_epoch_frequency=LOG_EPOCH_FREQUENCY,
                                             scale_loss_task_size=SCALE_LOSS_TASK_SIZE)

                # store the metrics
                metrics_history = pd.DataFrame(metrics_history)
                metrics_history.insert(0, 'fold', i_fold)
                metrics_history.insert(0, 'model fit', i_model_fit)
                for col in reversed(optimiser_parameters):
                    metrics_history.insert(0, col, optimiser_parameters[col])
                print(model_parameters)
                for col in reversed(model_parameters):
                    metrics_history.insert(0, col, model_parameters[col])
                metrics_history_trial.append(metrics_history)

                # create folder to store the model fitting results
                outp = output_path / f'trial_{trial.number}_fold_{i_fold}_model_fit_{i_model_fit}'
                outp.mkdir(parents=True, exist_ok=True)

                # plot and save the model convergence
                task_names = list(dsets.keys())
                plot_metrics_convergence(metrics_history, task_names=task_names, stages=['train', 'eval'],
                                         output=outp)

                # save the model
                torch.save(net, outp / 'model.pth')

                # save the metrics history
                metrics_history.to_excel(outp / 'metrics_history.xlsx', index=False)

        # compute the objective function value as the mean for all folds, taking the max for all model fits in the fold
        metrics_history_trial = pd.concat(metrics_history_trial, axis='index', sort=False, ignore_index=True)
        msk = (metrics_history_trial['type'] == 'aggregate (epoch)') & metrics_history_trial['task'].isnull() & (metrics_history_trial['stage'] == 'eval')
        objective_function_value = metrics_history_trial.loc[msk].groupby(['fold'])['balanced accuracy'].max().mean()

        # set the metrics for each task for the evaluation set as a user attribute in the study (this is taken simply as the mean of the balanced accuracy for all model fits in the fold)
        for i_task, task in enumerate(dsets):
            ba_eval_task = []
            for i_fold in range(K_FOLD):
                for i_model_fit in range(NUMBER_MODEL_FITS):
                    msk = ((metrics_history_trial['fold'] == i_fold)
                           & (metrics_history_trial['model fit'] == i_model_fit)
                           & (metrics_history_trial['type'] == 'aggregate (epoch)') & metrics_history_trial['task'].isnull() & (metrics_history_trial['stage'] == 'eval'))
                    best_epoch = metrics_history_trial.loc[msk].set_index('epoch')['balanced accuracy'].idxmax()
                    msk = ( (metrics_history_trial['epoch'] == best_epoch)
                           & (metrics_history_trial['fold'] == i_fold)
                           & (metrics_history_trial['model fit'] == i_model_fit)
                           & (metrics_history_trial['type'] == 'aggregate (epoch)') & (metrics_history_trial['task'] == i_task) & (metrics_history_trial['stage'] == 'eval'))
                    ba_eval_task.append(metrics_history_trial.loc[msk, 'balanced accuracy'].iloc[0])
            ba_eval_task = pd.Series(ba_eval_task).mean()
            trial.set_user_attr(f'mean BA across folds/fits for {task}', ba_eval_task)

        return objective_function_value

    except Exception as ex:
        log.error(f'Trial will be pruned because of: {ex}')
        raise optuna.TrialPruned()


study = optuna.create_study(
    sampler = optuna.samplers.GPSampler(seed = PYTORCH_SEED),
    storage = f"sqlite:///{output_path}/db.sqlite3",  # Specify the storage URL here.
    study_name = STUDY_NAME,
    load_if_exists = True,
    direction = 'maximize'
)
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
# .. store the study for future analysis (in pickle format)
with open(output_path/'study.pickle', 'wb') as f:
    pickle.dump(study, f)
# .. store the study for future analysis (as dataframe)
study.trials_dataframe().to_excel(output_path/'study.xlsx', index=False)



# refit the best model configuration to the whole training set, we do multiple fits and all are used for inference
# .. read the study
with open(output_path/'study.pickle', 'rb') as f:
    study = pickle.load(f)
study.trials_dataframe()
# .. find the optimal model configuration
best_trial = study.best_trial
model_parameters = dict()
for parameter in hyperparameters['model parameters']:
    # model_parameters[parameter.name] = getattr(trial, parameter.type)(parameter.name, parameter.lower_bound, parameter.upper_bound, log=parameter.log)
    model_parameters[parameter] = best_trial.params[parameter]
optimiser_parameters = dict()
for parameter in hyperparameters['optimiser parameters']:
    # optimiser_parameters[parameter.name] = getattr(trial, parameter.type)(parameter.name, parameter.lower_bound, parameter.upper_bound, log=parameter.log)
    optimiser_parameters[parameter] = best_trial.params[parameter]
# model_parameters = {'size_hidden_layers': 250, 'number_hidden_layers': 2, 'dropout': 0.5954408488128021}
# optimiser_parameters = {'learning_rate': 0.006730594003229105, 'weight_decay': 0.00029401811032115814, 'scheduler_decay': 0.99}
# .. create the loaders
train_loaders, eval_loaders = [], []
train_set_size_max = max(len(dsets[task]['dset']) for task in dsets)  # largest train set size among tasks
for task in dsets:
    # train set is the whole set and is also used as eval set but without dropping the last batch
    train_set = dsets[task]['dset']
    batch_size = round(BATCH_SIZE_MAX * len(train_set) / float(train_set_size_max))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=DROP_LAST_TRAINING if len(train_set) > batch_size else False)
    train_loaders.append(train_loader)
    # eval set
    eval_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_loaders.append(eval_loader)
    log.info(f'task {task}, train set: {len(train_set):4d} data points in {len(train_loader)} batches, eval set: {len(train_set):4d} data points in {len(eval_loader)} batches')
# number of classes
n_classes = [2] * len(dsets)
# if specified, scale the loss so that each class contributes according to its size or equally
# default reduction is mean
if SCALE_LOSS_CLASS_SIZE is None:
    global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]).to(device))
elif SCALE_LOSS_CLASS_SIZE == 'equal class (global)':
    global_loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([fraction_positives, 1. - fraction_positives]).to(device))
elif SCALE_LOSS_CLASS_SIZE == 'equal class (task)':
    # in this case we define a separate loss function per task in the train_eval function
    global_loss_fn = None
# reset the seed for deterministic behaviour
torch.manual_seed(PYTORCH_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(PYTORCH_SEED)
# repeat model fits with different initial seeds
for i_model_fit in range(NUMBER_MODEL_FITS):
    log.info(f'Initiating model fit {i_model_fit}')
    # set up the model
    hidden_layers = [model_parameters['size_hidden_layers']] * model_parameters['number_hidden_layers']
    net = FFNNModel(n_input=fingerprint_parameters['fpSize'], hidden_layers=hidden_layers,
                    dropout=model_parameters['dropout'], n_classes=n_classes)
    net.to(device)
    # set up the optimiser
    optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'],
                                 betas=[0.9, 0.999], eps=1e-08,
                                 weight_decay=optimiser_parameters['weight_decay'], amsgrad=False)
    # set up the scheduler
    lambda_group = lambda epoch: optimiser_parameters['scheduler_decay'] ** epoch
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])
    # train the model
    metrics_history = train_eval(net, train_loaders, eval_loaders, global_loss_fn,
                                 optimizer, scheduler, MAX_NUM_EPOCHS,
                                 weight_converge_path=None,
                                 early_stopping={'loss_eval': EARLY_STOPPING_LOSS_EVAL,
                                                 'roc_eval': EARLY_STOPPING_ROC_EVAL,
                                                 'threshold': EARLY_STOPPING_THRESHOLD},
                                 log_epoch_frequency=LOG_EPOCH_FREQUENCY,
                                 scale_loss_task_size=SCALE_LOSS_TASK_SIZE)
    # store the metrics
    metrics_history = pd.DataFrame(metrics_history)
    metrics_history.insert(0, 'fold', i_fold)
    metrics_history.insert(0, 'model fit', i_model_fit)
    for col in reversed(optimiser_parameters):
        metrics_history.insert(0, col, optimiser_parameters[col])
    print(model_parameters)
    for col in reversed(model_parameters):
        metrics_history.insert(0, col, model_parameters[col])
    # create folder to store the model fitting results
    outp = output_path / f'best_configuration_model_fit_{i_model_fit}'
    outp.mkdir(parents=True, exist_ok=True)
    # plot and save the model convergence
    task_names = list(dsets.keys())
    plot_metrics_convergence(metrics_history, task_names=task_names, stages=['train', 'eval'],
                             output=outp)
    # save the model
    torch.save(net, outp / 'model.pth')
    # save the metrics history
    metrics_history.to_excel(outp / 'metrics_history.xlsx', index=False)

