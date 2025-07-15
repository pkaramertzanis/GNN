# setup logging
import logging

from optuna.trial import TrialState

import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_model_fit.log', level_stream=logging.DEBUG, level_file=logging.DEBUG)

# import and configure pandas globally (this needs to be imported first as other modules import pandas too)
import pandas_config
import pandas as pd

from pathlib import Path
import numpy as np
import torch

from models.PyG_Dataset import PyG_Dataset

from collections import Counter
import pickle
import json

from sklearn.model_selection import StratifiedKFold

from data.combine import create_sdf

from models.MPNN_GNN.MPNN_GNN import MPNN_GNN
from models.AttentiveFP_GNN.AttentiveFP_GNN import AttentiveFP_GNN
from models.GATConv_GNN.GATConv_GNN import GATConv_GNN
from models.GCNConv_GNN.GCNConv_GNN import GCNConv_GNN

from visualisations.task_concordance import visualise_task_concordance
from visualisations.database_concordance import visualise_database_concordance

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from models.PyG_train import train_eval
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.storages import RetryFailedTrialCallback
from models.metrics import plot_metrics_convergence

# set the model architecture
MODEL_NAME = 'AttentiveFP_GNN' # name of the model, can be 'AttentiveFP_GNN', 'GCNConv_GNN' or 'GATConv_GNN'
STUDY_NAME = "make_figure" # name of the study in the Optuna sqlit database

# location to store the results
output_path = Path(rf'output/{MODEL_NAME}/{STUDY_NAME}')
# output_path = Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\ToxvalDB')
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
       {'filters': {'assay': ['bacterial reverse mutation assay'], 'cell line/species': [
                                                                                           'Escherichia coli (WP2 Uvr A)',
                                                                                           'Salmonella typhimurium (TA 102)',
                                                                                          'Salmonella typhimurium (TA 100)',
                                                                                          'Salmonella typhimurium (TA 1535)',
                                                                                          'Salmonella typhimurium (TA 98)',
                                                                                          'Salmonella typhimurium (TA 1537)'
                                                                                          ], 'metabolic activation': [
           'yes',
           'no'
       ]},
         'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation']},

    #  {'filters': {'assay': ['bacterial reverse mutation assay']},
    #   'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},
    #
    # {'filters': {'assay': ['in vitro mammalian cell micronucleus test']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},
    # # # #
    #   {'filters': {'assay': ['in vitro mammalian chromosome aberration test']},
    #    'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

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
BATCH_SIZE_MAX = 220 # maximum batch size (largest task, the smaller tasks are scaled accordingly so the number of batches is the same)
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




# visualise the task concordance (if more than one task)
if len(tasks) > 1:
    visualise_task_concordance(outp_sdf, [output_path/'task_concordance.png',
                                                                         output_path/'task_co-occurrence.png'])

# visualise the database concordance
visualise_database_concordance(outp_tab, output_path)

# features, checkers and standardisers
NODE_FEATS = ['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization', 'num_rings', 'num_Hs']
EDGE_FEATS = ['bond_type', 'is_conjugated', 'num_rings'] # ['bond_type', 'is_conjugated', 'stereo_type']

# hyperparameter search space
if MODEL_NAME == 'AttentiveFP_GNN':
    model = AttentiveFP_GNN
    hyperparameters = {
        'model parameters': {
            'hidden_channels': IntDistribution(low=50, high=300, log=False, step=25),
            'num_layers': IntDistribution(low=2, high=8, log=False, step=1),
            'num_timesteps': IntDistribution(low=2, high=8, log=False, step=1),
            'dropout': FloatDistribution(low=0.0, high=0.8, step=None, log=False),
        },
        'optimiser parameters': {
            'learning_rate': FloatDistribution(low=1.e-5, high=1.e-2, step=None, log=True),
            'weight_decay': FloatDistribution(low=1.e-7, high=1.e-2, step=None, log=True),
            'scheduler_decay': FloatDistribution(low=0.94, high=0.99, step=None, log=False)
        }
    }
elif MODEL_NAME == 'GATConv_GNN':
    model = GATConv_GNN
    hyperparameters = {
        'model parameters': {
            'n_conv': IntDistribution(low=2, high=8, log=False, step=1),
            'n_conv_hidden': IntDistribution(low=50, high=250, log=False, step=25),
            'n_heads': IntDistribution(low=2, high=5, log=False, step=1),
            'v2': CategoricalDistribution(choices=[True, False]),
            'pool': CategoricalDistribution(choices=['mean', 'add']),
            'n_lin': IntDistribution(low=0, high=1, log=False, step=1),
            'n_lin_hidden': IntDistribution(low=50, high=300, log=False, step=25),
            'dropout_lin': FloatDistribution(low=0.0, high=0.8, step=None, log=False),
            'dropout_conv': FloatDistribution(low=0.0, high=0.5, step=None, log=False),
        },
        'optimiser parameters': {
            'learning_rate': FloatDistribution(low=1.e-5, high=1.e-2, step=None, log=True),
            'weight_decay': FloatDistribution(low=1.e-7, high=1.e-2, step=None, log=True),
            'scheduler_decay': FloatDistribution(low=0.94, high=0.99, step=None, log=False)
        }
    }
elif MODEL_NAME == 'GCNConv_GNN':
    model = GCNConv_GNN
    hyperparameters = {
        'model parameters': {
            'n_conv': IntDistribution(low=2, high=8, log=False, step=1),
            'n_conv_hidden': IntDistribution(low=50, high=250, log=False, step=25),
            'pool': CategoricalDistribution(choices=['mean', 'add']),
            'n_lin': IntDistribution(low=0, high=1, log=False, step=1),
            'n_lin_hidden': IntDistribution(low=50, high=300, log=False, step=25),
            'dropout_lin': FloatDistribution(low=0.0, high=0.8, step=None, log=False),
        },
        'optimiser parameters': {
            'learning_rate': FloatDistribution(low=1.e-5, high=1.e-2, step=None, log=True),
            'weight_decay': FloatDistribution(low=1.e-7, high=1.e-2, step=None, log=True),
            'scheduler_decay': FloatDistribution(low=0.94, high=0.99, step=None, log=False)
        }
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
    # obtain the node and edge feature names and tasks that are needed for inference; given that all tasks are the same we only do this for the frist task
    if i_task == 0:
        node_feature_values, edge_attr_values = dset.get_node_edge_feature_names()
        all_features_tasks = {'node features': NODE_FEATS,
                              'edge features': EDGE_FEATS,
                              'node feature values': node_feature_values,
                              'edge feature values': edge_attr_values,
                              'tasks': tasks}
        with open(output_path / 'feature_task_info.json', 'w') as f:
            json.dump(all_features_tasks, f)
    # instead of moving the dataset to device, we move the dataloader otherwise looping over the data loader is very slow on GPUs
    # https://stackoverflow.com/questions/78477632/pytorch-geometric-dataloader-is-doing-strange-things
    # if dset.x.device != device:
    #     dset.to(device)
    # store the dataset in the dset dictionary
    entry['dset'] = dset
    if len(dset) >= MINIMUM_TASK_DATASET:
       dsets[task] = entry
    else:
        log.warning(f'task {task} has less than {MINIMUM_TASK_DATASET} data points, skipping')



# set up the cross-validation splits
cv = StratifiedKFold(n_splits=K_FOLD, random_state=PYTORCH_SEED, shuffle=True)
splits = []
for i_task, task in enumerate(dsets):
    dset = dsets[task]['dset']
    y_task = [d.assay_data for d in dset]
    y_task_dist = dict(Counter(y_task))
    per_pos_task = 100 * y_task_dist['positive'] / len(y_task)
    log.info(f"task {task}: {len(y_task):4d} data points, positives: {y_task_dist['positive']:4d} ({per_pos_task:.2f}%)")

    for i_fold, (train_indices, eval_indices) in enumerate(cv.split(range(len(dset)), y=y_task)):
        y_train = [d.assay_data for i, d in enumerate(dset) if i in train_indices]
        y_train_dist = dict(Counter(y_train))
        per_pos_train = 100 * y_train_dist['positive'] / len(y_train)
        log.info(f"task {task}, fold {i_fold}, train set: {len(y_train):4d} data points, positives: {y_train_dist['positive']:4d} ({per_pos_train:.2f}%)")

        y_eval = [d.assay_data for i, d in enumerate(dset) if i in eval_indices]
        y_eval_dist = dict(Counter(y_eval))
        per_pos_eval = 100 * y_eval_dist['positive'] / len(y_eval)
        log.info(f"task {task}, fold {i_fold}, eval set: {len(y_eval):4d} data points, positives: {y_eval_dist['positive']:4d} ({per_pos_eval:.2f}%)")
        entry = {'task ID': i_task,
                 'task': task, 'fold': i_fold,
                 'train indices': train_indices, 'eval indices': eval_indices,
                 'task # data points': len(dset),
                 'eval # data points': len(eval_indices),
                 'train # data points': len(train_indices),
                 'task %positives': per_pos_task,
                 'train %positives': per_pos_train,
                 'test %positives': per_pos_eval,
         }
        splits.append(entry)
splits = pd.DataFrame(splits)
splits.to_excel(output_path/'splits.xlsx', index=False)

# compute the overall fraction of positives (for all tasks)
y_all = []
for task in dsets:
    y_all.extend([d.assay_data for d in dsets[task]['dset']])
fraction_positives = sum([1 for y in y_all if y == 'positive']) / len(y_all)




def objective(trial) -> float:
    '''
    Optuna objective function to optimise the hyperparameters by minimising the balanced accuracy.
    For performance reasons, the function will return the balanced accuracy for the evaluation set of the first fold if it is less than the best value - 0.03. In this case
    no user attributes will be set in the trial but the trial will not be pruned because this seems to make Optuna repeat the same configuration.
    :param trial: Optuna trial
    :return: balanced accuracy for the eval set
    '''

    # check if the trial is a duplicate and return the same value
    for previous_trial in trial.study.trials:
        if previous_trial.state == optuna.trial.TrialState.COMPLETE and trial.params == previous_trial.params:
            log.warn(f"Duplicated trial: {trial.params}, return {previous_trial.value}")
            return previous_trial.value

    # fetch the best value so far, will return after the 1st fold if the balanced accuracy is less than best value - 0.03
    best_value =  trial.study.best_value if [trial for trial in  trial.study.get_trials() if trial.state == optuna.trial.TrialState.COMPLETE] else 0.

    try:
        # fetch the model and optimiser parameters
        model_parameters = dict()
        for parameter in hyperparameters['model parameters']:
            # model_parameters[parameter.name] = getattr(trial, parameter.type)(parameter.name, parameter.lower_bound, parameter.upper_bound, log=parameter.log)
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
                train_set = dsets[task]['dset'].index_select(splits.loc[msk, 'train indices'].iloc[0].tolist())
                batch_size = round(BATCH_SIZE_MAX * len(train_set) / float(train_set_size_max))
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=DROP_LAST_TRAINING if len(train_set) > batch_size else False, num_workers=0, pin_memory=True)
                train_loaders.append(train_loader)
                eval_set = dsets[task]['dset'].index_select(splits.loc[msk, 'eval indices'].iloc[0].tolist())
                batch_size = round(BATCH_SIZE_MAX * len(eval_set) / float(eval_set_size_max))
                eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)
                eval_loaders.append(eval_loader)
                log.info(f'task {task}, train set: {len(train_set):4d} data points in {len(train_loader)} batches, eval set: {len(eval_set):4d} data points in {len(eval_loader)} batches')

            # number of node and edge features
            num_node_features = (train_loaders[0].dataset).num_node_features
            num_edge_features = (train_loaders[0].dataset).num_edge_features
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
                net = model(num_node_features=num_node_features, num_edge_features=num_edge_features,
                            **model_parameters,
                            n_classes=n_classes)
                net.to(device)
                # set up the optimiser
                optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'],
                                             betas=[0.9, 0.999], eps=1e-08,
                                             weight_decay=optimiser_parameters['weight_decay'], amsgrad=False)
                # set up the scheduler
                lambda_group = lambda epoch: optimiser_parameters['scheduler_decay'] ** epoch
                scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])

                # train the model
                metrics_history, model_summary = train_eval(net, train_loaders, eval_loaders, global_loss_fn,
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

                # if the first fold had balanced accuracy less than 3% of the maximum seen balanced accuracy do not continue with other folds, return the objective function value
                msk = (metrics_history['type'] == 'aggregate (epoch)') & metrics_history['task'].isnull() & (metrics_history['stage'] == 'eval')
                objective_function_value_fold = metrics_history.loc[msk].groupby(['model fit', 'fold'])['balanced accuracy'].max().mean()
                if objective_function_value_fold < best_value - 0.100000000:
                    objective_function_value = objective_function_value_fold
                    msg = f'balanced accuracy for fold {i_fold} is less than 3% of the maximum seen balanced accuracy, pruning the trial'
                    log.info(msg)
                    return objective_function_value

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
        log.info(f'Trial will be pruned because of: {ex}')
        raise optuna.TrialPruned()

storage = optuna.storages.RDBStorage(
    url=f"sqlite:///{output_path}/db.sqlite3",
    heartbeat_interval=60,
    grace_period=120,
    failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
)


study = optuna.create_study(
    sampler = optuna.samplers.GPSampler(seed = PYTORCH_SEED),
    storage = storage,  # Specify the storage URL here.
    study_name = STUDY_NAME,
    load_if_exists = True,
    direction = 'maximize'
)
study.optimize(lambda trial: objective(trial), n_trials=N_TRIALS, n_jobs=1)

# store the study for future analysis (as pickle)
with open(output_path/'study.pickle', 'wb') as f:
    pickle.dump(study, f)
# store the study for future analysis (as dataframe)
study.trials_dataframe().to_excel(output_path / 'study.xlsx', index=False)



# refit the best model configuration to the whole training set, we do multiple fits and all are used for inference
# .. find the optimal model configuration
best_trial = study.best_trial
# .. find the optimal epoch for each model fit
model_fits = list(output_path.glob(f'trial_{best_trial.number}_fold_*_model_fit_*'))
optimal_epochs = []
for model_fit in model_fits:
    metrics = pd.read_excel(model_fit / 'metrics_history.xlsx')
    msk = (metrics['type'] == 'aggregate (epoch)') & metrics['task'].isnull() & (metrics['stage'] == 'eval')
    idx = metrics.loc[msk, 'balanced accuracy'].idxmax()
    optimal_epoch = metrics.loc[idx, 'epoch']
    optimal_epochs.append(optimal_epoch)
optimal_epoch = max(optimal_epochs)
log.info(f'final model fitting will be terminated at epoch {optimal_epoch}')
MAX_NUM_EPOCHS = optimal_epoch
model_parameters = dict()
for parameter in hyperparameters['model parameters']:
    # model_parameters[parameter.name] = getattr(trial, parameter.type)(parameter.name, parameter.lower_bound, parameter.upper_bound, log=parameter.log)
    model_parameters[parameter] = best_trial.params[parameter]
optimiser_parameters = dict()
for parameter in hyperparameters['optimiser parameters']:
    # optimiser_parameters[parameter.name] = getattr(trial, parameter.type)(parameter.name, parameter.lower_bound, parameter.upper_bound, log=parameter.log)
    optimiser_parameters[parameter] = best_trial.params[parameter]
# .. create the loaders
train_loaders, eval_loaders = [], []
train_set_size_max = max(len(dsets[task]['dset']) for task in dsets)  # largest train set size among tasks
for task in dsets:
    # train set is the whole set and is also used as eval set but without dropping the last batch
    train_set = dsets[task]['dset']
    batch_size = round(BATCH_SIZE_MAX * len(train_set) / float(train_set_size_max))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=DROP_LAST_TRAINING if len(train_set) > batch_size else False, num_workers=0, pin_memory=True)
    train_loaders.append(train_loader)
    # eval set
    eval_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)
    eval_loaders.append(eval_loader)
    log.info(f'task {task}, train set: {len(train_set):4d} data points in {len(train_loader)} batches, eval set: {len(train_set):4d} data points in {len(eval_loader)} batches')
# .. set the number of node and edge features
num_node_features = (train_loaders[0].dataset).num_node_features
num_edge_features = (train_loaders[0].dataset).num_edge_features
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
    net = model(num_node_features=num_node_features, num_edge_features=num_edge_features,
                        **model_parameters,
                        n_classes=n_classes)
    net.to(device)
    # set up the optimiser
    optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'],
                                         betas=[0.9, 0.999], eps=1e-08,
                                         weight_decay=optimiser_parameters['weight_decay'], amsgrad=False)
    # set up the scheduler
    lambda_group = lambda epoch: optimiser_parameters['scheduler_decay'] ** epoch
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])
    # train the model
    metrics_history, model_summary = train_eval(net, train_loaders, eval_loaders, global_loss_fn,
                                                        optimizer, scheduler, MAX_NUM_EPOCHS,
                                                        weight_converge_path=None,
                                                        early_stopping={'loss_eval': EARLY_STOPPING_LOSS_EVAL,
                                                                        'roc_eval': EARLY_STOPPING_ROC_EVAL,
                                                                        'threshold': EARLY_STOPPING_THRESHOLD},
                                                        log_epoch_frequency=LOG_EPOCH_FREQUENCY,
                                                        scale_loss_task_size=SCALE_LOSS_TASK_SIZE)
    # store the metrics
    metrics_history = pd.DataFrame(metrics_history)
    metrics_history.insert(0, 'fold', None)
    metrics_history.insert(0, 'model fit', i_model_fit)
    for col in reversed(optimiser_parameters):
        metrics_history.insert(0, col, optimiser_parameters[col])
    for col in reversed(model_parameters):
        metrics_history.insert(0, col, model_parameters[col])
    # create folder to store the model fitting results
    outp = output_path / f'best_configuration_model_fit_early_stopping_{i_model_fit}'
    outp.mkdir(parents=True, exist_ok=True)
    # plot and save the model convergence
    task_names = list(dsets.keys())
    plot_metrics_convergence(metrics_history, task_names=task_names, stages=['train', 'eval'],
                             output=outp)
    # save the model
    torch.save(net, outp / 'model.pth')
    # save the metrics history
    metrics_history.to_excel(outp / 'metrics_history.xlsx', index=False)


