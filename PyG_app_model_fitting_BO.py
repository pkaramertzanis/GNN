# setup logging
import logging
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

from sklearn.model_selection import StratifiedKFold

from data.combine import create_sdf

from models.MPNN_GNN.MPNN_GNN import MPNN_GNN
from models.AttentiveFP_GNN.AttentiveFP_GNN import AttentiveFP_GNN
from models.GAT_GNN.GAT_GNN import GAT_GNN

from visualisations.task_concordance import visualise_task_concordance

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from models.PyG_train import train_eval
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from models.metrics import plot_metrics_convergence

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
outp_sdf = Path(r'data/combined/sdf/genotoxicity_dataset.sdf')
outp_tab = Path(r'data/combined/tabular/genotoxicity_dataset.xlsx')
tasks = create_sdf(flat_datasets = flat_datasets,
                   task_specifications = task_specifications,
                   outp_sdf = outp_sdf,
                   outp_tab = outp_tab)
# set general parameters
PYTORCH_SEED = 1 # seed for PyTorch random number generator, it is also used for splits and shuffling to ensure reproducibility
MINIMUM_TASK_DATASET = 300 # minimum number of data points for a task
BATCH_SIZE_MAX = 1024 # maximum batch size (largest task, the smaller tasks are scaled accordingly so the number of batches is the same)
K_FOLD = 5 # number of folds for the cross-validation
MAX_NUM_EPOCHS = 500 # maximum number of epochs
MODEL_NAME = 'AttentiveFP_GNN' # name of the model, can be 'MPNN_GNN', 'AttentiveFP_GNN' or 'GAT_GNN'
SCALE_LOSS_TASK_SIZE = None # how to scale the loss function, can be 'equal task' or None
SCALE_LOSS_CLASS_SIZE = 'equal class (task)' # how to scale the loss function, can be 'equal class (task)', 'equal class (global)' or None
HANDLE_AMBIGUOUS = 'ignore' # how to handle ambiguous outcomes, can be 'keep', 'set_positive', 'set_negative' or 'ignore', but the model fitting does not support 'keep'
DROP_LAST_TRAINING = True # we can drop the last to have stable gradients and possibly NAN loss function due to lack of positives
LOG_EPOCH_FREQUENCY = 10
EARLY_STOPPING_LOSS_EVAL = 20
EARLY_STOPPING_ROC_EVAL = 10
EARLY_STOPPING_THRESHOLD = 1.e-2
NUMBER_MODEL_FITS = 1  # number of model fits in each fold


# location to store the metrics logs
output_path = Path(rf'D:\myApplications\local\2024_01_21_GCN_Muta\output\iteration122')/MODEL_NAME
output_path.mkdir(parents=True, exist_ok=True)


# visualise the task concordance (if more than one task)
if len(tasks) > 1:
    visualise_task_concordance(outp_sdf, [output_path/'task_concordance.png',
                                                output_path/'task_co-occurrence.png'])


# features, checkers and standardisers
NODE_FEATS = ['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization', 'num_rings', 'num_Hs']
EDGE_FEATS = ['bond_type', 'is_conjugated', 'num_rings'] # ['bond_type', 'is_conjugated', 'stereo_type']


# hyperparameter search space
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
    hyperparameters = {
        'model parameters': {
            'hidden_channels': IntDistribution(low=50, high=300, log=False, step=25),
            'num_layers': IntDistribution(low=2, high=7, log=False, step=1),
            'num_timesteps': IntDistribution(low=2, high=7, log=False, step=1),
            'dropout': FloatDistribution(low=0.0, high=0.8, step=None, log=False),
        },
        'optimiser parameters': {
            'learning_rate': FloatDistribution(low=1.e-5, high=1.e-2, step=None, log=True),
            'weight_decay': FloatDistribution(low=1.e-6, high=1.e-2, step=None, log=True),
            'scheduler_decay': FloatDistribution(low=0.94, high=0.98, step=None, log=False)
        }
    }
elif MODEL_NAME == 'GAT_GNN':
    model = GAT_GNN
    hyperparameters = {
        'model parameters': {
            'v2': CategoricalDistribution(choices=[True, False]),
            'n_conv': IntDistribution(low=2, high=7, log=False, step=1),
            'n_conv_hidden_per_head': IntDistribution(low=20, high=100, log=False, step=20),
            'n_heads': IntDistribution(low=2, high=6, log=False, step=1),
            'n_lin': IntDistribution(low=1, high=1, log=False, step=1),
            'n_lin_hidden': IntDistribution(low=50, high=300, log=False, step=25),
            'dropout': FloatDistribution(low=0.0, high=0.8, step=None, log=False),
        },
        'optimiser parameters': {

            'learning_rate': FloatDistribution(low=1.e-5, high=1.e-2, step=None, log=True),
            'weight_decay': FloatDistribution(low=1.e-6, high=1.e-2, step=None, log=True),
            'scheduler_decay': FloatDistribution(low=0.94, high=0.98, step=None, log=False)
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
    # move the dataset to device immediately after creation
    if dset.x.device != device:
        dset.to(device)
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


# compute the overall fraction of positives (for all tasks)
y_all = []
for task in dsets:
    y_all.extend([d.assay_data for d in dsets[task]['dset']])
fraction_positives = sum([1 for y in y_all if y == 'positive']) / len(y_all)




def objective(trial, best_value: float) -> float:
    '''
    Optuna objective function to optimise the hyperparameters by minimising the balanced accuracy
    :param trial: Optuna trial
    param best_value: best value so far
    :return: balanced accuracy for the eval set
    '''



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
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=DROP_LAST_TRAINING if len(train_set) > batch_size else False)
                train_loaders.append(train_loader)
                eval_set = dsets[task]['dset'].index_select(splits.loc[msk, 'eval indices'].iloc[0].tolist())
                batch_size = round(BATCH_SIZE_MAX * len(eval_set) / float(eval_set_size_max))
                eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=False)
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

                # if the first fold had balanced accuracy less than 3% of the maximum seen balanced accuracy do not continue with other folds
                msk = (metrics_history['type'] == 'aggregate (epoch)') & metrics_history['task'].isnull() & (metrics_history['stage'] == 'eval')
                objective_function_value_fold = metrics_history.loc[msk].groupby(['model fit', 'fold'])['balanced accuracy'].max().mean()
                if objective_function_value_fold < best_value - 0.03:
                    objective_function_value = objective_function_value_fold
                    msg = f'balanced accuracy for fold {i_fold} is less than 3% of the maximum seen balanced accuracy, pruning the trial'
                    log.info(msg)
                    raise Exception(msg)

        # compute the objective function value as the mean for all folds and model fits
        metrics_history_trial = pd.concat(metrics_history_trial, axis='index', sort=False, ignore_index=True)
        msk = (metrics_history_trial['type'] == 'aggregate (epoch)') & metrics_history_trial['task'].isnull() & (metrics_history_trial['stage'] == 'eval')
        objective_function_value = metrics_history_trial.loc[msk].groupby(['model fit', 'fold'])['balanced accuracy'].max().mean()

        # # set the metrics for each task for the evaluation set as a user attribute in the study
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
            trial.set_user_attr(f'BA {task}', ba_eval_task)

        return objective_function_value

    except Exception as ex:
        log.info(f'Trial will be pruned because of: {ex}')
        raise optuna.TrialPruned()

study = optuna.create_study(
    sampler = optuna.samplers.GPSampler(seed = PYTORCH_SEED),
    storage = "sqlite:///db.sqlite3",  # Specify the storage URL here.
    study_name = "122_AttentiveFP_TA100/98+-_MN_CA_GM",
    load_if_exists = True,
    direction = 'maximize'
)
study.optimize(lambda trial: objective(trial, study.best_value if  [trial for trial in study.get_trials() if trial.state == optuna.trial.TrialState.COMPLETE] else 0.), n_trials=180, n_jobs=1)

# store the study for future analysis
with open(output_path/'study.pickle', 'wb') as f:
    pickle.dump(study, f)
study.trials_dataframe()



