# setup logging
import glob
import random

import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_muta_model.log')

from pathlib import Path
import pandas as pd
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
import random
from datetime import datetime

from sklearn.model_selection import StratifiedKFold

from models.DMPNN_GCN.DMPNN_GCN import DMPNN_GCN
from models.metrics import plot_metrics
from models.train import train_eval

# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr', False)
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


PYTORCH_SEED = 1 # seed for PyTorch random number generator
MINIMUM_TASK_DATASET = 256 # minimum number of data points for a task
BATCH_SIZE_MAX = 256 # maximum batch size (largest task, the smaller tasks are scaled accordingly so the number of batches is the same)
K_FOLD_INNER = 5 # number of folds for the inner cross-validation
K_FOLD_OUTER = 10 # number of folds for the outer cross-validation
NUM_EPOCHS = 3 # number of epochs

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# build the PyG datasets (no split at this stage)
target_level = 'genotoxicity_assay_level'
tasks = ['in_vitro_chrom_ab_cho',
          'hgprt_mut',
          'mouse_lymphoma_act',
          'mouse_lymphoma_unact',
          'bacterial_mutation',
          'salmonella_mut',
          'in_vivo_micronuc_mouse',
          'in_vitro_sce_cho',
          'in_vitro_sce_comp',
          'in_vivo_rodent_dl_mut',
          'in_vivo_rodent_mut',
          'in_vivo_chrom_ab_comp',
          'in_vivo_chrom_ab_other',
          'e_coli_sal_102_a_t_mut',
          'in_vitro_chrom_ab_chl',
          'in_vivo_chrom_ab_rat',
          'in_vitro_sce_other']
dsets = {}
for task in tasks:
    log.info(f'preparing dataset for task: {task}')
    entry = {}
    target_assay_endpoint = task
    dset = PyG_Dataset(root=Path(rf'D:\myApplications\local\2024_01_21_GCN_Muta\datasets\Leadscope\processed\PyTorch_Geometric_{target_assay_endpoint.replace(" ", "_")}'),
                       target_level=target_level, target_assay_endpoint=target_assay_endpoint, ambiguous_outcomes='ignore',
                       force_reload=False,
                       node_feats=['atom_symbol', 'atom_charge', 'atom_degree', 'atom_hybridization'],
                       edge_feats=['bond_type', 'is_conjugated'],
                       checker_ops = {'allowed_atoms': ['C', 'O', 'N', 'Cl', 'S', 'F', 'Br', 'P', 'B', 'Si', 'I', 'H']},
                       standardiser_ops = ['cleanup', 'addHs']
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
        per_pos_test =100 * y_outer_test_dist['positive'] / len(y_outer_test)
        log.info(f"task {task}, outer fold {i_outer}, test set: {len(y_outer_test):4d} data points, positives: {y_outer_test_dist['positive']:4d} ({per_pos_test:.2f}%)")

        y_inner = [d.assay_data for i, d in enumerate(dset) if i in tmp_indices]
        for i_inner, (train_indices, evaluate_indices) in enumerate(cv_inner.split(range(len(tmp_indices)), y=y_inner)):
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
                     'train %positives': per_pos_inner_train}
            splits.append(entry)
splits = pd.DataFrame(splits)
for task in dsets:
    tmp = splits.loc[splits['task']==task, ['task %positives', 'test %positives', 'train %positives']].describe().loc[['min', 'max']].to_markdown()
    log.info(f'task {task}, % positives in different splits\n{tmp}')


# set up the model configurations
configuration_ID = 0
configurations = []
for n_conv_ in [5, 6]: # [1, 2, 3, 4, 5, 6]:
    for n_lin_ in [2]: #[1, 2, 3, 4]:
        for n_conv_hidden_ in [64]: #[32, 64, 128, 256]:
            for n_edge_NN_ in [65]: #[32, 64, 128, 256]:
                for n_lin_hidden_ in [64]: #[32, 64, 128, 256, 512]:
                    for dropout_ in [0.5]: #[0.5, 0.6, 0.7, 0.8]:
                        configurations.append({'configuration_ID': configuration_ID, 'n_conv': n_conv_, 'n_lin': n_lin_, 'n_conv_hidden': n_conv_hidden_, 'n_edge_NN': n_edge_NN_, 'n_lin_hidden': n_lin_hidden_, 'dropout': dropout_})
                        configuration_ID += 1
log.info(f'number of configurations: {len(configurations)}')
random.seed(PYTORCH_SEED)
random.shuffle(configurations)

# location to store the metrics logs
metrics_history_path = Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output')

# outer loop of the nested cross-validation
for i_outer in range(K_FOLD_OUTER):

    # loop over the model configurations
    for i_configuration, configuration in enumerate(configurations, 0):
        configuration_ID = configuration['configuration_ID']
        log.info(f'Trialing model configuration {configuration_ID} ({i_configuration} out of {len(configurations)})')
        n_conv_ = configuration['n_conv']
        n_lin_ = configuration['n_lin']
        n_conv_hidden_ = configuration['n_conv_hidden']
        n_edge_NN_ = configuration['n_edge_NN']
        n_lin_hidden_ = configuration['n_lin_hidden']
        dropout_ = configuration['dropout']

        # inner loop of the nested cross-validation
        metrics_history_configuration = []
        for i_inner in range(K_FOLD_INNER):
            # .. create the train and eval set loaders
            train_loaders, eval_loaders = [], []
            msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == i_inner)
            train_set_size_max = max(len(idxs) for idxs in splits.loc[msk, 'train indices'])  # largest train set size among tasks
            eval_set_size_max = max(len(idxs) for idxs in splits.loc[msk, 'eval indices'])  # largest eval set size among tasks
            for task in dsets:
                msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == i_inner) & (splits['task'] == task)
                train_set = dsets[task]['dset'].index_select(splits.loc[msk, 'train indices'].iloc[0].tolist())
                batch_size = int(BATCH_SIZE_MAX * len(train_set) / float(train_set_size_max))
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True) # .. we drop the last to have stable gradients
                train_loaders.append(train_loader)
                eval_set = dsets[task]['dset'].index_select(splits.loc[msk, 'eval indices'].iloc[0].tolist())
                batch_size = int(BATCH_SIZE_MAX * len(eval_set) / float(eval_set_size_max))
                eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=False)
                eval_loaders.append(eval_loader)

            # set up the model
            num_node_features = (train_loaders[0].dataset).num_node_features
            num_edge_features = (train_loaders[0].dataset).num_edge_features
            n_conv = n_conv_  # n_conv = 3
            n_edge_NN = n_edge_NN_  # n_edge_NN = 32
            n_conv_hidden = 48  #
            n_lin = n_lin_  # n_lin = 2
            n_lin_hidden = n_lin_hidden_  # n_lin_hidden = 256
            dropout = dropout_  # dropout = 0.5
            activation_function = torch.nn.functional.leaky_relu
            n_classes = [2] * len(dsets)
            # set the seed for reproducibility
            torch.manual_seed(PYTORCH_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(PYTORCH_SEED)
            # set the model
            net = DMPNN_GCN(num_node_features, num_edge_features,
                            n_conv, n_edge_NN, n_conv_hidden,
                            n_lin, n_lin_hidden,
                            dropout, activation_function, n_classes)
            net.to(device)
            # optimiser
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=[0.9, 0.999], eps=1e-08, weight_decay=0, amsgrad=False)
            # loss function, default reduction is mean
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
            # scheduler
            lambda_group = lambda epoch: 0.96 ** epoch
            scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])

            # train the model
            num_epochs = 5
            metrics_history = train_eval(net, train_loaders, eval_loaders, optimizer, loss_fn, scheduler, num_epochs, log_epoch_frequency=10)
            metrics_history = pd.DataFrame(metrics_history)

            # plot the metrics
            outp = metrics_history_path/f'outer_fold_{i_outer}_configuration_ID_{configuration_ID}_inner_fold_{i_inner}'
            outp.mkdir(parents=True, exist_ok=True)
            plot_metrics(metrics_history, outp)

            # log the metrics for the training set and evaluation set
            cols = {'time': datetime.now(),
                    'outer fold': i_outer,
                    'configuration ID': configuration_ID,
                    'n_conv' : n_conv_,
                    'n_lin':  n_lin_,
                    'n_conv_hidden': n_conv_hidden_,
                    'n_edge_NN': n_edge_NN_,
                    'n_lin_hidden': n_lin_hidden_,
                    'dropout': dropout_,
                    'inner fold': i_inner}
            for i_col, (col_name, col_value) in enumerate(cols.items()):
                metrics_history.insert(i_col, col_name, col_value)

            # append the results to the metric history log
            with open(metrics_history_path/'metrics_history.tsv', mode='at', encoding='utf-8', buffering=1, newline='') as f:
                metrics_history.to_csv(f, header=f.tell()==0, index=False, sep='\t', lineterminator='\n')

    # assert 1==0

    # find the optimal configuration by using the average eval f1 score over the inner folds (avoid reading the whole file in memory)
    chunk_iterator = pd.read_csv(metrics_history_path, chunksize=10_000, sep='\t')
    metrics_history_configuration = []
    for chunk in chunk_iterator:
        # .. select the eval rows that provide the aggregate metrics across tasks and batches
        msk = (chunk['batch'].isnull()) & (chunk['task'].isnull()) & (chunk['stage'] == 'eval') & (chunk['type'] == 'aggregate (epoch)')
        metrics_history_configuration.append(chunk.loc[msk])
    metrics_history_configuration = pd.concat(metrics_history_configuration, axis=0, sort=False, ignore_index=True)
    # .. keep the last three epochs
    msk = (metrics_history_configuration['epoch'] >= metrics_history_configuration['epoch'].max() - 3)
    metrics_history_configuration = metrics_history_configuration.loc[msk]
    f1_eval_inner_folds = metrics_history_configuration.groupby('configuration ID')['f1 score'].mean()
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
        batch_size = int(BATCH_SIZE_MAX * len(train_eval_set) / float(train_eval_set_size_max))
        train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size, shuffle=True, drop_last=False)
        train_eval_loaders.append(train_eval_loader)
        test_set = dsets[task]['dset'].index_select(splits.loc[msk, 'test indices'].iloc[0].tolist())
        batch_size = int(BATCH_SIZE_MAX * len(test_set) / float(test_set_size_max))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loaders.append(test_loader)
    # .. train the model
    # .. set the seed for reproducibility
    torch.manual_seed(PYTORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PYTORCH_SEED)
    # set the model
    configuration = [configuration for configuration in configurations if configuration['configuration_ID'] == best_configuration_ID][0]
    n_conv_ = configuration['n_conv']
    n_lin_ = configuration['n_lin']
    n_conv_hidden_ = configuration['n_conv_hidden']
    n_edge_NN_ = configuration['n_edge_NN']
    n_lin_hidden_ = configuration['n_lin_hidden']
    dropout_ = configuration['dropout']
    net = DMPNN_GCN(num_node_features, num_edge_features,
                    n_conv, n_edge_NN, n_conv_hidden,
                    n_lin, n_lin_hidden,
                    dropout, activation_function, n_classes)
    net.to(device)
    # optimiser
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=[0.9, 0.999], eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    # loss function, default reduction is mean
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
    # scheduler
    lambda_group = lambda epoch: 0.96 ** epoch
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])
    # train the model
    num_epochs = 5
    metrics_history = train_eval(net, train_eval_loaders, test_loaders, optimizer, loss_fn, scheduler, num_epochs, log_epoch_frequency=10)
    metrics_history = pd.DataFrame(metrics_history)

    # plot the metrics
    outp = metrics_history_path/f'outer_fold_{i_outer}_configuration_ID_{configuration_ID}'
    outp.mkdir(parents=True, exist_ok=True)
    plot_metrics(metrics_history, outp)

    # log the metrics for the training set and evaluation set
    cols = {'time': datetime.now(),
            'outer fold': i_outer,
            'configuration ID': configuration_ID,
            'n_conv': n_conv_,
            'n_lin': n_lin_,
            'n_conv_hidden': n_conv_hidden_,
            'n_edge_NN': n_edge_NN_,
            'n_lin_hidden': n_lin_hidden_,
            'dropout': dropout_,
            'inner fold': None}
    metrics_history['stage'] = np.where(metrics_history['stage']=='train', 'train+eval', 'test')
    for i_col, (col_name, col_value) in enumerate(cols.items()):
        metrics_history.insert(i_col, col_name, col_value)

    # append the results to the metric history log
    with open(metrics_history_path/'metrics_history.tsv', mode='at', encoding='utf-8', buffering=1, newline='') as f:
        metrics_history.to_csv(f, header=f.tell() == 0, index=False, sep='\t', lineterminator='\n')


# ------------------------------------------------------------








converged_metrics_history = []
for fpath in glob.glob(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\*.xlsx'):
    log.info(f'processing {fpath}')
    metrics_history = pd.read_excel(fpath)
    msk = (metrics_history['epoch'] == metrics_history['epoch'].max()) & metrics_history['batch'].isnull() & metrics_history['task'].isnull() & (metrics_history['type']=='aggregate (epoch)')
    converged_metrics_history.append(metrics_history[msk])
converged_metrics_history = pd.concat(converged_metrics_history, axis='index', ignore_index=True, sort=False)
# sort the configurations according to the test f1 score (ascending)
# converged_metrics_history = converged_metrics_history.sort_values(by='configuration ID', ascending=True)
converged_metrics_history_test = converged_metrics_history.loc[converged_metrics_history['stage'] == 'test']
ordered_configuration_IDs = converged_metrics_history_test[['configuration ID', 'f1 score']].sort_values(by='f1 score', ascending=True)['configuration ID'].to_list()
converged_metrics_history = pd.concat([converged_metrics_history.loc[converged_metrics_history['stage']=='train'].set_index('configuration ID').reindex(ordered_configuration_IDs).reset_index(),
                                       converged_metrics_history.loc[converged_metrics_history['stage']=='test'].set_index('configuration ID').reindex(ordered_configuration_IDs).reset_index()],
                                      axis='index', sort=False, ignore_index=True)
cols = ['n_conv', 'n_lin', 'n_conv_hidden', 'n_edge_NN', 'n_lin_hidden', 'dropout', 'f1 score', 'stage']
fig = plt.figure(figsize=(12, 6))
axs = fig.subplots(len(cols)-1, 1, sharex=True)
fig.subplots_adjust(hspace=0.4)  # Increase horizontal and vertical space
msk_train = converged_metrics_history['stage'] == 'train'
idx = list(range(msk_train.sum()))
axs[0].scatter(idx, converged_metrics_history.loc[msk_train, 'f1 score'], marker='o', alpha=0.5, edgecolors='k', facecolor='k', label='train')
axs[0].scatter(idx, converged_metrics_history.loc[~msk_train, 'f1 score'], marker='o', alpha=0.5, edgecolors='r', facecolor='r', label='test')
axs[0].hlines(0.8, 0, msk_train.sum(), colors='k', linestyles='dotted', label='80%', linewidth=0.5)
axs[0].hlines(0.85, 0, msk_train.sum(), colors='k', linestyles='dotted', label='85%', linewidth=0.5)
axs[0].hlines(0.90, 0, msk_train.sum(), colors='k', linestyles='dotted', label='90%', linewidth=0.5)
axs[0].legend(loc='upper left', fontsize=6, frameon=False, ncols=3)
f1_scores_train = converged_metrics_history.loc[msk_train,  'f1 score'].to_list()
f1_scores_test = converged_metrics_history.loc[~msk_train,  'f1 score'].to_list()
# for i in range(len(f1_scores_test)):
#     value = max(f1_scores_train[i], f1_scores_test[i])
#     axs[0].annotate(f'{f1_scores_test[i]: 0.3f}', xy=(i, value), xycoords='data', textcoords="data", xytext=(i, value+0.05), va='center', ha='center', fontsize=6, rotation=90)
axs[0].set_ylim(0.65, 1)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_title('f1 score', fontsize=6)
msk_train = converged_metrics_history['stage'] == 'train'
for i_col, col in enumerate(['n_conv', 'n_lin', 'n_conv_hidden', 'n_edge_NN', 'n_lin_hidden', 'dropout'], 1):
    axs[i_col].scatter(idx, converged_metrics_history.loc[msk_train, col], marker='o', alpha=1, edgecolors='k', facecolor='k')
    axs[i_col].set_title(col, fontsize=6)
    axs[i_col].spines['top'].set_visible(False)
    axs[i_col].spines['right'].set_visible(False)
plt.show()
fig.savefig(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\configuration_trials.png', dpi=600)