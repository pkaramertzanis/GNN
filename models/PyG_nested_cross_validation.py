import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math

from models.PyG_train import train_eval
from models.metrics import plot_metrics_convergence
# configurations
# K_FOLD_OUTER, K_FOLD_INNER
# dsets
# BATCH_SIZE_MAX
# PYTORCH_SEED
# model
# metrics_history_path

def nested_cross_validation(model: torch.nn.Module,
                            dsets: dict,
                            splits: pd.DataFrame,
                            configurations: list,
                            PYTORCH_SEED: int,
                            BATCH_SIZE_MAX: int,
                            NUM_EPOCHS: int,
                            SCALE_LOSS_CLASS_SIZE: Union[None, str],
                            SCALE_LOSS_TASK_SIZE: Union[None, str],
                            LOG_EPOCH_FREQUENCY: int,
                            device: torch.device,
                            metrics_history_path: Path) -> None:
    '''
    Nested cross-validation for a PyTorch geometric model
    :param model: PyTorch model
    :param dsets: dictionary with the datasets for each task, each dataset is a PyTorch Geometric dataset
    :param splits: dataframe with the splits for the nested cross-validation
    :param configurations: list with the model configurations
    :param PYTORCH_SEED: seed for PyTorch
    :param BATCH_SIZE_MAX: maximum batch size
    :param NUM_EPOCHS: number of epochs
    :param LOG_EPOCH_FREQUENCY: frequency to log the epoch
    :param device: device to use for the training
    :param metrics_history_path: path to store the metrics history
    '''

    # work out the number of folds from the splits
    K_FOLD_OUTER = splits['outer fold'].nunique()
    K_FOLD_INNER = splits['inner fold'].nunique()

    # compute the overall fraction of positives (for all tasks)
    y_all = []
    for task in dsets:
        y_all.extend([d.assay_data for d in dsets[task]['dset']])
    fraction_positives = sum([1 for y in y_all if y == 'positive']) / len(y_all)

    # .. delete left over metrics
    (metrics_history_path / 'metrics_history.tsv').unlink(missing_ok=True)
    # .. move dsets to device
    for task in dsets:
        if dsets[task]['dset'].x.device != device:
            dsets[task]['dset'].to(device)
    # ..initiate the outer loop of the nested cross validation
    for i_outer in range(K_FOLD_OUTER):
        log.info(f'Initiating outer iteration {i_outer}')

        if len(configurations) == 1:
            best_configuration_ID = 0
        else:
            # loop over the model configurations
            for i_configuration, configuration in enumerate(configurations, 0):
                configuration_ID = configuration['configuration_ID']
                log.info(
                    f'Trialing model/optimiser configuration {configuration_ID} ({i_configuration + 1} out of {len(configurations)})')
                model_parameters = {k: v for k, v in configuration.items() if
                                    k not in ['configuration_ID', 'learning_rate', 'weight_decay']}
                optimiser_parameters = {k: v for k, v in configuration.items() if
                                        k in ['configuration_ID', 'learning_rate', 'weight_decay']}

                # inner loop of the nested cross-validation
                metrics_history_configuration = []
                for i_inner in range(K_FOLD_INNER):
                    log.info(f'Initiating inner iteration {i_inner}')
                    # .. create the train and eval set loaders
                    train_loaders, eval_loaders = [], []
                    msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == i_inner)
                    train_set_size_max = max(
                        len(idxs) for idxs in splits.loc[msk, 'train indices'])  # largest train set size among tasks
                    eval_set_size_max = max(
                        len(idxs) for idxs in splits.loc[msk, 'eval indices'])  # largest eval set size among tasks
                    for task in dsets:
                        msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == i_inner) & (
                                    splits['task'] == task)
                        train_set = dsets[task]['dset'].index_select(splits.loc[msk, 'train indices'].iloc[0].tolist())
                        batch_size = round(BATCH_SIZE_MAX * len(train_set) / float(train_set_size_max))
                        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                  drop_last=True)  # .. we drop the last to have stable gradients
                        train_loaders.append(train_loader)
                        eval_set = dsets[task]['dset'].index_select(splits.loc[msk, 'eval indices'].iloc[0].tolist())
                        batch_size = round(BATCH_SIZE_MAX * len(eval_set) / float(eval_set_size_max))
                        eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True, drop_last=False)
                        eval_loaders.append(eval_loader)
                        log.info(
                            f'task {task}, train set: {len(train_set):4d} data points in {len(train_loader)} batches, eval set: {len(eval_set):4d} data points in {len(eval_loader)} batches')

                    torch.manual_seed(PYTORCH_SEED)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(PYTORCH_SEED)

                    # set the model
                    num_node_features = (train_loaders[0].dataset).num_node_features
                    num_edge_features = (train_loaders[0].dataset).num_edge_features
                    n_classes = [2] * len(dsets)
                    net = model(num_node_features=num_node_features, num_edge_features=num_edge_features,
                                **model_parameters,
                                n_classes=n_classes)
                    net.to(device)

                    # if specified, scale the loss so that each class contributes according to its size or equally
                    # default reduction is mean
                    if SCALE_LOSS_CLASS_SIZE is None:
                        global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
                    elif SCALE_LOSS_CLASS_SIZE == 'equal class (global)':
                        global_loss_fn = torch.nn.CrossEntropyLoss(
                            weight=torch.tensor([fraction_positives, 1. - fraction_positives]))
                    elif SCALE_LOSS_CLASS_SIZE == 'equal class (task)':
                        # in this case we define a separate loss function per task in the train_eval function
                        global_loss_fn = None

                    # optimiser
                    optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'],
                                                 betas=[0.9, 0.999], eps=1e-08,
                                                 weight_decay=optimiser_parameters['weight_decay'], amsgrad=False)

                    # scheduler
                    lambda_group = lambda epoch: 0.97 ** epoch
                    scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])

                    # train the model
                    outp = metrics_history_path / f'outer_fold_{i_outer}_configuration_ID_{configuration_ID}_inner_fold_{i_inner}'
                    outp.mkdir(parents=True, exist_ok=True)
                    metrics_history, model_summary = train_eval(net, train_loaders, eval_loaders, global_loss_fn,
                                                                optimizer, scheduler, NUM_EPOCHS,
                                                                outp / 'model_weights_diff_quantiles.tsv',
                                                                log_epoch_frequency=LOG_EPOCH_FREQUENCY,
                                                                scale_loss_task_size=SCALE_LOSS_TASK_SIZE)

                    # store the model summary
                    with open(outp / 'model_summary.txt', mode='wt', encoding='utf-8') as f:
                        for task_summary in model_summary:
                            f.write(f"task: {task_summary['task']}\n{task_summary['summary']}\n")

                    # log the metrics for the training set and evaluation set
                    metrics_history = pd.DataFrame(metrics_history)
                    # remove the roc columns so that we can store the metrics as a dataframe
                    metrics_history = metrics_history.drop(columns=['roc'], axis='columns')
                    cols = {'time': datetime.now(), 'outer fold': i_outer}
                    cols.update(configuration)
                    cols.update({'inner fold': i_inner})
                    for i_col, (col_name, col_value) in enumerate(cols.items()):
                        metrics_history.insert(i_col, col_name, col_value)
                    with open(metrics_history_path / 'metrics_history.tsv', mode='at', encoding='utf-8', buffering=1,
                              newline='') as f:
                        metrics_history.to_csv(f, header=f.tell() == 0, index=False, sep='\t', lineterminator='\n')

                    # plot the metrics
                    task_names = list(dsets.keys())
                    plot_metrics_convergence(metrics_history, task_names=task_names, stages=['train', 'eval'],
                                             output=outp)

            # find the optimal configuration by using the average eval balanced accuracy over the inner folds (avoid reading the whole file in memory)
            chunk_iterator = pd.read_csv(metrics_history_path / 'metrics_history.tsv', chunksize=10_000, sep='\t')
            metrics_history_configuration = []
            for chunk in chunk_iterator:
                # .. select the eval rows that provide the aggregate metrics across tasks and batches
                msk = (chunk['batch'].isnull()) & (chunk['task'].isnull()) & (chunk['stage'] == 'eval') & (
                            chunk['type'] == 'aggregate (epoch)')
                metrics_history_configuration.append(chunk.loc[msk])
            metrics_history_configuration = pd.concat(metrics_history_configuration, axis=0, sort=False,
                                                      ignore_index=True)
            # .. keep the last three epochs
            msk = (metrics_history_configuration['epoch'] >= metrics_history_configuration['epoch'].max() - 3)
            metrics_history_configuration = metrics_history_configuration.loc[msk]
            balanced_accuracy_eval_inner_folds = metrics_history_configuration.groupby('configuration_ID')[
                'balanced accuracy'].mean()
            best_configuration_ID = balanced_accuracy_eval_inner_folds.idxmax()
            log.info(
                f'outer fold {i_outer}, best configuration ID: {best_configuration_ID} with balanced accuracy: {balanced_accuracy_eval_inner_folds.max():.4} (range: {balanced_accuracy_eval_inner_folds.min():.4} - {balanced_accuracy_eval_inner_folds.max():.4})')

        # refit using the whole train + eval sets and evaluate in the test set
        msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == 0)
        train_eval_set_size_max = max(len(idxs_train.tolist() + idxs_eval.tolist()) for idxs_train, idxs_eval in
                                      zip(splits.loc[msk, 'train indices'], splits.loc[msk, 'eval indices']))
        test_set_size_max = max(
            len(idxs) for idxs in splits.loc[msk, 'test indices'])  # largest test set size among tasks
        train_eval_loaders, test_loaders = [], []
        for task in dsets:
            msk = (splits['outer fold'] == i_outer) & (splits['inner fold'] == 0) & (splits['task'] == task)
            train_eval_set = dsets[task]['dset'].index_select(
                splits.loc[msk, 'train indices'].iloc[0].tolist() + splits.loc[msk, 'eval indices'].iloc[0].tolist())
            batch_size = math.ceil(BATCH_SIZE_MAX * len(train_eval_set) / float(train_eval_set_size_max))
            train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size, shuffle=True,
                                           drop_last=True)  # .. we drop the last to have stable gradients
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
        configuration = [configuration for configuration in configurations if
                         configuration['configuration_ID'] == best_configuration_ID][0]
        configuration_ID = configuration['configuration_ID']
        model_parameters = {k: v for k, v in configuration.items() if
                            k not in ['configuration_ID', 'learning_rate', 'weight_decay']}
        optimiser_parameters = {k: v for k, v in configuration.items() if
                                k in ['configuration_ID', 'learning_rate', 'weight_decay']}
        num_node_features = (test_loaders[0].dataset).num_node_features
        num_edge_features = (test_loaders[0].dataset).num_edge_features
        n_classes = [2] * len(dsets)
        net = model(num_node_features=num_node_features, num_edge_features=num_edge_features,
                    **model_parameters,
                    n_classes=n_classes)
        net.to(device)

        # if specified, scale the loss so that each class contributes according to its size or equally
        # default reduction is mean
        if SCALE_LOSS_CLASS_SIZE is None:
            global_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
        elif SCALE_LOSS_CLASS_SIZE == 'equal class (global)':
            global_loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([fraction_positives, 1. - fraction_positives]))
        elif SCALE_LOSS_CLASS_SIZE == 'equal class (task)':
            # in this case we define a separate loss function per task in the train_eval function
            global_loss_fn = None

        # optimiser
        optimizer = torch.optim.Adam(net.parameters(), lr=optimiser_parameters['learning_rate'], betas=[0.9, 0.999],
                                     eps=1e-08, weight_decay=optimiser_parameters['weight_decay'],
                                     amsgrad=False)

        # scheduler
        lambda_group = lambda epoch: 0.97 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda_group])

        # train the model, double the number of epochs for the final training
        outp = metrics_history_path / f'outer_fold_{i_outer}_configuration_ID_{configuration_ID}'
        outp.mkdir(parents=True, exist_ok=True)
        metrics_history, model_summary = train_eval(net, train_eval_loaders, test_loaders, global_loss_fn, optimizer,
                                                    scheduler, 2 * NUM_EPOCHS, outp=None,
                                                    log_epoch_frequency=LOG_EPOCH_FREQUENCY,
                                                    scale_loss_task_size=SCALE_LOSS_TASK_SIZE)

        # store the model summary
        with open(outp / 'model_summary.txt', mode='wt', encoding='utf-8') as f:
            for task_summary in model_summary:
                f.write(f"task: {task_summary['task']}\n{task_summary['summary']}\n")

        # log the metrics for the training set and evaluation set
        metrics_history = pd.DataFrame(metrics_history)
        # remove the roc columns so that we can store the metrics as a dataframe and store the roc for further processing
        roc = (metrics_history
               .dropna(subset=['roc'])
               .filter(['epoch', 'stage', 'task', 'roc'], axis='columns')
               .pipe(lambda df: df.assign(**{'stage': np.where(df['stage'] == 'train', 'train+eval', 'test')}))
               )
        roc.to_pickle(outp / 'roc_outer.pkl')

        metrics_history = metrics_history.drop(columns=['roc'], axis='columns')
        cols = {'time': datetime.now(), 'outer fold': i_outer}
        cols.update(configuration)
        cols.update({'inner fold': None})
        for i_col, (col_name, col_value) in enumerate(cols.items()):
            metrics_history.insert(i_col, col_name, col_value)
        metrics_history['stage'] = np.where(metrics_history['stage'] == 'train', 'train+eval', 'test')
        with open(metrics_history_path / 'metrics_history.tsv', mode='at', encoding='utf-8', buffering=1,
                  newline='') as f:
            metrics_history.to_csv(f, header=f.tell() == 0, index=False, sep='\t', lineterminator='\n')

        # plot the metrics
        task_names = list(dsets.keys())
        plot_metrics_convergence(metrics_history, task_names=task_names, stages=['train+eval', 'test'], output=outp)

        # save the model
        torch.save(net, outp / 'model.pth')