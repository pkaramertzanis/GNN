import logger
log = logger.get_logger(__name__)

import pandas as pd
from pathlib import Path
import math

import torch
from torch_geometric.data import Data, DataLoader

import matplotlib.pyplot as plt

from models.DMPNN_GCN import DMPNN_GCN
from models.metrics import compute_metrics

from utilities import zip_recycle



def compute_metrics(tp, tn, fp, fn) -> dict:
    """
    Compute metrics (accuracy, precision, recall, f1 score) from true positive, true negative, false positive, and false negative counts
    :param tp: true positive
    :param tn: true negative
    :param fp: false positive
    :param fn: false negative
    :return: dictionary with accuracy, precision, recall, and f1 score in addition to the input tp, tn, fp and fn
    """
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    metrics = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
               'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1 score': f1_score}
    return metrics

def train_eval(net: DMPNN_GCN,
               train_loaders: list[DataLoader],
               eval_loaders: list[DataLoader],
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.modules.loss._Loss,
               scheduler: torch.optim.lr_scheduler.LRScheduler,
               num_epochs: int,
               log_epoch_frequency: int = 10,
               metrics_history = None):
    """
    Train the GNN model using the eval set and evaluate its performance on the eval set. This function essentially
    executes a single inner loop of the nested cross-validation procedure.
    :param net: PyTorch Geometric model
    :param train_loaders: PyTorch Geometric DataLoaders with the training data for the different tasks
    :param eval_loaders: PyTorch Geometric DataLoaders with the eval data for the different tasks
    :param optimizer: PyTorch optimizer
    :param loss_fn: PyTorch loss function
    :param scheduler: PyTorch learning rate scheduler
    :param num_epochs: number of epochs
    :param log_epoch_frequency: log the metrics every so many epochs
    :param metrics_history: list with metrics history
    :return:
    """
    # metrics history, continue from the previous run if not None
    if metrics_history is None:
        metrics_history = []

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train
    for i_epoch in range(num_epochs):
        metrics_epoch = []
        net.train()
        # loop over the batches, the train loader for each task may have slightly different number of batches, in which case we recycle the shorter train loaders
        for i_batch, batches in enumerate(zip_recycle(*train_loaders)):
            metrics_batch = []
            optimizer.zero_grad()
            loss = torch.tensor(0.)
            n_datapoints = 0
            for i_task, task_batch in enumerate(batches):
                metrics_batch_task = {'epoch': i_epoch, 'batch': i_batch, 'task': i_task, 'stage': 'train', 'type': 'raw', 'number of datapoints': len(task_batch)}
                n_datapoints += len(task_batch)

                y = [1 if assay_data=='positive' else 0 for assay_data in task_batch.assay_data]
                y = torch.tensor(y, dtype=torch.long).to(device)

                # pred = net(batch)
                pred = net(task_batch.x, task_batch.edge_index, task_batch.edge_attr, task_batch.batch, task_id=i_task)

                # true positive, true negative, false positive, and false negative counts for each task
                tp = ((torch.argmax(pred, dim=1) == y) & (y == 1)).int().sum()
                tn = ((torch.argmax(pred, dim=1) == y) & (y == 0)).int().sum()
                fp = ((torch.argmax(pred, dim=1) != y) & (y == 0)).int().sum()
                fn = ((torch.argmax(pred, dim=1) != y) & (y == 1)).int().sum()

                loss_task = loss_fn(pred, y)
                loss += loss_task * len(task_batch)

                metrics_batch_task['loss (mean)'] = loss_task.item()
                metrics_batch_task.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
                metrics_batch.append(metrics_batch_task)

            # report the metrics for the batch
            tmp = pd.DataFrame(metrics_batch)
            loss_mean = (tmp['number of datapoints']*tmp['loss (mean)']).sum()/tmp['number of datapoints'].sum()
            tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), tmp['number of datapoints'].sum()
            tmp = {'epoch': i_epoch, 'batch': i_batch, 'task': None, 'stage': 'train', 'type': 'aggregate (batch)', 'number of datapoints': number_of_datapoints}
            tmp['loss (mean)'] = loss_mean
            tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
            metrics_batch.append(tmp)

            # log the metrics
            log.info(pd.DataFrame(metrics_batch))

            metrics_epoch.extend(metrics_batch)
            # average the loss for all tasks in the batch and back propagate
            loss = loss/float(n_datapoints)
            loss.backward()
            optimizer.step()

        if (i_epoch % log_epoch_frequency == 0) or (i_epoch >= num_epochs - 3):  # the last 3 epochs are always reported in case we wish to average the metrics

            # report the metrics for the epoch (for all tasks)
            tmp = pd.DataFrame(metrics_epoch)
            msk = (tmp['stage'] == 'train') & (tmp['type'] == 'raw')
            tmp = tmp.loc[msk]
            loss_mean = (tmp['number of datapoints']*tmp['loss (mean)']).sum()/tmp['number of datapoints'].sum()
            tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), tmp['number of datapoints'].sum()
            tmp = {'epoch': i_epoch, 'batch': None, 'task': None, 'stage': 'train', 'type': 'aggregate (epoch)', 'number of datapoints': number_of_datapoints}
            tmp['loss (mean)'] = loss_mean
            tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
            metrics_epoch.append(tmp)

            # report the metrics for the epoch (for each task)
            for i_task in range(len(train_loaders)):
                tmp = pd.DataFrame(metrics_epoch)
                msk = (tmp['stage'] == 'train') & (tmp['type'] == 'raw') & (tmp['task'] == i_task)
                tmp = tmp.loc[msk]
                loss_mean = (tmp['number of datapoints']*tmp['loss (mean)']).sum()/tmp['number of datapoints'].sum()
                tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), tmp['number of datapoints'].sum()
                tmp = {'epoch': i_epoch, 'batch': None, 'task': i_task, 'stage': 'train', 'type': 'aggregate (epoch)', 'number of datapoints': number_of_datapoints}
                tmp['loss (mean)'] = loss_mean
                tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
                metrics_epoch.append(tmp)

            scheduler.step()

            # evaluate the model on the eval set
            net.eval()
            for i_batch, batches in enumerate(zip_recycle(*eval_loaders)):
                metrics_batch = []
                loss = torch.tensor(0.)
                n_datapoints = 0
                for i_task, task_batch in enumerate(batches):
                    metrics_batch_task = {'epoch': i_epoch, 'batch': i_batch, 'task': i_task, 'stage': 'eval', 'type': 'raw', 'number of datapoints': len(task_batch)}
                    n_datapoints += len(task_batch)

                    y = [1 if assay_data == 'positive' else 0 for assay_data in task_batch.assay_data]
                    y = torch.tensor(y, dtype=torch.long).to(device)

                    # pred = net(batch)
                    pred = net(task_batch.x, task_batch.edge_index, task_batch.edge_attr, task_batch.batch, task_id=i_task)

                    # true positive, true negative, false positive, and false negative counts for each task
                    tp = ((torch.argmax(pred, dim=1) == y) & (y == 1)).int().sum()
                    tn = ((torch.argmax(pred, dim=1) == y) & (y == 0)).int().sum()
                    fp = ((torch.argmax(pred, dim=1) != y) & (y == 0)).int().sum()
                    fn = ((torch.argmax(pred, dim=1) != y) & (y == 1)).int().sum()

                    loss_task = loss_fn(pred, y)
                    loss += loss_task * len(task_batch)

                    metrics_batch_task['loss (mean)'] = loss_task.item()
                    metrics_batch_task.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
                    metrics_batch.append(metrics_batch_task)

                # report the metrics for the batch
                tmp = pd.DataFrame(metrics_batch)
                loss_mean = (tmp['number of datapoints'] * tmp['loss (mean)']).sum() / tmp['number of datapoints'].sum()
                tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), \
                tmp['number of datapoints'].sum()
                tmp = {'epoch': i_epoch, 'batch': i_batch, 'task': None, 'stage': 'eval', 'type': 'aggregate (batch)', 'number of datapoints': number_of_datapoints}
                tmp['loss (mean)'] = loss_mean
                tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
                metrics_batch.append(tmp)

                # log the metrics
                log.info(pd.DataFrame(metrics_batch))

                metrics_epoch.extend(metrics_batch)

            # report the metrics for the epoch (for all tasks)
            tmp = pd.DataFrame(metrics_epoch)
            msk = (tmp['stage'] == 'eval') & (tmp['type'] == 'raw')
            tmp = tmp.loc[msk]
            loss_mean = (tmp['number of datapoints']*tmp['loss (mean)']).sum()/tmp['number of datapoints'].sum()
            tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), tmp['number of datapoints'].sum()
            tmp = {'epoch': i_epoch, 'batch': None, 'task': None, 'stage': 'eval', 'type': 'aggregate (epoch)', 'number of datapoints': number_of_datapoints}
            tmp['loss (mean)'] = loss_mean
            tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
            metrics_epoch.append(tmp)

            # report the metrics for the epoch (for each task)
            for i_task in range(len(train_loaders)):
                tmp = pd.DataFrame(metrics_epoch)
                msk = (tmp['stage'] == 'eval') & (tmp['type'] == 'raw') & (tmp['task'] == i_task)
                tmp = tmp.loc[msk]
                loss_mean = (tmp['number of datapoints']*tmp['loss (mean)']).sum()/tmp['number of datapoints'].sum()
                tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), tmp['number of datapoints'].sum()
                tmp = {'epoch': i_epoch, 'batch': None, 'task': i_task, 'stage': 'eval', 'type': 'aggregate (epoch)', 'number of datapoints': number_of_datapoints}
                tmp['loss (mean)'] = loss_mean
                tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
                metrics_epoch.append(tmp)

            metrics_history.extend(metrics_epoch)

    return metrics_history



def plot_metrics(metrics_history: dict, output: Path):
    """
    Plot the metrics history
    :param metrics_history: list with metrics history dictionaries for training and validation for each epoch
    :param output: output file path
    :return:
    """
    plt.interactive(False)

    df = pd.DataFrame(metrics_history)

    # overall loss for train and eval set
    msk = df['batch'].isnull() & df['task'].isnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == 'train')
    loss_train = df.loc[msk, ['epoch', 'loss (mean)']]
    msk = df['batch'].isnull() & df['task'].isnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == 'eval')
    loss_eval = df.loc[msk, ['epoch', 'loss (mean)']]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.subplots()
    ax.plot(loss_train['epoch'], loss_train['loss (mean)'], label='train', c='k', linewidth=0.5)
    ax.plot(loss_eval['epoch'], loss_eval['loss (mean)'], label='eval', c='k', linestyle='dashed', linewidth=0.5)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'loss_overall.png', dpi=600)

    # loss for train and eval set for each task
    fig = plt.figure(figsize=(10, 6))
    n_tasks = df['task'].dropna().nunique()
    axs = fig.subplots(math.ceil(n_tasks/3), 3, sharex=True, sharey=True)
    for i_task, ax in enumerate(axs.flatten()):
        msk = df['batch'].isnull() & df['task'].notnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == 'train') & (df['task'] == i_task)
        loss_train = df.loc[msk, ['epoch', 'task', 'loss (mean)']]
        msk = df['batch'].isnull() & df['task'].notnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == 'eval') & (df['task'] == i_task)
        loss_eval = df.loc[msk, ['epoch', 'task', 'loss (mean)']]
        ax.plot(loss_train['epoch'], loss_train['loss (mean)'], label='train', c='k', linewidth=0.5)
        ax.plot(loss_eval['epoch'], loss_eval['loss (mean)'], label='eval', c='k', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel('epoch', fontsize=6)
        ax.set_ylabel('loss', fontsize=6)
        ax.set_title(f'task: {i_task}', fontsize=6)
        ax.legend(fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
    fig.tight_layout()
    fig.savefig(output / 'loss_task.png', dpi=600)

    # overall accuracy, precision, recall and F1-score
    msk = df['batch'].isnull() & df['task'].isnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == 'train')
    metrics_train = df.loc[msk, ['epoch', 'accuracy', 'precision', 'recall', 'f1 score']]
    msk = df['batch'].isnull() & df['task'].isnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == 'eval')
    metrics_eval = df.loc[msk, ['epoch', 'accuracy', 'precision', 'recall', 'f1 score']]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.subplots()
    ax.hlines(0.8, 0, len(metrics_train), colors='k', linestyles='dashed', label='80%', linewidth=0.5)
    ax.plot(metrics_train['epoch'], metrics_train['accuracy'], label='accuracy (train)', c='b', linewidth=0.5)
    ax.plot(metrics_eval['epoch'], metrics_eval['accuracy'], label='accuracy (eval)', c='b', linestyle='dashed', linewidth=0.5)
    ax.plot(metrics_train['epoch'], metrics_train['precision'], label='precision (train)', c='r', linewidth=0.5)
    ax.plot(metrics_eval['epoch'], metrics_eval['precision'], label='precision (eval)', c='r', linestyle='dashed', linewidth=0.5)
    ax.plot(metrics_train['epoch'], metrics_train['recall'], label='recall (train)', c='g', linewidth=0.5)
    ax.plot(metrics_eval['epoch'], metrics_eval['recall'], label='recall (eval)', c='g', linestyle='dashed', linewidth=0.5)
    ax.plot(metrics_train['epoch'], metrics_train['f1 score'], label='f1 score (train)', c='k', linewidth=0.5)
    ax.plot(metrics_eval['epoch'], metrics_eval['f1 score'], label='f1 score (eval)', c='k', linestyle='dashed', linewidth=0.5)
    ax.set_xlabel('epoch', fontsize=6)
    ax.set_ylabel('metric', fontsize=6)
    ax.legend(fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    fig.tight_layout()
    fig.savefig(output / 'metrics_overall.png', dpi=600)

    # accuracy, precision, recall and F1-score for each task
    fig = plt.figure(figsize=(10, 6))
    n_tasks = df['task'].dropna().nunique()
    axs = fig.subplots(math.ceil(n_tasks/3), 3, sharex=True, sharey=True)
    for i_task, ax in enumerate(axs.flatten()):
        msk = df['batch'].isnull() & df['task'].notnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == 'train') & (df['task'] == i_task)
        metrics_train = df.loc[msk, ['epoch', 'task', 'number of datapoints', 'accuracy', 'precision', 'recall', 'f1 score']]
        msk = df['batch'].isnull() & df['task'].notnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == 'eval') & (df['task'] == i_task)
        metrics_eval = df.loc[msk, ['epoch', 'task', 'number of datapoints', 'accuracy', 'precision', 'recall', 'f1 score']]
        ax.hlines(0.8, 0, len(metrics_train), colors='k', linestyles='dashed', label='80%', linewidth=0.5)
        ax.plot(metrics_train['epoch'], metrics_train['accuracy'], label='accuracy (train)', c='b', linewidth=0.5)
        ax.plot(metrics_eval['epoch'], metrics_eval['accuracy'], label='accuracy (eval)', c='b', linestyle='dashed', linewidth=0.5)
        ax.plot(metrics_train['epoch'], metrics_train['precision'], label='precision (train)', c='r', linewidth=0.5)
        ax.plot(metrics_eval['epoch'], metrics_eval['precision'], label='precision (eval)', c='r', linestyle='dashed', linewidth=0.5)
        ax.plot(metrics_train['epoch'], metrics_train['recall'], label='recall (train)', c='g', linewidth=0.5)
        ax.plot(metrics_eval['epoch'], metrics_eval['recall'], label='recall (eval)', c='g', linestyle='dashed', linewidth=0.5)
        ax.plot(metrics_train['epoch'], metrics_train['f1 score'], label='f1 score (train)', c='k', linewidth=0.5)
        ax.plot(metrics_eval['epoch'], metrics_eval['f1 score'], label='f1 score (eval)', c='k', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel('epoch', fontsize=6)
        ax.set_ylabel('metric', fontsize=6)
        ax.set_title(f"task: {i_task}, train set size: {metrics_train['number of datapoints'].iloc[0]}, eval set size: {metrics_eval['number of datapoints'].iloc[0]}", fontsize=6)
        ax.legend(loc='lower center', fontsize=4, frameon=False, ncols=4)
        ax.tick_params(axis='both', which='major', labelsize=6)
    fig.tight_layout()
    fig.savefig(output / 'metrics_task.png', dpi=600)

    plt.interactive(True)

