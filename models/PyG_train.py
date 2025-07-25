import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np

from pathlib import Path
import math
from typing import Union
import time
from collections import deque

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary

import matplotlib.pyplot as plt

from models.MPNN_GNN import MPNN_GNN
from models.metrics import compute_metrics

from utilities import zip_recycle

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, roc_curve

def train_eval(net,
               train_loaders: list[DataLoader],
               eval_loaders: list[DataLoader],
               global_loss_fn: Union[torch.nn.CrossEntropyLoss, None],
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler.LRScheduler,
               num_epochs: int,
               weight_converge_path: Union[Path, None],
               early_stopping: Union[dict, None] = {'loss_eval': 20, 'roc_eval': 10, 'threshold': 0.},
               log_epoch_frequency: int = 10,
               scale_loss_task_size: Union[None, str] = None,
               metrics_history = None) -> list[dict]:
    """
    Train the GNN model using the eval set and evaluate its performance on the eval set. This function essentially
    executes a single inner loop of the nested cross-validation procedure. It is also used for refitting the model with
    the optimal configuration in the outer loop.

    Although the loss function (cross entropy) allows multiclass classification (i.e. we can include the ambiguous calls)
    the metrics reported assume binary classification and hence this function should only be used for binary classification tasks.
    In any case, the handling of ambiguous calls should be done using ordinal regression as the classes have order.

    :param net: PyTorch Geometric model
    :param train_loaders: PyTorch Geometric DataLoaders with the training data for the different tasks
    :param eval_loaders: PyTorch Geometric DataLoaders with the eval data for the different tasks
    :param global_loss_fn: PyTorch cross entropy loss function, if None we define a loss function per task/batch by scaling the positives
    :param optimizer: PyTorch optimizer
    :param scheduler: PyTorch learning rate scheduler
    :param weight_converge_path: Path to store the model weights every log_epoch_frequency epochs, if None model weights are not stored
    :param num_epochs: maximum number of epochs to train the model
    :param early_stopping: dictionary with the early stopping criteria, if None early stopping is not applied. For example,
                           {'loss_eval': 20, 'roc_eval': 10, 'threshold': 0.} means stop training if the loss on the eval set does not
                           improve for 20 epochs or the ROC AUC on the eval set does not improve for 10 epochs; threshold defines how large the improvement should be
    :param log_epoch_frequency: log the metrics every so many epochs, if early stopping is set then logging is done in
                                every epoch. The last 3 epochs are always reported
    :param scale_loss_task_size: if None then each task contributes to the loss function according to the number of datapoints in each task,
                                 and if 'equal task' each task contributes equally
    :param metrics_history: list with metrics history to append to, if None, a new list is created
    :return: list with metrics history dictionaries (one dictionary per logging event), and list with the model summary for each task

    Note: Early stopping has not been implemented. For an easy implementation in PyTorch see
          https://medium.com/@vrunda.bhattbhatt/a-step-by-step-guide-to-early-stopping-in-tensorflow-and-pytorch-59c1e3d0e376
    """

    # metrics history, continue from the previous run if not None
    if metrics_history is None:
        metrics_history = []

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataframe to control early stopping
    early_stopping_loss_roc = pd.DataFrame()

    # model summary
    model_summary = []

    # train
    for i_epoch in range(num_epochs):
        # clear up the cache before each epoch
        torch.cuda.empty_cache()

        metrics_epoch = []
        net.train()

        # loop over the batches, the train loader for each task may have slightly different number of batches, in which case we recycle the shorter train loaders
        for i_batch, batches in enumerate(zip_recycle(*train_loaders)):

            metrics_batch = []
            optimizer.zero_grad()
            loss = torch.tensor(0.)
            n_datapoints = 0
            total_batch_size = sum([len(task_batch) for task_batch in batches])
            losses_task = torch.zeros(len(train_loaders)).to(device)
            for i_task, task_batch in enumerate(batches):
                task_batch.to(device)
                metrics_batch_task = {'epoch': i_epoch, 'batch': i_batch, 'task': i_task, 'stage': 'train', 'type': 'raw', 'number of datapoints': len(task_batch)}
                n_datapoints += len(task_batch)

                y = [1 if assay_data=='positive' else 0 for assay_data in task_batch.assay_data]
                fraction_positives = sum(y)/float(len(y))
                y = torch.tensor(y, dtype=torch.long).to(device)

                # pred = net(batch)
                pred = net(task_batch.x, task_batch.edge_index, task_batch.edge_attr, task_batch.batch, task_id=i_task)

                # summarise the model
                if i_epoch == 0 and i_batch == 0:
                    model_summary.append({'task': i_task, 'summary': summary(net, task_batch.x, task_batch.edge_index, task_batch.edge_attr, task_batch.batch, task_id=i_task)})

                # true positive, true negative, false positive, and false negative counts for each task
                pred_class = torch.argmax(pred.detach(), dim=1)
                tp = ((pred_class == y) & (y == 1)).int().sum().item()
                tn = ((pred_class == y) & (y == 0)).int().sum().item()
                fp = ((pred_class != y) & (y == 0)).int().sum().item()
                fn = ((pred_class != y) & (y == 1)).int().sum().item()


                # if the loss function is not specified, we define a loss function per task/batch by scaling the positives
                if global_loss_fn is None:
                    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([fraction_positives, 1. - fraction_positives]).to(device))
                else:
                    loss_fn = global_loss_fn

                # compute the loss
                loss_task = loss_fn(pred, y) # this is the mean loss (default)

                # if specified, scale the loss so that each task contributes according to its size or equally
                if scale_loss_task_size is None:
                    # loss += loss_task * len(task_batch)
                    losses_task[i_task] = loss_task * len(task_batch)
                elif scale_loss_task_size == 'equal task':
                    # loss += loss_task * (total_batch_size/len(batches))
                    losses_task[i_task] = loss_task * (total_batch_size/len(batches))
                loss = losses_task.sum()

                metrics_batch_task['loss (mean)'] = loss_task.item()
                metrics_batch_task.update(compute_metrics(tp, tn, fp, fn))
                metrics_batch.append(metrics_batch_task)

            # report the metrics for the batch
            tmp = pd.DataFrame(metrics_batch)
            loss_mean = (tmp['number of datapoints']*tmp['loss (mean)']).sum()/tmp['number of datapoints'].sum()
            tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), tmp['number of datapoints'].sum()
            tmp = {'epoch': i_epoch, 'batch': i_batch, 'task': None, 'stage': 'train', 'type': 'aggregate (batch)', 'number of datapoints': number_of_datapoints}
            tmp['loss (mean)'] = loss_mean
            tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
            metrics_batch.append(tmp)

            # # log the metrics
            # log.info(pd.DataFrame(metrics_batch))

            metrics_epoch.extend(metrics_batch)

            # average the loss for all tasks in the batch and back propagate
            start_time = time.time()
            loss = loss/float(n_datapoints)
            loss.backward()
            optimizer.step()
            log.info(f'backward pass took {time.time()-start_time:.2} seconds')

        # store the model weight absolute delta percentiles in subsequent epochs to check convergence
        if weight_converge_path is not None:
            weight_values = [w_value for w_name in net.state_dict() for w_value in
                             net.state_dict()[w_name].cpu().numpy().flatten()]
            if i_epoch > 0:
                # with open(weight_converge_path, 'ta') as f:
                weight_abs_diff = np.abs(np.array(weight_values) - np.array(weight_values_previous))
                weight_abs_diff_quantiles = np.quantile(weight_abs_diff, np.arange(0.01, 1, 0.01))
                with open(weight_converge_path, 'ta') as f:
                    f.write('\n' + '\t'.join([f'{diff:0.5}' for diff in weight_abs_diff_quantiles]))
            else:
                with open(weight_converge_path, 'ta') as f:
                    f.write('\t'.join([f'{quant:0.5}' for quant in np.arange(0.01, 1, 0.01)]))
            weight_values_previous = weight_values

        scheduler.step()

        # check if reporting on the evaluation set needs to take place
        if early_stopping is None:
            do_eval_reporting = (i_epoch % log_epoch_frequency == 0 and i_epoch>0) or (i_epoch >= num_epochs - 3) # the last 3 epochs are always reported in case we wish to average the metrics
        else:
            do_eval_reporting = True # always report the metrics if early stopping is set because we need to monitor progress constantly

        if do_eval_reporting:
            net.eval()

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

            with torch.no_grad():
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

                    # compute the ROC AUC for the train set
                    prob_train_epoch = []
                    y_train_epoch = []
                    for task_batch in train_loaders[i_task]:
                        task_batch.to(device)
                        pred = net(task_batch.x, task_batch.edge_index, task_batch.edge_attr, task_batch.batch, task_id=i_task)
                        prob_train_epoch.append(torch.nn.functional.softmax(pred, dim=1).detach().cpu().numpy())
                        y_train_epoch.append([1 if assay_data == 'positive' else 0 for assay_data in task_batch.assay_data])
                    prob_train_epoch = np.concatenate(prob_train_epoch, axis=0)
                    y_train_epoch = np.concatenate(y_train_epoch, axis=0)
                    roc_auc_train =  roc_auc_score(y_train_epoch, prob_train_epoch[:, 1])
                    roc_train = roc_curve(y_train_epoch, prob_train_epoch[:, 1])
                    tmp['roc auc'] = roc_auc_train
                    tmp['roc'] = roc_train # tuple with fpr, tpr, thresholds

                    metrics_epoch.append(tmp)

                # evaluate the model on the eval set
                for i_batch, batches in enumerate(zip_recycle(*eval_loaders)):
                    task_batch.to(device)
                    metrics_batch = []
                    loss = torch.tensor(0.).to(device)
                    n_datapoints = 0
                    total_batch_size = sum([len(task_batch) for task_batch in batches])
                    for i_task, task_batch in enumerate(batches):
                        task_batch.to(device)
                        metrics_batch_task = {'epoch': i_epoch, 'batch': i_batch, 'task': i_task, 'stage': 'eval', 'type': 'raw', 'number of datapoints': len(task_batch)}
                        n_datapoints += len(task_batch)

                        y = [1 if assay_data == 'positive' else 0 for assay_data in task_batch.assay_data]
                        fraction_positives = sum(y)/float(len(y))
                        y = torch.tensor(y, dtype=torch.long).to(device)

                        pred = net(task_batch.x, task_batch.edge_index, task_batch.edge_attr, task_batch.batch, task_id=i_task)

                        # true positive, true negative, false positive, and false negative counts for each task
                        pred_class = torch.argmax(pred, dim=1).detach()
                        tp = ((pred_class == y) & (y == 1)).int().sum().item()
                        tn = ((pred_class == y) & (y == 0)).int().sum().item()
                        fp = ((pred_class != y) & (y == 0)).int().sum().item()
                        fn = ((pred_class != y) & (y == 1)).int().sum().item()

                        # if the loss function is not specified, we define a loss function per task/batch by scaling the positives
                        if global_loss_fn is None:
                            loss_fn = torch.nn.CrossEntropyLoss(
                                weight=torch.tensor([fraction_positives, 1. - fraction_positives]).to(device))
                        else:
                            loss_fn = global_loss_fn

                        loss_task = loss_fn(pred, y) # this is mean loss (default)

                        # if specified, scale the loss so that each task contributes according to its size or equally
                        if scale_loss_task_size is None:
                            loss += loss_task * len(task_batch)
                        elif scale_loss_task_size == 'equal task':
                            loss += loss_task * (total_batch_size / len(batches))

                        metrics_batch_task['loss (mean)'] = loss_task.item()
                        metrics_batch_task.update(compute_metrics(tp, tn, fp, fn))

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

                    # # log the metrics
                    # log.info(pd.DataFrame(metrics_batch))

                    metrics_epoch.extend(metrics_batch)


            # report the metrics for the epoch (for each task)
            prob_eval_epoch_all_tasks = []
            y_eval_epoch_all_tasks = []
            for i_task in range(len(train_loaders)):
                tmp = pd.DataFrame(metrics_epoch)
                msk = (tmp['stage'] == 'eval') & (tmp['type'] == 'raw') & (tmp['task'] == i_task)
                tmp = tmp.loc[msk]
                loss_mean = (tmp['number of datapoints']*tmp['loss (mean)']).sum()/tmp['number of datapoints'].sum()
                tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), tmp['number of datapoints'].sum()
                tmp = {'epoch': i_epoch, 'batch': None, 'task': i_task, 'stage': 'eval', 'type': 'aggregate (epoch)', 'number of datapoints': number_of_datapoints}
                tmp['loss (mean)'] = loss_mean
                tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
                # compute the ROC AUC for the eval set (for each task)
                prob_eval_epoch = []
                y_eval_epoch = []
                for task_batch in eval_loaders[i_task]:
                    task_batch.to(device)
                    pred = net(task_batch.x, task_batch.edge_index, task_batch.edge_attr, task_batch.batch, task_id=i_task)
                    prob_eval_epoch.append(torch.nn.functional.softmax(pred, dim=1).detach().cpu().numpy())
                    y_eval_epoch.append([1 if assay_data == 'positive' else 0 for assay_data in task_batch.assay_data])
                prob_eval_epoch_all_tasks.extend(prob_eval_epoch)
                y_eval_epoch_all_tasks.extend(y_eval_epoch)
                prob_eval_epoch = np.concatenate(prob_eval_epoch, axis=0)
                y_eval_epoch = np.concatenate(y_eval_epoch, axis=0)
                roc_auc_eval =  roc_auc_score(y_eval_epoch, prob_eval_epoch[:, 1])
                roc_eval = roc_curve(y_eval_epoch, prob_eval_epoch[:, 1])
                tmp['roc auc'] = roc_auc_eval
                tmp['roc'] = roc_eval  # tuple with fpr, tpr, thresholds
                metrics_epoch.append(tmp)


            # report the metrics for the epoch (for all tasks)
            tmp = pd.DataFrame(metrics_epoch)
            msk = (tmp['stage'] == 'eval') & (tmp['type'] == 'raw')
            tmp = tmp.loc[msk]
            loss_mean = (tmp['number of datapoints']*tmp['loss (mean)']).sum()/tmp['number of datapoints'].sum()
            tp, tn, fp, fn, number_of_datapoints = tmp['tp'].sum(), tmp['tn'].sum(), tmp['fp'].sum(), tmp['fn'].sum(), tmp['number of datapoints'].sum()
            tmp = {'epoch': i_epoch, 'batch': None, 'task': None, 'stage': 'eval', 'type': 'aggregate (epoch)', 'number of datapoints': number_of_datapoints}
            tmp['loss (mean)'] = loss_mean
            # compute the ROC AUC for the eval set (for all tasks)
            prob_eval_epoch_all_tasks = np.concatenate(prob_eval_epoch_all_tasks, axis=0)
            y_eval_epoch_all_tasks = np.concatenate(y_eval_epoch_all_tasks, axis=0)
            roc_auc_eval_all_tasks = roc_auc_score(y_eval_epoch_all_tasks, prob_eval_epoch_all_tasks[:, 1])
            roc_eval_all_tasks = roc_curve(y_eval_epoch_all_tasks, prob_eval_epoch_all_tasks[:, 1])
            tmp['roc auc'] = roc_auc_eval_all_tasks
            tmp['roc'] = roc_eval_all_tasks
            early_stopping_loss_roc = pd.concat([early_stopping_loss_roc, pd.Series({'epoch': i_epoch, 'loss (mean)': loss_mean, 'roc auc': roc_auc_eval_all_tasks}).to_frame().T], ignore_index=True, axis='index', sort=False)
            tmp.update(compute_metrics(tp.item(), tn.item(), fp.item(), fn.item()))
            metrics_epoch.append(tmp)

            metrics_history.extend(metrics_epoch)

            # apply early stopping
            if early_stopping is not None:
                tmp = early_stopping_loss_roc.assign(**{'loss (mean) minimum': None, 'roc auc (maximum)': None}).set_index('epoch')
                tmp.loc[tmp['loss (mean)'].idxmin(), 'loss (mean) minimum'] = '<- min loss (mean)'
                tmp.loc[tmp['roc auc'].idxmax(), 'roc auc (maximum)'] = '<- max roc auc'
                log.info('early stopping history based on the loss and roc auc of the evaluation/test set\n' + tmp.to_markdown())

                threshold = early_stopping['threshold']
                # smallest epoch with the eval loss within (1+threshold)*minimum
                min_loss_mean_epoch = (early_stopping_loss_roc['loss (mean)'] <= early_stopping_loss_roc['loss (mean)'].min() * (1 + threshold)).idxmax()
                # smallest epoch with the eval roc auc within (1-threshold)*maximum
                max_roc_auc_epoch = (early_stopping_loss_roc['roc auc'] >= early_stopping_loss_roc['roc auc'].max() * (1 - threshold)).idxmax()
                if i_epoch > min_loss_mean_epoch + early_stopping['loss_eval'] and i_epoch > max_roc_auc_epoch + early_stopping['roc_eval']:
                    log.info(f'early stopping at epoch {i_epoch}, evaluation loss did not significantly improve for {early_stopping["loss_eval"]} epochs and ROC AUC did not significantly improve for {early_stopping["roc_eval"]} epochs')
                    break

                # if early_stopping_loss_roc['loss (mean)'].values.argmin() < len(early_stopping_loss_roc) - early_stopping['loss_eval'] and \
                #    early_stopping_loss_roc['roc auc'].values.argmax() < len(early_stopping_loss_roc) - early_stopping['roc_eval']:
                #        log.info(f'early stopping at epoch {i_epoch}, evaluation loss did not improve for {early_stopping["loss_eval"]} epochs and ROC AUC did not improve for {early_stopping["roc_eval"]} epochs')
                #        break

        # log the metrics
        log.info(pd.DataFrame(metrics_epoch).drop(columns='roc'))

    # empty the cache
    torch.cuda.empty_cache()

    return metrics_history, model_summary