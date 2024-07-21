import numpy as np

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

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def train_eval(net: DMPNN_GCN,
               train_loaders: list[DataLoader],
               eval_loaders: list[DataLoader],
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.modules.loss._Loss,
               scheduler: torch.optim.lr_scheduler.LRScheduler,
               num_epochs: int,
               outp: Path,
               log_epoch_frequency: int = 10,
               scale_loss = None,
               metrics_history = None):
    """
    Train the GNN model using the eval set and evaluate its performance on the eval set. This function essentially
    executes a single inner loop of the nested cross-validation procedure.

    Although the loss function (cross entropy) allows multiclass classification (i.e. we can include the ambguous calls)
    the metrics reported assume binary classification and hence this function should only be used for binary classification tasks.
    In any case, the handling of ambiguous calls should be done using ordinal regression as the classes have order.

    :param net: PyTorch Geometric model
    :param train_loaders: PyTorch Geometric DataLoaders with the training data for the different tasks
    :param eval_loaders: PyTorch Geometric DataLoaders with the eval data for the different tasks
    :param optimizer: PyTorch optimizer
    :param loss_fn: PyTorch loss function
    :param scheduler: PyTorch learning rate scheduler
    :param outp: Path to store the model weights every log_epoch_frequency epochs, if None model weights are not stored
    :param num_epochs: number of epochs
    :param log_epoch_frequency: log the metrics every so many epochs
    :param scale_loss: if None then each task contributes to the loss function according to the number of datapoints in each task,
                       and if 'equal task' each task contributes equally
    :param metrics_history: list with metrics history to append to, if None, a new list is created
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
            total_batch_size = sum([len(task_batch) for task_batch in batches])
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

                loss_task = loss_fn(pred, y) # this is the mean loss (default)
                if scale_loss is None:
                    loss += loss_task * len(task_batch)
                elif scale_loss == 'equal task':
                    loss += loss_task * (total_batch_size/len(batches))

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

        scheduler.step()

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



            # evaluate the model on the eval set
            net.eval()
            for i_batch, batches in enumerate(zip_recycle(*eval_loaders)):
                metrics_batch = []
                loss = torch.tensor(0.)
                n_datapoints = 0
                total_batch_size = sum([len(task_batch) for task_batch in batches])
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

                    loss_task = loss_fn(pred, y) # this is mean loss (default)
                    if scale_loss is None:
                        loss += loss_task * len(task_batch)
                    elif scale_loss == 'equal task':
                        loss += loss_task * (total_batch_size / len(batches))

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

            # store the model weight absolute delta percentiles in subsequent epochs to check convergence
            if outp is not None:
                weight_values = [w_value for w_name in net.state_dict() for w_value in net.state_dict()[w_name].cpu().numpy().flatten()]
                if i_epoch > 0:
                    # with open(outp, 'ta') as f:
                    weight_abs_diff = np.abs(np.array(weight_values) - np.array(weight_values_previous))
                    weight_abs_diff_quantiles = np.quantile(weight_abs_diff, np.arange(0.01, 1, 0.01))
                    with open(outp, 'ta') as f:
                        f.write('\n' + '\t'.join([f'{diff:0.5}' for diff in weight_abs_diff_quantiles]))
                else:
                    with open(outp, 'ta') as f:
                        f.write('\t'.join([f'{quant:0.5}' for quant in np.arange(0.01, 1, 0.01)]))
                weight_values_previous = weight_values

                #
                # weight_values_previous = None if i_epoch == 0 else weight_values
                #
                #
                # with open(outp, 'ta') as f:
                #     if i_epoch == 0:
                #         weight_names = '\t'.join([w_name for w_name in net.state_dict().keys()])
                #         f.write(weight_names)
                #     weight_values = '\n' + '\t'.join([f'{w_value:0.5}' for w_name in net.state_dict() for w_value in net.state_dict()[w_name].cpu().numpy().flatten()])
                #     f.write(weight_values)

    return metrics_history


