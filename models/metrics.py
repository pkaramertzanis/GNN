import logger
log = logger.get_logger(__name__)

import pandas as pd
from pathlib import Path
import math

import matplotlib.pyplot as plt



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
    plt.close(fig)


    # loss for train and eval set for each task
    fig = plt.figure(figsize=(10, 6))
    n_tasks = df['task'].dropna().nunique()
    axs = fig.subplots(math.ceil(n_tasks/3), 3, sharex=True, sharey=True)
    for i_task, ax in enumerate(axs.flatten()[slice(n_tasks)]):
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
    plt.close(fig)


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
    plt.close(fig)

    # accuracy, precision, recall and F1-score for each task
    fig = plt.figure(figsize=(10, 6))
    n_tasks = df['task'].dropna().nunique()
    axs = fig.subplots(math.ceil(n_tasks/3), 3, sharex=True, sharey=True)
    for i_task, ax in enumerate(axs.flatten()[slice(n_tasks)]):
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
    plt.close(fig)


    plt.interactive(True)

