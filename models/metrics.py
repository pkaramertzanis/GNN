import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
import math
import re

from typing import Union
import matplotlib.pyplot as plt

from sklearn.metrics import auc


def compute_metrics(tp, tn, fp, fn) -> dict:
    """
    Compute metrics (accuracy, precision, recall, f1 score) from the true positive, true negative, false positive, and
    false negative counts
    :param tp: true positive
    :param tn: true negative
    :param fp: false positive
    :param fn: false negative
    :return: dictionary with accuracy, precision, recall, and f1 score in addition to the input tp, tn, fp and fn
    """

    # accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # positive predictive value, precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # true positive rate, recall, sensitivity
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # true negative rate, specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # f1 score
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # balanced accuracy
    balanced_accuracy = (recall + specificity) / 2

    metrics = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
               'accuracy': accuracy, 'precision': precision, 'recall': recall, 'specificity': specificity,
               'f1 score': f1_score, 'balanced accuracy': balanced_accuracy}
    return metrics

def plot_metrics_convergence(metrics_history: dict,
                             task_names: list[str],
                             stages: list[str],
                             output: Path,
                             metrics: Union[list[str], None] = None,
                             drop_first_epoch=True):
    """
    Plot the metrics convergence as a function of the epoch number for a single model fit run
    :param metrics_history: dataframe with metrics
    :param task_names: list with task names
    :param stages: list with stages to plot, can be either ['train', 'eval'] for inner cross validation fits or
                   ['train+eval', 'test'] for outer cross validation fits
    :param output: output file path
    :param metrics: list with metrics to plot, if None then recall, specificity, and balanced accuracy and f1 score will be plotted
    :param drop_first_epoch: drop the first epoch from the plot
    :return:
    """
    plt.interactive(False)

    df = metrics_history.copy()

    if metrics is None:
        metrics = ['recall', 'specificity', 'balanced accuracy', 'f1 score']

    if stages != ['train', 'eval'] and stages != ['train+eval', 'test']:
        ex = ValueError(f"stages must be either ['train', 'eval'] or ['train+eval', 'test']")
        log.error(ex)
        raise ex
    else:
        stage1 = stages[0]
        stage2 = stages[1]

    if drop_first_epoch:
        msk = df['epoch'] > 0
        df = df.loc[msk].copy()

    # overall loss for train and eval set
    msk = df['batch'].isnull() & df['task'].isnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == stage1)
    loss_train = df.loc[msk, ['epoch', 'loss (mean)']]
    msk = df['batch'].isnull() & df['task'].isnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == stage2)
    loss_eval = df.loc[msk, ['epoch', 'loss (mean)']]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.subplots()
    ax.plot(loss_train['epoch'], loss_train['loss (mean)'], label=stage1, c='k', linewidth=0.5)
    ax.plot(loss_eval['epoch'], loss_eval['loss (mean)'], label=stage2, c='k', linestyle='dashed', linewidth=0.5)
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
        msk = df['batch'].isnull() & df['task'].notnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == stage1) & (df['task'] == i_task)
        loss_train = df.loc[msk, ['epoch', 'task', 'loss (mean)']]
        msk = df['batch'].isnull() & df['task'].notnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == stage2) & (df['task'] == i_task)
        loss_eval = df.loc[msk, ['epoch', 'task', 'loss (mean)']]
        ax.plot(loss_train['epoch'], loss_train['loss (mean)'], label=stage1, c='k', linewidth=0.5)
        ax.plot(loss_eval['epoch'], loss_eval['loss (mean)'], label=stage2, c='k', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel('epoch', fontsize=6)
        ax.set_ylabel('loss', fontsize=6)
        ax.set_title(f"task: {i_task} ({task_names[i_task]})", fontsize=2)
        ax.legend(fontsize=6)
        ax.tick_params(axis='both', which='major', labelsize=6)
    fig.tight_layout()
    fig.savefig(output / 'loss_task.png', dpi=600)
    plt.close(fig)

    colors = ['b', 'r', 'g', 'k', 'y', 'm', 'c']

    # overall accuracy, precision, recall and F1-score
    msk = df['batch'].isnull() & df['task'].isnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == stage1)
    metrics_train = df.loc[msk, ['epoch'] + metrics]
    msk = df['batch'].isnull() & df['task'].isnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == stage2)
    metrics_eval = df.loc[msk, ['epoch'] + metrics]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.subplots()
    ax.hlines(0.8, 0, metrics_train['epoch'].max(), colors='y', linestyles='-', label='80%', linewidth=0.5)
    ax.hlines(0.85, 0, metrics_train['epoch'].max(), colors='y', linestyles='-', label='85%', linewidth=0.5)
    for i_metric, metric in enumerate(metrics):
        ax.plot(metrics_train['epoch'], metrics_train[metric], label=metric+' ('+stage1+')', c=colors[i_metric], linewidth=0.5)
        ax.plot(metrics_eval['epoch'], metrics_eval[metric], label=metric+' ('+stage2+')', c=colors[i_metric], linestyle='dashed', linewidth=0.5)
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
        msk = df['batch'].isnull() & df['task'].notnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == stage1) & (df['task'] == i_task)
        metrics_train = df.loc[msk, ['epoch', 'task', 'number of datapoints'] + metrics]
        msk = df['batch'].isnull() & df['task'].notnull() & (df['type'] == 'aggregate (epoch)') & (df['stage'] == stage2) & (df['task'] == i_task)
        metrics_eval = df.loc[msk, ['epoch', 'task', 'number of datapoints'] + metrics]
        ax.hlines(0.8, 0, metrics_train['epoch'].max(), colors='y', linestyles='-', label='80%', linewidth=0.5)
        ax.hlines(0.85, 0, metrics_train['epoch'].max(), colors='y', linestyles='-', label='85%', linewidth=0.5)
        for i_metric, metric in enumerate(metrics):
            ax.plot(metrics_train['epoch'], metrics_train[metric], label=metric+' ('+stage1+')', c=colors[i_metric], linewidth=0.5)
            ax.plot(metrics_eval['epoch'], metrics_eval[metric], label=metric+' ('+stage2+')', c=colors[i_metric], linestyle='dashed', linewidth=0.5)
        ax.set_xlabel('epoch', fontsize=6)
        ax.set_ylabel('metric', fontsize=6)
        ax.set_title(f"task: {i_task} ({task_names[i_task]}), train set size: {metrics_train['number of datapoints'].iloc[0]}, eval set size: {metrics_eval['number of datapoints'].iloc[0]}", fontsize=2)
        ax.legend(loc='lower center', fontsize=4, frameon=False, ncols=4)
        ax.tick_params(axis='both', which='major', labelsize=6)
    fig.tight_layout()
    fig.savefig(output / 'metrics_task.png', dpi=600)
    plt.close(fig)

    plt.interactive(True)


def consolidate_metrics_outer(metrics_history_path: Path, metrics_outer_path: Path,
                              task_names: list[str],
                              configuration_parameters: list[str]) -> None:
    '''
    Computes the average metrics for each outer iteration of the nested cross-validation and overall, across all outer
    loop iterations
    :param metrics_history_path: path to the tab separated file with the collected metrics data
    :param metrics_outer_path: path to the output excel file with the consolidated metrics
    :return: None
    '''
    # retrieve the metrics for each outer iteration and list the optimal configuration for each outer iteration
    metrics_history = pd.read_csv(metrics_history_path, sep='\t')
    with pd.ExcelWriter(metrics_outer_path, engine='xlsxwriter') as writer:
        # over all
        # keep only the last three epochs (or averaging)
        msk1 = metrics_history['epoch'] >= metrics_history['epoch'].max() - 3
        # keep only the test and train+eval stages
        msk2 = metrics_history['stage'].isin(['test', 'train+eval'])
        # keep aggregates at epoch level
        msk3 = metrics_history['type'] == 'aggregate (epoch)'
        # keep aggregates across tasks
        msk4 = metrics_history['task'].isnull()
        # keep only the outer fold rows
        msk5 = metrics_history['inner fold'].isnull()
        # keep only the
        # average across the epochs
        res = (metrics_history
               .loc[msk1 & msk2 & msk3 & msk4 & msk5, ['outer fold', 'configuration_ID', 'stage', 'accuracy',
                                                       'precision', 'recall', 'specificity', 'f1 score',
                                                       'balanced accuracy']]
               .pivot_table(index=['outer fold', 'configuration_ID'], columns='stage',
                            values=['accuracy', 'precision', 'recall', 'specificity', 'f1 score', 'balanced accuracy'],
                            aggfunc='mean', margins=True)
               )
        res.columns = ['_'.join(col).strip() for col in res.columns.values]
        res = res.drop([col for col in res.columns if col.endswith('_All')], axis='columns')
        res = res.reset_index().merge(
            metrics_history[['outer fold'] + configuration_parameters].drop_duplicates(),
            on=['outer fold', 'configuration_ID'], how='left')
        res.to_excel(writer, sheet_name='overall', index=True)
        # per task
        # keep only the last three epochs (or averaging)
        msk1 = metrics_history['epoch'] >= metrics_history['epoch'].max() - 3
        # keep only the test and train+eval stages
        msk2 = metrics_history['stage'].isin(['test', 'train+eval'])
        # keep aggregates at epoch level
        msk3 = metrics_history['type'] == 'aggregate (epoch)'
        # keep aggregates across tasks
        msk4 = metrics_history['task'].notnull()
        # keep only the outer fold rows
        msk5 = metrics_history['inner fold'].isnull()
        # average across the epochs list(dsets.keys())
        res = (metrics_history
               .loc[msk1 & msk2 & msk3 & msk4 & msk5, ['outer fold', 'configuration_ID', 'task', 'stage', 'accuracy',
                                                       'precision', 'recall', 'specificity', 'f1 score',
                                                       'balanced accuracy', 'roc auc']]
               .pipe(lambda df: df.assign(**{'task': df['task'].map(lambda i_task: task_names[int(i_task)])}))  # map the task index to the task name
               .pivot_table(index=['outer fold', 'configuration_ID'], columns=['task', 'stage'],
                            values=['accuracy', 'precision', 'recall', 'specificity', 'f1 score', 'balanced accuracy', 'roc auc'],
                            aggfunc='mean', margins=True)
               .swaplevel(2, 0, axis='columns')
               .swaplevel(0, 1, axis='columns')
               )
        res.columns = ['_'.join(col).strip() for col in res.columns.values]
        res = res.drop([col for col in res.columns if col.startswith('All_')], axis='columns')
        res = res.reindex(res.columns.sort_values(), axis='columns')
        res = res.reset_index().merge(
            metrics_history[['outer fold'] + configuration_parameters].drop_duplicates(),
            on=['outer fold', 'configuration_ID'], how='left')
        res.to_excel(writer, sheet_name='task', index=True)
        res[[col for col in res.columns if 'f1 score' in col and 'test' in col]].to_clipboard()


def plot_metrics_convergence_outer_average(metrics_history_path: Path,
                                           metrics_convergence_outer_path: Path,
                                           task_names,
                                           metrics_specs: dict = None) -> None:
    '''
    Plot the average metrics for all outer iterations as a function of epoch (range is shown as a shaded area)
    :param metrics_history_path: path to the metrics history file
    :param metrics_convergence_outer_path: path to the output plot
    :param task_names: list of task names
    :param metrics_specs: dictionary with metrics to plot as keys and plot specifications as values. If None the
                          following dictionary is used:
                          metrics = {'recall': {'color': 'b', 'show std': False},
                                     'specificity': {'color': 'r', 'show std': False},
                                     'balanced accuracy': {'color': 'k', 'show std': True}}
    :return: None
    '''

    metrics_history = pd.read_csv(metrics_history_path, sep='\t')
    if metrics_specs is None:
        metrics = {'recall': {'color': 'b', 'show std': False},
                   'specificity': {'color': 'r', 'show std': False},
                   'balanced accuracy': {'color': 'k', 'show std': True}}

    # keep only the test and train+eval stages
    msk1 = metrics_history['stage'].isin(['test', 'train+eval'])
    # keep aggregates at epoch level
    msk2 = metrics_history['type'] == 'aggregate (epoch)'
    # keep aggregates for tasks
    msk3 = metrics_history['task'].notnull()
    # keep only the outer fold rows
    msk4 = metrics_history['inner fold'].isnull()
    res = metrics_history.loc[msk1 & msk2 & msk3 & msk4].melt(id_vars=['outer fold', 'epoch', 'stage', 'task'],
                                                              value_vars=metrics.keys(), var_name='metric name',
                                                              value_name='metric value')
    res = res.pivot(index=['epoch', 'stage', 'task', 'metric name'], columns='outer fold', values='metric value')
    res['mean'] = res.mean(axis='columns')
    res['std'] = res.std(axis='columns')
    res = res[['mean', 'std']].reset_index()
    res['task'].nunique()
    plt.interactive(False)
    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(res['task'].nunique(), 2, sharex=True, sharey=True).reshape(res['task'].nunique(), -1)
    for i_task in range(res['task'].nunique()):
        for i_stage, stage in enumerate(['train+eval', 'test']):
            msk = (res['stage'] == stage) & (res['task'] == i_task)
            for metric_name, settings in metrics.items():
                color = settings['color']
                show_std = settings['show std']
                msk2 = msk & (res['metric name'] == metric_name)
                x = res.loc[msk2, 'epoch']
                y = res.loc[msk2, 'mean']
                yerr = res.loc[msk2, 'std']
                axs[i_task, i_stage].plot(x, y, label=metric_name, color=color, linewidth=0.5)
                if show_std:
                    axs[i_task, i_stage].fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color, edgecolor=None)
                if i_stage == 0:
                    axs[i_task, i_stage].set_ylabel('metric')
                if i_task == 0:
                    axs[i_task, i_stage].set_title(stage, fontsize=10)
                if i_task == res['task'].nunique() - 1:
                    axs[i_task, i_stage].set_xlabel('epoch')
                # add the task name
                axs[i_task, i_stage].text(0.8, 0.1, task_names[i_task], style='normal', fontsize=6,
                                          transform=axs[i_task, i_stage].transAxes, ha='center')
                axs[i_task, i_stage].ylim = (0.75, 1)
                axs[i_task, i_stage].spines['top'].set_visible(False)
                axs[i_task, i_stage].spines['right'].set_visible(False)
                axs[i_task, i_stage].spines['bottom'].set_position(('outward', 5))
                axs[i_task, i_stage].spines['left'].set_position(('outward', 5))
            # draw horizontal lines at 0.8 and 0.85 to guide the eye
            axs[i_task, i_stage].hlines(0.8, 0, res['epoch'].max(), colors='k', linestyles='dotted', label=None,
                                        linewidth=0.5)
            axs[i_task, i_stage].hlines(0.85, 0, res['epoch'].max(), colors='k', linestyles='dotted', label=None,
                                        linewidth=0.5)
    # adjust the spacing around the subplots
    fig.subplots_adjust(bottom=0.12)
    fig.legend(metrics.keys(), fontsize=10, loc='lower center', frameon=False, ncol=3, bbox_to_anchor=(0.5, 0.00))
    # fig.tight_layout()
    fig.savefig(metrics_convergence_outer_path, dpi=600)
    plt.interactive(True)

def plot_roc_curve_outer_average(metrics_history_path: Path,
                                 roc_curve_outer_path: Path) -> None:
    '''
    Plots the average ROC curve for all outer iterations and all tasks
    :param metrics_history_path: path where the outer iteration metrics are stored
    :param roc_curve_outer_path: path to export the ROC curve plot
    :return:
    '''

    plt.interactive(False)

    # read in the roc curves
    rocs = []
    for outer_folder in set(metrics_history_path.glob('outer_fold_*')) - set(metrics_history_path.glob('*inner_fold_*')):
        i_outer = int(re.findall(r'^outer_fold_(\d+)_configuration_ID_(?:\d+)$',outer_folder.name)[0])
        try:
            rocs.append(pd.read_pickle(outer_folder/'roc_outer.pkl'))

        except:
            pass
    rocs = pd.concat(rocs, axis='index', ignore_index=True, sort=False)

    # keep the last epoch
    rocs = rocs.loc[rocs['epoch'] == rocs['epoch'].max()]

    # compute the ROC AUC
    rocs['roc auc'] = rocs['roc'].apply(lambda roc: auc(roc[0], roc[1]))
    aucs_task = rocs.pivot_table(index='task', columns='stage', values='roc auc', aggfunc=['mean', 'std', 'min', 'max'])
    log.info('ROC AUC (per task)\n'+aucs_task.to_markdown())

    aucs_overall = rocs.groupby('stage')[['roc auc']].agg(['mean', 'std', 'min', 'max']).droplevel(0, axis='columns')
    log.info('ROC AUC (overall)\n'+aucs_overall.to_markdown())


    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2, sharey=True)

    for i_stage, stage in enumerate(['train+eval', 'test']):
        # keep only the test stage
        msk = rocs['stage'] == stage
        roc_curves = rocs['roc'].loc[msk].to_list()
        # Interpolate ROC curves to a common FPR grid
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        tprs = []

        for fpr, tpr, _ in roc_curves:
            interp_tpr = np.interp(all_fpr, fpr, tpr)
            tprs.append(interp_tpr)
            mean_tpr += interp_tpr

        mean_tpr /= len(tprs)

        # Compute standard deviation of TPRs
        tprs = np.array(tprs)
        std_tpr = np.std(tprs, axis=0)

        for fpr, tpr, _ in roc_curves:
            axs[i_stage].plot(fpr, tpr, lw=0.5, alpha=0.2, color='k')

        # Plot mean ROC curve
        axs[i_stage].plot(all_fpr, mean_tpr, color='b', label='mean ROC', lw=2.)

        # Plot standard deviation area
        axs[i_stage].fill_between(all_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='grey', alpha=0.2, label='± 1 std. dev.', edgecolor=None)

        # Plot diagonal line for random classifier
        axs[i_stage].plot([0, 1], [0, 1], linestyle='--', color='k', label='random classifier', lw=1.)

        axs[i_stage].set_xlabel('false positive rate (1 - specificity)')
        axs[i_stage].set_ylabel('true positive rate')
        axs[i_stage].legend(loc='lower right', frameon=False)

        axs[i_stage].set_title(stage, fontsize=10)
        axs[i_stage].spines['top'].set_visible(False)
        axs[i_stage].spines['right'].set_visible(False)
        axs[i_stage].spines['bottom'].set_position(('outward', 5))
        axs[i_stage].spines['left'].set_position(('outward', 5))

        if stage == 'test':
            axs[i_stage].annotate(f"AUC {aucs_overall.loc['test','mean']:.3f} ± {aucs_overall.loc['test','std']:.3f}", xy=(0.5, 0.6),  horizontalalignment='right', fontsize=6)

        # add two points from the Hansen paper for comparison
        if stage == 'test':
            roc_hansen_points = ((0.15, 0.7), (0.225, 0.8))
            for i_point, roc_hansen_point in enumerate(roc_hansen_points):
                x, y = roc_hansen_point[0], roc_hansen_point[1]
                axs[i_stage].plot(x, y, 'ko', markersize=4)
                if i_point == 0:
                    axs[i_stage].annotate('Hansen 2009', xy=(x, y), xytext=(x, y + 0.3), fontsize=6, arrowprops=dict(facecolor='black', arrowstyle='-|>', linewidth=0.5), horizontalalignment='center')
                else:
                    delta_x = x - roc_hansen_points[i_point-1][0]
                    delta_y = y - roc_hansen_points[i_point-1][1]
                    axs[i_stage].annotate('', xy=(x, y), xytext=(x - delta_x, y + 0.3 - delta_y), arrowprops=dict(facecolor='black', arrowstyle='-|>', linewidth=0.5), horizontalalignment='center')

    fig.tight_layout()
    fig.savefig(roc_curve_outer_path, dpi=600)
    plt.interactive(True)