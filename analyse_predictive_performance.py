'''
Analyse the combined dataset used for model development
'''
# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from tqdm import tqdm

import math
import pickle

import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr',False)
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


# negative AMES and positive other in vitro tests
data = pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\training_eval_dataset\tabular\genotoxicity_dataset.xlsx')
msk = data['genotoxicity'].isin(['positive', 'negative'])
tmp = data.loc[msk]
tmp = tmp.pivot(index='smiles_std', columns='task', values='genotoxicity')
msk1 = tmp['in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay'].isin(['negative'])
cols = ['in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test',
        'in vitro, in vitro gene mutation study in mammalian cells',
       'in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test']
msk2 = tmp[cols].isin(['positive']).any(axis='columns')
print(f'{msk1.sum()} structures with negative AMES out of which {(msk1 & msk2).sum()} are positive in one or more other in vitro tests')

# predictivity of the TA98/100 strains for the overall AMES outcome
data = pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_5_strains\training_eval_dataset\tabular\genotoxicity_dataset.xlsx')
msk = data['genotoxicity'].isin(['positive', 'negative']) & data['task'].str.contains('in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay')
tmp = data.loc[msk]
tmp = tmp.pivot(index='smiles_std', columns='task', values='genotoxicity')
cols = [col for col in tmp.columns if 'TA 98' in col or 'TA 100' in col]
msk1 = tmp[cols].isin(['positive']).sum(axis='columns')>=1
msk2 = tmp.isin(['positive']).sum(axis='columns')>=1
msk3 = tmp[cols].notnull().sum(axis='columns')==4
print(f"from the {msk2.sum()} structures with a positive Ames outcome, {msk1.sum()} are positive in the TA98/100 strains")
print(f"from the {msk2.sum()-msk1.sum()} structures with a positive Ames outcome and no positive outcome in the TA98/100 strains, {(msk3 & ~msk1 & msk2).sum()} have all TA 98/100 S9+- tests")

# number of unique structures per task
data = pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\training_eval_dataset\tabular\genotoxicity_dataset.xlsx')
print('number of unique structures per task:')
print(data.groupby('task')['smiles_std'].nunique())

# number of unique structures overall
print('number of unique structures overall:')
print(data['smiles_std'].nunique())

# number of unique structures with positive and negative genotoxicity calls that are used for modelling per task
msk = data['genotoxicity'].isin(['positive', 'negative'])
data.loc[msk].groupby('task')['smiles_std'].nunique()
print('number of unique structures with positive and negative genotoxicity calls that are used for modelling per task:')
print(data.loc[msk].groupby('task')['smiles_std'].nunique())

# number of unique structures with positive and negative genotoxicity calls that are used for modelling overall
print('number of unique structures with positive and negative genotoxicity calls that are used for modelling overall:')
print(data.loc[msk, 'smiles_std'].nunique())

# ratio for positives and negatives in the REACH database
tmp = data.copy()
tmp['genotoxicity details'] = data['genotoxicity details'].apply(json.loads)
tmp = tmp.explode('genotoxicity details')
tmp[['genotoxicity (source)', 'source']] = tmp['genotoxicity details'].apply(lambda row: pd.Series([row['genotoxicity (source)'], row['source']]))
msk = (tmp['genotoxicity (source)'].isin(['positive', 'negative'])) & (tmp['source'] == 'REACH data')
res = tmp.loc[msk].groupby(['task', 'genotoxicity (source)'])['smiles_std'].nunique().unstack()
print('ratio of negatives to positives in the REACH database per task:')
print(res['negative']/ res['positive'])

# ratio for positives and negatives overall
msk = data['genotoxicity'].isin(['positive', 'negative'])
res = data.loc[msk].groupby(['task', 'genotoxicity'])['smiles_std'].nunique().unstack()
print('ratio of negatives to positives per task:')
print(res['negative']/ res['positive'])

# number of structures with conflicting calls among the different databases per task as a percentage
tmp = data.copy()
tmp['genotoxicity details'] = data['genotoxicity details'].apply(json.loads)
tmp = tmp.explode('genotoxicity details')
tmp[['genotoxicity (source)', 'source']] = tmp['genotoxicity details'].apply(lambda row: pd.Series([row['genotoxicity (source)'], row['source']]))
tmp = tmp.groupby(['smiles_std', 'task'])['genotoxicity (source)'].apply(lambda x: 'yes' if len({'positive', 'negative'}.intersection(set(x)))==2 else 'no').rename('conflicting genotoxicity').reset_index()
tmp = tmp.groupby(['task', 'conflicting genotoxicity'])['smiles_std'].nunique().unstack()
tmp['percentage'] = tmp['yes']/(tmp['yes']+tmp['no'])*100
print('percentage of structures with conflicting calls among the different databases per task:')
print(tmp)

# number of structures with different number of positive and negative test results available
msk = data['genotoxicity'].isin(['positive', 'negative'])
res = data.loc[msk].groupby(['smiles_std'])['task'].nunique().value_counts().sort_index()
print('number of structures with different number of positive and negative genotoxicity assay calls available:')
print(res)


# cross-validation model performance metrics
import re
model_fit_folder = Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN')
try:
    # load the fingerprint parameters and task names
    with open(model_fit_folder/'fingerprint_tasks.json', 'r') as f:
        fingerprint_tasks = json.load(f)
        tasks = fingerprint_tasks['tasks']
except:
    with (open(model_fit_folder / 'feature_task_info.json', 'r') as f):
        all_features_tasks = json.load(f)
        tasks = all_features_tasks['tasks']
# find the best trial (set manually if the study pickle cannot be read)
study_trials = pd.read_excel(model_fit_folder/'study.xlsx')
best_trial = study_trials.loc[study_trials['value'].idxmax(), 'number']
print('best trial ', best_trial)
# best_trial = 22
# The approach is as follows
# - read the metrics for each fit
# - for each fit find the best epoch based on the overall evaluation balanced accuracy
# - find the best fit for each fold
# - use the best fits to compute the metrics for each task
fits = list(model_fit_folder.glob(rf'trial_{best_trial}_fold_*_model_fit_*'))
cross_val_metrics = []
for i_fit, fit in tqdm(enumerate(fits)):
    print(f'processing fit {fit}')
    entry = re.fullmatch(r'trial_(?P<trial>\d+)_fold_(?P<fold>\d+)_model_fit_(?P<fit>\d+)', fit.stem).groupdict()
    metrics_history = pd.read_excel(Path(fit)/'metrics_history.xlsx')
    # .. fetch the best epoch and corresponding evaluation overall balanced accuracy
    msk = (metrics_history['stage'] == 'eval') & (metrics_history['type'] == 'aggregate (epoch)') & (metrics_history['task'].isnull())
    entry['best epoch'] = metrics_history.loc[metrics_history.loc[msk, 'balanced accuracy'].idxmax(), 'epoch']
    entry['evaluation overall balanced accuracy (best epoch)'] = metrics_history.loc[msk, 'balanced accuracy'].max()
    # .. fetch the evaluation metrics for each task
    msk = (metrics_history['epoch'] == entry['best epoch']) & (metrics_history['stage'] == 'eval') & (metrics_history['type'] == 'aggregate (epoch)') & (metrics_history['task'].notnull())
    metric_names = ['recall', 'specificity', 'balanced accuracy', 'roc auc']
    entry['evaluation task metrics (best epoch)'] = metrics_history.loc[msk, ['task']+metric_names].reset_index()
    entry['evaluation task metrics (best epoch)']['task'] = entry['evaluation task metrics (best epoch)']['task'].astype(int).apply(lambda x: tasks[x])
    cross_val_metrics.append(entry)
cross_val_metrics = pd.DataFrame(cross_val_metrics).reset_index(drop=True)
print('objective function maximum ', cross_val_metrics.groupby('fold')['evaluation overall balanced accuracy (best epoch)'].max().mean())
# .. find the best fit for each fold
best_fold_fit_combinations = cross_val_metrics.loc[cross_val_metrics.groupby('fold')['evaluation overall balanced accuracy (best epoch)'].idxmax().to_list(), ['fold', 'fit']]
# use the best fit in each fold to compute the metrics for each task
res = cross_val_metrics.merge(best_fold_fit_combinations, on=['fold', 'fit'], how='inner')
res['evaluation task metrics (best epoch)'] = res['evaluation task metrics (best epoch)'].apply(lambda x: x.to_dict(orient='records'))
res = res.explode(column='evaluation task metrics (best epoch)').reset_index(drop=True)
res = pd.concat([res[['trial', 'fold', 'fit', 'best epoch', 'evaluation overall balanced accuracy (best epoch)']], pd.json_normalize(res['evaluation task metrics (best epoch)'])], axis='columns')
stats_all = []
for task in res['task'].unique():
    msk = res['task'] == task
    stats = res.loc[msk].describe().loc[['mean', 'std']]
    stats = stats[metric_names].apply(lambda row: '{:.3f} ± {:.3f}'.format(row['mean'], row['std']), axis='index')
    stats = stats.to_frame().T
    stats.insert(loc=0, column='task', value=task)
    stats_all.append(stats)
stats_all = pd.concat(stats_all, axis='index', ignore_index=True, sort=False)
stats_all.to_clipboard()
print(stats_all)




# boxplot, one plot per task and metric (all tasks)
# perf_metrics_path = r'D:\myApplications\local\2024_01_21_GCN_Muta\output/compare_predictive_performance_AMES_5_strains.xlsx'
perf_metrics_path = r'D:\myApplications\local\2024_01_21_GCN_Muta\output/compare_predictive_performance_AMES_agg_GM_CA_MN.xlsx'
perf_metrics = pd.read_excel(perf_metrics_path, sheet_name='cross-validation', skiprows=[0])
# perf_metrics = pd.read_excel(perf_metrics_path, sheet_name='external (hold-out) test set', skiprows=[0])
model_architecture_desired_order = ['FFNN', 'Att. FP', 'GAT', 'MPNN']
# model_architecture_desired_order = ['FFNN']
assay_desired_order = ['Ames', 'GM', 'CA', 'MN']
# assay_desired_order = ['TA100\nS9+', 'TA100\nS9-', 'TA98\nS9+', 'TA98\nS9-', 'TA1535\nS9+', 'TA1535\nS9-', 'TA1537\nS9+', 'TA1537\nS9-', 'E. coli WP2\nUvr A S9+', 'E. coli WP2\nUvr A S9-']
# assay_desired_order = ['TA100\nS9+', 'TA100\nS9-', 'TA98\nS9+', 'TA98\nS9-', 'TA1535\nS9+', 'TA1535\nS9-', 'TA1537\nS9+', 'TA1537\nS9-']
metrics_desired_order = ['sens.', 'spec.', 'bal. acc.', 'AUC']
mt_st_desired_order = ['ST', 'MT']
perf_metrics['model architecture'] = pd.Categorical(perf_metrics['model architecture'], categories=model_architecture_desired_order, ordered=True)
perf_metrics['task'] = pd.Categorical(perf_metrics['task'], categories=assay_desired_order, ordered=True)
perf_metrics['MT/ST'] = pd.Categorical(perf_metrics['MT/ST'], categories=mt_st_desired_order, ordered=True)
# fig = plt.figure(figsize=(10,  6))
fig = plt.figure(figsize=(10,  12))
n_metrics = len(metrics_desired_order)
n_assays = len(assay_desired_order)
axs = fig.subplots(nrows=n_assays+1, ncols=n_metrics+1, gridspec_kw={'width_ratios': [0.2]+[1.]*n_metrics, 'height_ratios': [0.2]+[1.]*n_assays})
for i_ax in range(n_assays+1):
    if i_ax > 0:
        assay = assay_desired_order[i_ax-1]
    for j_ax in range(n_metrics+1):
        if i_ax == j_ax == 0:
            axs[i_ax, j_ax].axis('off')
            # .. add legend
            legend_labels = ['MT', 'ST']
            legend_colors = ['k', 'r']
            legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in legend_colors]
            axs[i_ax, j_ax].legend(legend_handles, legend_labels, loc='lower right', fontsize=10, frameon=False)
        if j_ax > 0:
            metric = metrics_desired_order[j_ax - 1]
        if (i_ax > 0) & (j_ax > 0):
            # create metrics plot
            msk = perf_metrics['task'] == assay
            tmp = perf_metrics.loc[msk, ['model architecture', 'MT/ST', metric+' (mean)', metric+' (std)']]
            tmp = tmp.sort_values(by=['model architecture', 'MT/ST'])
            tmp['model architecture/MT/ST'] = tmp['model architecture'].astype(str) + ' ' + tmp['MT/ST'].astype(str)
            tmp = tmp.drop(columns=['model architecture', 'MT/ST'])
            tmp = tmp.reset_index(drop=True)
            symbol_width = 0.8
            for idx, row in tmp.iterrows():
                mean = row[metric + ' (mean)']
                std = row[metric + ' (std)']
                color = 'k' if row['model architecture/MT/ST'].split()[-1] == 'MT' else 'r'
                axs[i_ax, j_ax].hlines(mean, xmin=idx - symbol_width/2, xmax=idx + symbol_width/2, color=color, linewidth=1.0)
                axs[i_ax, j_ax].hlines(mean + std, xmin=idx - symbol_width/4, xmax=idx + symbol_width/4, color=color, linewidth=0.5)
                axs[i_ax, j_ax].hlines(mean - std, xmin=idx - symbol_width/4, xmax=idx + symbol_width/4, color=color, linewidth=0.5)
                axs[i_ax, j_ax].vlines(idx, ymin=mean - std, ymax=mean + std, color=color, linewidth=0.5)
            axs[i_ax, j_ax].spines['bottom'].set_visible(False)
            axs[i_ax, j_ax].spines['top'].set_visible(False)
            axs[i_ax, j_ax].spines['right'].set_visible(False)
            axs[i_ax, j_ax].spines['left'].set_position(('outward', 10))
            # .. set the common y limits for each assay
            msk = perf_metrics['task'] == assay
            tmp2 = perf_metrics.loc[msk, [metric+' (mean)', metric+' (std)']]
            tmp2 = tmp2.loc[tmp2[metric+' (mean)']>0.5]
            if tmp2.empty:
                continue
            y_min, y_max = ((math.floor((tmp2[metric+' (mean)'] - tmp2[metric+' (std)']).min()*100)/100), (math.ceil((tmp2[metric+' (mean)'] + tmp2[metric+' (std)']).max()*100)/100))
            axs[i_ax, j_ax].set_ylim(y_min, y_max)
            if i_ax < n_assays:
                axs[i_ax, j_ax].set_xticks([])
            else:
                axs[i_ax, j_ax].set_xticks(range(tmp.shape[0]))
                axs[i_ax, j_ax].set_xticklabels(tmp['model architecture/MT/ST'].values, rotation=90, fontsize=10)
                axs[i_ax, j_ax].tick_params(axis='x', bottom=False, labelbottom=True)
        # .. create metric titles
        if (i_ax == 0) & (j_ax >= 1):
            axs[i_ax, j_ax].text(0.5, 0.5, metric, fontsize=10, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", linewidth=0.5, facecolor='lightgrey'))
            axs[i_ax, j_ax].axis('off')
        # create assay titles
        if (j_ax == 0) & (i_ax >= 1):
            axs[i_ax, j_ax].text(0.5, 0.5, assay, fontsize=10, ha='center', va='center', rotation=90, bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", linewidth=0.5, facecolor='lightgrey'))
            axs[i_ax, j_ax].axis('off')
fig.tight_layout()
# fig.savefig(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\compare_predictive_performance_AMES_5_strains.png', dpi=600, bbox_inches='tight')
fig.savefig(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\compare_predictive_performance_AMES_agg_GM_CA_M.png', dpi=600, bbox_inches='tight')



# create structures for REACH registered substances
from cheminformatics.DSStox_structure_retrieval import retrieve_structure
reg_substances = pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\data/2024_12_16_reg_substances_dissemination_export.xlsx')
identifiers = reg_substances['Cas Number'].str.extractall(r'(\d{2,7}-\d{2}-\d)').squeeze().dropna().drop_duplicates().to_list()
ccte_data = retrieve_structure(identifiers)
reg_substances = reg_substances.merge(ccte_data[['identifier', 'smiles']], left_on='Cas Number', right_on='identifier', how='left')


# compare model predictions with external (hold-out) test set
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
tasks = [
     'in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test',
     'in vitro, in vitro gene mutation study in mammalian cells',
     'in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test',
      # 'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay'
]
# tasks = ['in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Escherichia coli (WP2 Uvr A), no',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Escherichia coli (WP2 Uvr A), yes',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Salmonella typhimurium (TA 100), no',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Salmonella typhimurium (TA 100), yes',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Salmonella typhimurium (TA 1535), no',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Salmonella typhimurium (TA 1535), yes',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Salmonella typhimurium (TA 1537), no',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Salmonella typhimurium (TA 1537), yes',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Salmonella typhimurium (TA 98), no',
#        'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay, Salmonella typhimurium (TA 98), yes']
for task in tasks:
    # predictions_path = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\GM/inference\leadscope/predictions.pickle'
    exp_data_training_path = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\training_eval_dataset\tabular\genotoxicity_dataset.xlsx'
    # exp_external_testset_path =r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\inference\leadscope\training_eval_dataset\tabular/genotoxicity_dataset.xlsx'
    predictions_path = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\ToxValDB/predictions.pickle'
    # exp_data_training_path = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_5_strains\training_eval_dataset\tabular\genotoxicity_dataset.xlsx'
    exp_external_testset_path =r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\ToxValDB\training_eval_dataset\tabular\genotoxicity_dataset.xlsx'
    # # .. load the predictions
    predictions = pd.read_pickle(predictions_path)
    msk = predictions['task'] == task
    predictions = predictions.loc[msk]
    if predictions.empty:
        continue
    # .. load the training set with positive and negative data
    exp_data_training = pd.read_excel(exp_data_training_path)
    msk = (exp_data_training['task'] == task) & exp_data_training['genotoxicity'].isin(['positive', 'negative'])
    exp_data_training = exp_data_training.loc[msk]
    # .. load the external test set with positive and negative data
    exp_data_external_testset = pd.read_excel(exp_external_testset_path)
    msk = (exp_data_external_testset['task'] ==task) & exp_data_external_testset['genotoxicity'].isin(['positive', 'negative'])
    exp_data_external_testset = exp_data_external_testset.loc[msk]
    if exp_data_external_testset.empty:
        continue
    # .. keep the predictions that are not in the training set
    msk = ~predictions['smiles'].isin(exp_data_training['smiles_std'].to_list())
    predictions = predictions.loc[msk]
    # .. compare predictions with experimental data
    metrics = []
    for model in predictions['model'].drop_duplicates().to_list():
        print('checking predictive performance with external (hold-out) set for model:', model)
        msk = predictions['model'] == model
        compare = (predictions.loc[msk, ['smiles', 'genotoxicity call', 'positive (probability)']].rename({'genotoxicity call': 'genotoxicity call (predicted)'}, axis='columns')
                   .merge(exp_data_external_testset[['smiles_std', 'genotoxicity']].rename({'genotoxicity': 'genotoxicity call (experimental)'}, axis='columns'),
                          left_on='smiles', right_on='smiles_std', how='inner'))
        compare[['genotoxicity call (predicted)', 'genotoxicity call (experimental)']].value_counts()
        # .. compute sensitivity, specificity, balanced accuracy and AUC with scikit learn
        sensitivity  = recall_score(compare['genotoxicity call (experimental)'], compare['genotoxicity call (predicted)'], pos_label='positive')
        specificity  = recall_score(compare['genotoxicity call (experimental)'], compare['genotoxicity call (predicted)'], pos_label='negative')
        balanced_accuracy = balanced_accuracy_score(compare['genotoxicity call (experimental)'], compare['genotoxicity call (predicted)'])
        roc_auc = roc_auc_score(compare['genotoxicity call (experimental)'].map({'positive': 1, 'negative': 0}), compare['positive (probability)'])
        metrics.append({'model': model, 'sensitivity': sensitivity, 'specificity': specificity, 'balanced accuracy': balanced_accuracy, 'roc auc': roc_auc})
    metrics = pd.DataFrame(metrics).set_index('model')
    metrics = pd.concat([metrics.mean().rename('mean'), metrics.std().rename('std')], axis='columns', sort=False)
    for row in metrics.iterrows():
        print(f'{task}, {row[0]}: {row[1]["mean"]:.3f} ± {row[1]["std"]:.3f}')
    print('number of datapoints: ', compare['smiles'].nunique())
    print('number of experimental positives: ', compare['genotoxicity call (experimental)'].value_counts()['positive'])




# compare classifiers using the external (hold-out) test set and the McNemar test
prediction_thresholds = {'positive': 0.5, 'negative': 0.5}
from mlxtend.evaluate import mcnemar_table, mcnemar
classifier_pairs =[
    {'model 1': 'FFNN MT GM', 'model 2': 'FFNN ST GM',
     'training set': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\training_eval_dataset\tabular\genotoxicity_dataset.xlsx',
     'external test set': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\ToxValDB\training_eval_dataset\tabular\genotoxicity_dataset.xlsx',
     'model 1 predictions':  r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\Ames_agg_GM_CA_MN\inference\ToxValDB/predictions.pickle',
     'model 2 predictions': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\GM\inference\ToxValDB/predictions.pickle',
     'task': 'in vitro, in vitro gene mutation study in mammalian cells'},
    {'model 1': 'FFNN MT CA', 'model 2': 'FFNN ST CA',
     'training set': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\training_eval_dataset\tabular\genotoxicity_dataset.xlsx',
     'external test set': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\ToxValDB\training_eval_dataset\tabular\genotoxicity_dataset.xlsx',
     'model 1 predictions': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\Ames_agg_GM_CA_MN\inference\ToxValDB/predictions.pickle',
     'model 2 predictions': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\CA\inference\ToxValDB/predictions.pickle',
     'task': 'in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test'},
    {'model 1': 'FFNN MT MN', 'model 2': 'FFNN ST MN',
     'training set': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\training_eval_dataset\tabular\genotoxicity_dataset.xlsx',
     'external test set': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\ToxValDB\training_eval_dataset\tabular\genotoxicity_dataset.xlsx',
     'model 1 predictions': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\Ames_agg_GM_CA_MN\inference\ToxValDB/predictions.pickle',
     'model 2 predictions': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\MN\inference\ToxValDB/predictions.pickle',
     'task': 'in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test'},
    {'model 1': 'FFNN MT Ames agg', 'model 2': 'FFNN ST Ames agg',
     'training set': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\training_eval_dataset\tabular\genotoxicity_dataset.xlsx',
     'external test set': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\bacterial_mutagenicity_issty_agg\training_eval_dataset\tabular\genotoxicity_dataset.xlsx',
     'model 1 predictions': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\Ames_agg_GM_CA_MN\inference\bacterial_mutagenicity_issty_agg/predictions.pickle',
     'model 2 predictions': r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\Ames_agg\inference\bacterial_mutagenicity_issty_agg/predictions.pickle',
     'task': 'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay'},
]
classifier_comparisons = []
for classifier_pair in classifier_pairs:
    task = classifier_pair['task']
    # .. load the training set with positive and negative data
    exp_data_training_path = classifier_pair['training set']
    exp_data_training = pd.read_excel(exp_data_training_path)
    msk = (exp_data_training['task'] == task) & exp_data_training['genotoxicity'].isin(['positive', 'negative'])
    exp_data_training = exp_data_training.loc[msk]
    # .. load the external test set with positive and negative data
    exp_external_testset_path = classifier_pair['external test set']
    exp_data_external_testset = pd.read_excel(exp_external_testset_path)
    msk = (exp_data_external_testset['task'] == task) & exp_data_external_testset['genotoxicity'].isin(['positive', 'negative'])
    exp_data_external_testset = exp_data_external_testset.loc[msk]
    # .. load the predictions for model 1 and keep the ones for structures not in the training set
    model_1_predictions_path = classifier_pair['model 1 predictions']
    model_1_predictions = pd.read_pickle(model_1_predictions_path)
    msk = model_1_predictions['task'] == task
    model_1_predictions = model_1_predictions.loc[msk]
    # .... keep the predictions that are not in the training set
    msk = ~model_1_predictions['smiles'].isin(exp_data_training['smiles_std'].to_list())
    model_1_predictions = model_1_predictions.loc[msk]
    # .... average the predictions for the same structure across the models (take the mean of probabilities)
    model_1_predictions = model_1_predictions.groupby(['smiles'])['positive (probability)'].mean().reset_index()
    model_1_predictions['genotoxicity call (predicted)'] = np.select([model_1_predictions['positive (probability)']>=prediction_thresholds['positive'], model_1_predictions['positive (probability)']<prediction_thresholds['negative']], ['positive', 'negative'], 'ambiguous')
    # .. load the predictions for model 2 and keep the ones for structures not in the training set
    model_2_predictions_path = classifier_pair['model 2 predictions']
    model_2_predictions = pd.read_pickle(model_2_predictions_path)
    msk = model_2_predictions['task'] == task
    model_2_predictions = model_2_predictions.loc[msk]
    # .... keep the predictions that are not in the training set
    msk = ~model_2_predictions['smiles'].isin(exp_data_training['smiles_std'].to_list())
    model_2_predictions = model_2_predictions.loc[msk]
    # .... average the predictions for the same structure across the models (take the mean of probabilities)
    model_2_predictions = model_2_predictions.groupby(['smiles'])['positive (probability)'].mean().reset_index()
    model_2_predictions['genotoxicity call (predicted)'] = np.select([model_2_predictions['positive (probability)']>=prediction_thresholds['positive'], model_2_predictions['positive (probability)']<prediction_thresholds['negative']], ['positive', 'negative'], 'ambiguous')
    # .. compare predictions with experimental data in the external (hold-out) test set
    compare_model_1 = (model_1_predictions[['smiles', 'genotoxicity call (predicted)', 'positive (probability)']]
               .merge(exp_data_external_testset[['smiles_std', 'genotoxicity']].rename({'genotoxicity': 'genotoxicity call (experimental)'}, axis='columns'),
                      left_on='smiles', right_on='smiles_std', how='inner'))
    sensitivity_model_1 = recall_score(compare_model_1['genotoxicity call (experimental)'], compare_model_1['genotoxicity call (predicted)'], pos_label='positive')
    specificity_model_1 = recall_score(compare_model_1['genotoxicity call (experimental)'], compare_model_1['genotoxicity call (predicted)'], pos_label='negative')
    balanced_accuracy_model_1 = balanced_accuracy_score(compare_model_1['genotoxicity call (experimental)'], compare_model_1['genotoxicity call (predicted)'])
    roc_auc_model_1 = roc_auc_score(compare_model_1['genotoxicity call (experimental)'].map({'positive': 1, 'negative': 0}), compare_model_1['positive (probability)'])
    compare_model_2 = (model_2_predictions[['smiles', 'genotoxicity call (predicted)', 'positive (probability)']]
               .merge(exp_data_external_testset[['smiles_std', 'genotoxicity']].rename({'genotoxicity': 'genotoxicity call (experimental)'}, axis='columns'),
                      left_on='smiles', right_on='smiles_std', how='inner'))
    sensitivity_model_2 = recall_score(compare_model_2['genotoxicity call (experimental)'], compare_model_2['genotoxicity call (predicted)'], pos_label='positive')
    specificity_model_2 = recall_score(compare_model_2['genotoxicity call (experimental)'], compare_model_2['genotoxicity call (predicted)'], pos_label='negative')
    balanced_accuracy_model_2 = balanced_accuracy_score(compare_model_2['genotoxicity call (experimental)'], compare_model_2['genotoxicity call (predicted)'])
    roc_auc_model_2 = roc_auc_score(compare_model_2['genotoxicity call (experimental)'].map({'positive': 1, 'negative': 0}), compare_model_2['positive (probability)'])
    tb = mcnemar_table(y_target=compare_model_1['genotoxicity call (experimental)'],
                       y_model1=compare_model_1['genotoxicity call (predicted)'],
                       y_model2=compare_model_2['genotoxicity call (predicted)'])
    _, p = mcnemar(ary=tb, exact=True)
    entry = classifier_pair
    print(f"{classifier_pair['model 1']} vs {classifier_pair['model 2']} -> McNemar exact p-value = {p:.3f}{'***' if p < 0.05 else ''}")
    print('McNemar table:\n', tb)
    entry['McNemar table'] = tb
    entry['p value'] = p
    classifier_comparisons.append(entry)
