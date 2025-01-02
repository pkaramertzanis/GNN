'''
Analyse the predictions, the experimental data and existing alerts to identify
structural moeities not covered by existing mechanistic knowledge.

The analysis is based on the AttentiveFP 4-task model trained on the Ames, GM, CA and MN datasets.
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
matplotlib.use('Tkagg')
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


# UMAP of embeddings for the Ames_agg_GM_CA_MN AttentiveFP model
import matplotlib.pyplot as plt
import seaborn as sns
import umap
# datamap plot, https://github.com/TutteInstitute/datamapplot
output_path = Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN')
predictions = pd.read_pickle(output_path/'inference/training_set/training_set_predictions.pickle')
tasks = predictions['task'].unique().tolist()
embeddings = pd.read_pickle(output_path/'inference/training_set/training_set_embeddings.pickle')
exp_data = pd.read_excel(output_path/r'training_eval_dataset\tabular\genotoxicity_dataset.xlsx')
# .. concetenate the embeddings of the different models
embeddings['i model'] = embeddings['model'].str.extract(r'best_configuration_model_fit_(\d+)')
emb_cols = [col for col in embeddings.columns if col.startswith('embedding')]
embeddings = embeddings.pivot(index=['i mol', 'smiles', 'smiles_std'], columns=['i model'], values=emb_cols)
# combine the two column levels
embeddings.columns = [f'{col[0]} (model {col[1]})' for col in embeddings.columns]
embeddings = embeddings.reset_index()
# .. combine the model predictions (take the probability mean from all model fits)
predictions['i model'] = predictions['model'].str.extract(r'best_configuration_model_fit_(\d+)')
predictions = predictions.pivot_table(index=['i mol', 'smiles', 'smiles_std'], columns=['task'], values=['positive (probability)'], aggfunc='mean')
predictions.columns = [col[1] for col in predictions.columns]
predictions = predictions.reset_index()
# .. umap clustering
reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
emb_cols = [col for col in embeddings.columns if col.startswith('embedding')]
# .. scale the embeddings to z-scores (number of standard deviations from the mean)
embeddings[emb_cols] = StandardScaler().fit_transform(embeddings[emb_cols])
reducer.fit(embeddings[emb_cols])
reduced_embeddings = reducer.transform(embeddings[emb_cols])
reduced_embeddings = pd.DataFrame(reduced_embeddings, columns=['dim 1', 'dim 2'])
reduced_embeddings = pd.concat([embeddings[['i mol', 'smiles', 'smiles_std']], reduced_embeddings], axis='columns', sort=False, ignore_index=False)
# check which structures are epoxides
# from rdkit import Chem
# smarts = r'[$([CX4]1[OX2][CX4]1),$([#6]1=[#6][#8]1)]'
# structures['epoxide'] = structures['mol'].apply(lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)))
fig = plt.figure(figsize=(8, 8))
axs = fig.subplots(2, 2)
axs = axs.flatten()
for i_task, task in enumerate(tasks):
    # positive predictions
    msk = predictions[task] >= 0.7
    i_mols = predictions.loc[msk, 'i mol'].to_list()
    msk = reduced_embeddings['i mol'].isin(i_mols)
    pred_pos = axs[i_task].scatter(
        reduced_embeddings.loc[msk]['dim 1'],
        reduced_embeddings.loc[msk]['dim 2'],
        marker='o',
        facecolor=[sns.color_palette()[1]],
        edgecolor=[sns.color_palette()[1]],
        alpha=0.25,
        s=8,
        zorder=1)
    # negative predictions
    msk = predictions[task] <= 0.3
    i_mols = predictions.loc[msk, 'i mol'].to_list()
    msk = reduced_embeddings['i mol'].isin(i_mols)
    pred_neg = axs[i_task].scatter(
        reduced_embeddings.loc[msk]['dim 1'],
        reduced_embeddings.loc[msk]['dim 2'],
        marker='o',
        facecolor=[sns.color_palette()[0]],
        edgecolor=[sns.color_palette()[0]],
        alpha=0.25,
        s=8,
        zorder=1)
    # equivocal predictions
    msk = (predictions[task] > 0.3) & (predictions[task] < 0.7)
    i_mols = predictions.loc[msk, 'i mol'].to_list()
    msk = reduced_embeddings['i mol'].isin(i_mols)
    pred_equiv = axs[i_task].scatter(
        reduced_embeddings.loc[msk]['dim 1'],
        reduced_embeddings.loc[msk]['dim 2'],
        marker='o',
        facecolor='#babbbb',
        edgecolor='#babbbb',
        alpha=0.25,
        s=8,
        zorder=1)
    title = {'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay': 'Ames',
             'in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test' : 'MN',
             'in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test' : 'CA',
             'in vitro, in vitro gene mutation study in mammalian cells': 'GM'}.get(task)
    axs[i_task].set_title(title, fontsize=10)
    msk_pos = (exp_data['task'] == task) & (exp_data['genotoxicity'].isin(['positive']))
    msk_neg = (exp_data['task'] == task) & (exp_data['genotoxicity'].isin(['negative']))
    positives = reduced_embeddings.merge(exp_data.loc[msk_pos, ['smiles_std', 'genotoxicity']], left_on='smiles', right_on='smiles_std', how='inner')
    negatives = reduced_embeddings.merge(exp_data.loc[msk_neg, ['smiles_std', 'genotoxicity']], left_on='smiles', right_on='smiles_std', how='inner')
    exp_pos =  axs[i_task].scatter(
        positives['dim 1'],
        positives['dim 2'],
        marker='o',
        alpha=1.,
        facecolor='r',
        edgecolor='r',
        linewidth=0.75,
        s=0.5,
        c='r',
        zorder=2)
    exp_neg = axs[i_task].scatter(
        negatives['dim 1'],
        negatives['dim 2'],
        marker='o',
        alpha=0.75,
        facecolor='k',
        edgecolor='k',
        linewidth=0.5,
        s=0.5,
        c='k',
        zorder=3)
    axs[i_task].axis('off')
    axs[i_task].set_aspect('equal')
    # add the legend to the middle of the figure
    if i_task == 0:
        legend_handles = [pred_pos, pred_neg, pred_equiv, exp_pos, exp_neg]
        legend_labels = ['predicted positive', 'predicted negative', 'predicted equivocal', 'exp. positive', 'exp. negative']
        axs[i_task].legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(1.15, -.25), fontsize=8, frameon=False, markerscale=3)
fig.savefig(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\\umap_AMES_agg_GM_CA_MN.png', dpi=600, bbox_inches='tight')


# put together predictions for AMes_agg_GM_CA_MN, experimental data, Grace's alerts, Derek alerts, QSAR Toolbox profilers
from rdkit import Chem
exp_data_training_path = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\training_eval_dataset\tabular\genotoxicity_dataset.xlsx'
exp_data_training = pd.read_excel(exp_data_training_path)
exp_data_training = exp_data_training.pivot(index='smiles_std', columns='task', values='genotoxicity').reset_index()
exp_data_training = exp_data_training.rename({'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay': 'Ames (experimental)',
                                  'in vitro, in vitro gene mutation study in mammalian cells': 'GM (experimental)',
                                  'in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test': 'CA (experimental)',
                                  'in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test': 'MN (experimental)'}, axis='columns')
res = exp_data_training.copy()
# .. add the predictions
predictions_path = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\inference\training_set\training_set_predictions.pickle'
predictions = pd.read_pickle(predictions_path)
predictions = predictions.pivot_table(index=['smiles'], columns='task', values='positive (probability)', aggfunc='mean').reset_index()
predictions = predictions.rename({'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay': 'Ames (positive probability)',
                                  'in vitro, in vitro gene mutation study in mammalian cells': 'GM (positive probability)',
                                  'in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test': 'CA (positive probability)',
                                  'in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test': 'MN (positive probability)'}, axis='columns')
res = res.merge(predictions.rename({'smiles': 'smiles_std'}, axis='columns'), on='smiles_std', how='inner')
# .. add the reduced embeddings
res = res.merge(reduced_embeddings.drop(['smiles_std', 'i mol'], axis='columns').rename({'smiles': 'smiles_std'}, axis='columns'), on='smiles_std', how='inner')
# .. apply Grace's alerts
alerts_results = exp_data_training[['smiles_std']].copy()
alerts_results['mol'] = alerts_results['smiles_std'].apply(Chem.MolFromSmiles)
alerts = pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\data/genetox_structural_alerts_201021.xlsx')
alerts['mol'] = alerts['SMARTS'].apply(Chem.MolFromSmarts)
for idx, row in alerts.iterrows():
    log.info(f'applying structural alert: {row["Name"]}')
    msk = alerts_results['mol'].apply(lambda mol: mol.HasSubstructMatch(row['mol']))
    alerts_results[f"structural alert {row['Name']}"] = msk
res = res.merge(alerts_results.drop('mol', axis='columns'), on='smiles_std', how='inner')
# .. add Derek alerts
derek_predictions = pd.read_parquet(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\inference\alerts_profilers/derek_predictions.parquet')
derek_predictions = derek_predictions.drop('components_tautomers', axis='columns').explode(column='Derek predictions')
derek_predictions= derek_predictions.rename({'prediction status': 'Derek prediction status'}, axis='columns')
derek_predictions = pd.concat([derek_predictions.drop('Derek predictions', axis='columns').reset_index(drop=True), pd.json_normalize(derek_predictions['Derek predictions']).reset_index(drop=True)], axis='columns', sort=False, ignore_index=False)
msk = derek_predictions['endpoint group'].str.contains('(?i)genotoxicity|mutagenicity|carcinogenicity', case=False, na=False, regex=True)
tmp = derek_predictions.loc[msk]
tmp['alerts'] = tmp['alerts'].apply(lambda alerts: ';'.join(sorted(set(alerts))) if isinstance(alerts, np.ndarray) else None)
tmp = tmp.pivot(index='smiles_std',columns=['species', 'endpoint group', 'endpoint'], values=['likelihood', 'alerts'])
tmp.columns = [f'Derek {col[0]} species:{col[1]} endpoint group: {col[2]}, endpoint: {col[3]}' for col in tmp.columns]
tmp = tmp.reset_index()
derek_predictions = derek_predictions[['smiles_std', 'Derek prediction status']].drop_duplicates().merge(tmp, on='smiles_std', how='left')
res = res.merge(derek_predictions, on='smiles_std', how='inner')
# .. add Toolbox alerts
toolbox_profilers =  pd.read_parquet(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\inference\alerts_profilers/profilers.parquet')
toolbox_predictions = pd.read_parquet(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\inference\alerts_profilers/profiling_results.parquet')
toolbox_predictions = toolbox_predictions.explode(column='alerts')
# .. set the QSAR Toolbox prediction status, this is a singe value per structure and is set to failed if any profiler failed
msk = (toolbox_predictions['chemId']=='not available') | toolbox_predictions['alerts'].isin(['Undefined', 'PROFILING TIMEOUT', 'ERROR!', '(N/A)'])
prediction_status = toolbox_predictions[['smiles_std']].drop_duplicates()
prediction_status['QSAR Toolbox prediction status'] = prediction_status['smiles_std'].apply(lambda smiles: 'failed' if smiles in toolbox_predictions.loc[msk, 'smiles_std'].to_list() else 'succeeded')
toolbox_predictions = toolbox_predictions.merge(toolbox_profilers[['Guid', 'Caption']].drop_duplicates().dropna().rename({'Guid': 'profilerGuid', 'Caption': 'profiler'}, axis='columns'), on='profilerGuid', how='left')
toolbox_predictions = toolbox_predictions.pivot_table(index=['smiles_std'], columns='profiler', values='alerts', aggfunc=lambda alerts: ';'.join(sorted(set(alerts.dropna()))))
toolbox_predictions.columns = [f'QSAR Toolbox profiler: {col}' for col in toolbox_predictions.columns]
toolbox_predictions = toolbox_predictions.reset_index()
toolbox_predictions = prediction_status.merge(toolbox_predictions, on='smiles_std', how='left')
res = res.merge(toolbox_predictions, on='smiles_std', how='inner')
# add the toxprints
toxprints = pd.read_csv(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\training_eval_dataset\tabular/toxprint_v2_vs_genotoxicity_dataset.tsv', sep='\t')
toxprints = toxprints.dropna(how='any').reset_index(drop=True)
toxprints = toxprints.drop('M_STRUCTURE_WARNING', axis='columns')
toxprints = toxprints.rename(lambda col: 'toxprint: '+col, axis='columns')
res = pd.concat([res, toxprints], axis='columns', sort=False, ignore_index=False)
res = res.copy() #defragment the dataframe
res.to_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\structural_alerts_coverage_AMES_agg_GM_CA_MN.xlsx', index=False)
# res['Ames (prediction)'] = np.select([res['Ames (positive probability)']>=0.7, res['Ames (positive probability)']<=0.3], ['positive', 'negative'], 'ambiguous')
# res.groupby(['Ames (experimental)', 'Ames (prediction)', 'Derek likelihood species:bacterium endpoint group: Genotoxicity (ALL)/Mutagenicity (ALL), endpoint: Mutagenicity in vitro']).size().sort_values(ascending=False).reset_index()


# verify the locally calculated toxprints with the development server of the dashboard
import requests
url = r'https://hazard-dev.sciencedataexperts.com/api/descriptors'
params = {'type': 'toxprints', 'smiles': res['smiles_std'].iloc[0], 'headers': True}
response = requests.get(url, params=params)
data = response.json()
toxprint = np.array(data['chemicals'][0]['descriptors']).astype(int)
assert np.all(np.equal(res[[col for col in res.columns if col.startswith('toxprint')]].loc[0].values.astype(int), toxprint))


# visualise the clusters (Derek bacterial mutation)
import datamapplot
structural_alerts = pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\structural_alerts_coverage_AMES_agg_GM_CA_MN.xlsx')
# keep only plausible and probable with one alert and large clusters
col_likelihood = 'Derek likelihood species:bacterium endpoint group: Genotoxicity (ALL)/Mutagenicity (ALL), endpoint: Mutagenicity in vitro'
col_alerts = 'Derek alerts species:bacterium endpoint group: Genotoxicity (ALL)/Mutagenicity (ALL), endpoint: Mutagenicity in vitro'
msk = (structural_alerts[col_likelihood].isin(['PLAUSIBLE', 'PROBABLE'])) & ~(structural_alerts[col_alerts].str.contains(';', na=True))
large_clusters = (structural_alerts.loc[msk, col_alerts].value_counts() > 10).to_frame().query('`count`').index.to_list()
msk2 = structural_alerts[col_alerts].isin(large_clusters)
labels = np.where(msk & msk2, structural_alerts[col_alerts], 'Unlabelled')
structural_alerts_data_map = structural_alerts[['dim 1', 'dim 2']].values.astype('float32') # float32 needed for numba otherwise it will raise an error
datamapplot.create_plot(structural_alerts_data_map, labels, use_medoids=True,  label_font_size=14)


# visualise the clusters (QSAR Toolbox profiler: DNA alerts for AMES, CA and MNT by OASIS)
import datamapplot
import colorcet
from matplotlib.colors import rgb2hex
structural_alerts = pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\structural_alerts_coverage_AMES_agg_GM_CA_MN.xlsx')
# keep only plausible and probable with one alert and large clusters
col_alert = 'QSAR Toolbox profiler: DNA alerts for AMES, CA and MNT by OASIS'
# keep only the level three alerts
def extract_level_three_alert(alerts):
    p = r'[^>]+>>[^>]+>>([^>]+)'
    level_three_alerts = set()
    for alert in alerts.split(';'):
        if re.match(p, alert):
            level_three_alerts.add((re.findall(p, alert)[0]).strip())
    return list(level_three_alerts)
structural_alerts[col_alert+' (processed)'] = structural_alerts[col_alert].apply(lambda alerts: extract_level_three_alert(alerts) if pd.notnull(alerts) else [])
msk = structural_alerts[col_alert+' (processed)'].apply(lambda alerts: len(alerts) == 1)
large_clusters = (structural_alerts.loc[msk, col_alert+' (processed)'].value_counts() > 15).to_frame().query('`count`').index.to_list()
msk2 = structural_alerts[col_alert+' (processed)'].isin(large_clusters)
labels = np.where(msk & msk2, structural_alerts[col_alert+' (processed)'].apply(lambda x: x[0] if x else None), 'Unlabelled')
structural_alerts_data_map = structural_alerts[['dim 1', 'dim 2']].values.astype('float32') # float32 needed for numba otherwise it will raise an error
for darkmode in [True, False]:
    fig, ax = datamapplot.create_plot(structural_alerts_data_map, labels, use_medoids=True,
                                      label_over_points=False,
                                      dynamic_label_size=True,
                                      dynamic_label_size_scaling_factor=0.5,
                                      max_font_size=14,
                                      min_font_size=10,
                                      point_size=8,
                                      darkmode=darkmode,
                                      force_matplotlib=True,
                                      cmap=colorcet.cm.colorwheel
                                      )
    # remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if darkmode:
        fig.savefig(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\DNA_alerts_AMES_CA_MNT_OASIS_overlay_AMES_CA_MNT_OASIS_embeddings_dark.png', dpi=600, bbox_inches='tight')
    else:
        fig.savefig(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\DNA_alerts_AMES_CA_MNT_OASIS_overlay_AMES_CA_MNT_OASIS_embeddings_light.png', dpi=600, bbox_inches='tight')
