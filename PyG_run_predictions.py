# setup logging
import logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/GNN_model_run.log', level_stream=logging.DEBUG, level_file=logging.DEBUG)

import torch
import json
from pathlib import Path

from data.combine import process_smiles
from models.PyG_eval import eval

import pandas as pd

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

# structures to run predictions for
structures_smiles = r'D:\myApplications\local\2024_01_21_GCN_Muta\data\combined\tabular\genotoxicity_dataset.xlsx'
structures = pd.read_excel(structures_smiles, usecols=['smiles_std']).dropna().drop_duplicates().rename({'smiles_std': 'smiles'}, axis='columns').reset_index() # this is a dataframe with a single smiles column
# structures = pd.DataFrame({'smiles': ['N#CO']})
# .. append the standardised smiles and mol objects, only valid structures that can be predicted will be kept
structures = pd.concat([structures['smiles'], pd.DataFrame(structures['smiles'].apply(lambda smiles: process_smiles(smiles)).to_list(), columns=['smiles_std', 'mol', 'processing_details'])], axis='columns', sort=False, ignore_index=False)
structures = structures.reset_index(drop=True).reset_index(drop=False).rename({'index': 'i mol'}, axis='columns')
mols = structures['mol'].dropna().drop_duplicates().to_list()


# model fit folder
output_path = Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\develop\AttentiveFP_GNN')

# load the node and edge features, feature values and tasks
with (open(output_path / 'feature_task_info.json', 'r') as f):
    all_features_tasks = json.load(f)
    NODE_FEATS, EDGE_FEATS, node_feature_names, edge_attr_names, tasks = (all_features_tasks['node features'],
                                                                          all_features_tasks['edge features'],
                                                                          all_features_tasks['node feature values'],
                                                                          all_features_tasks['edge feature values'],
                                                                          all_features_tasks['tasks'])

# run predictions with all final fitted models
model_paths = list((output_path).glob(r'best_configuration_model_fit_*/model.pth'))
all_predictions = []
for i_model, model_path in enumerate(model_paths):
    log.info(f'running predictions for model: {i_model}')
    # load the model
    net = torch.load(model_path)
    # run the predictions
    predictions = eval(mols, net, tasks, NODE_FEATS, EDGE_FEATS, node_feature_names, edge_attr_names)
    predictions.insert(loc=0, column='model', value=str(model_path))
    all_predictions.append(predictions)
all_predictions = pd.concat(all_predictions, axis='index', sort=False, ignore_index=True)
all_predictions = structures.merge(all_predictions, on='i mol', how='inner')

all_predictions.groupby(['i mol', 'task'])['positive (probality)'].agg(['min', 'max']).reset_index()



# ------ compare with the experimental data used for the training ------
tmp = all_predictions.loc[all_predictions['model']==r'D:\myApplications\local\2024_01_21_GCN_Muta\output\develop\AttentiveFP_GNN\best_configuration_model_fit_0\model.pth']
training_test_set = pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\data\combined\tabular\genotoxicity_dataset.xlsx')
training_test_set = training_test_set[['smiles_std', 'task', 'genotoxicity']].rename({'genotoxicity': 'genotoxicity (experimental)', 'smiles_std': 'smiles'}, axis='columns')
task = 'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay'
msk = (training_test_set['genotoxicity (experimental)'].isin(['positive', 'negative']))
training_test_set = training_test_set.loc[msk]
compare = training_test_set.merge(tmp, left_on=['smiles', 'task'], right_on=['smiles', 'task'], how='inner')
accuracy = compare.groupby('task').apply(lambda x: sum(x['genotoxicity (experimental)']==x['genotoxicity call'])/len(x), include_groups=False).reindex(tasks)







# delete after this point
msk = (training_test_set['task']=='in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay') & (training_test_set['genotoxicity (experimental)'].isin(['positive', 'negative'])
x1 = training_test_set.loc[training_test_set['task']=='in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay']
set(x1['smiles_std'].drop_duplicates()) - set(all_predictions['smiles_std'].drop_duplicates())


# ------ compare with the ECVAM databases (only new structures) ------
# smiles of structures to run predictions for
datasets = [r'D:\myApplications\local\2024_01_21_GCN_Muta\data\ECVAM_AmesPositive\tabular\ECVAM_Ames_positive_genotoxicity.xlsx',
                  r'D:\myApplications\local\2024_01_21_GCN_Muta\data\ECVAM_AmesNegative\tabular\ECVAM_Ames_negative_genotoxicity.xlsx']
exp_data = []
for dataset in datasets:
    exp_data.append(pd.read_excel(dataset, usecols=['smiles', 'endpoint', 'assay', 'cell line/species', 'metabolic activation', 'genotoxicity']).dropna().drop_duplicates().rename({'smiles_std': 'smiles'}, axis='columns').reset_index())
exp_data = pd.concat(exp_data, axis='index', sort=False, ignore_index=True)
exp_data = exp_data.rename({'genotoxicity': 'genotoxicity (experimental)'}, axis='columns')

# append the standardised smiles and mol objects
exp_data = pd.concat([exp_data, pd.DataFrame(exp_data['smiles'].apply(lambda smiles: process_smiles(smiles)).to_list(), columns=['smiles_std', 'mol', 'processing_details'])], axis='columns', sort=False, ignore_index=False)
# eliminate structures that could not be standardised
msk = exp_data['mol'].notnull()
exp_data = exp_data.loc[msk].reset_index(drop=True).reset_index(drop=False).rename({'index': 'i mol'}, axis='columns')

# run the predictions
predictions = eval(exp_data['mol'].to_list(), net, tasks, NODE_FEATS, EDGE_FEATS, node_feature_names, edge_attr_names)
predictions = exp_data[['i mol', 'smiles_std']].merge(predictions, on='i mol', how='inner')

# remove the predictions that can be found in the experimental dataset used to train the model
tmp = predictions.merge(training_test_set[['smiles_std', 'task']], on=['smiles_std', 'task'], how='left', indicator=True)
predictions_new = tmp[tmp['_merge']=='left_only'].drop(['_merge', 'i mol'], axis='columns')
predictions_new = predictions_new.rename({'genotoxicity call': 'genotoxicity (predicted)'}, axis='columns')

# compare predictions with the experimental data, task in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test
task = 'in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test'
msk1 = predictions_new['task'] == task
msk2 = exp_data['endpoint'] == 'in vitro micronucleus study'
compare = predictions_new.loc[msk1].merge(exp_data.loc[msk2], on=['smiles_std'], how='inner')
accuracy = ((compare['genotoxicity (predicted)']==compare['genotoxicity (experimental)']) & compare['genotoxicity (experimental)'].isin(['positive', 'negative'])).sum()/compare['genotoxicity (experimental)'].isin(['positive', 'negative']).sum()
print(f"model accuracy for task {task}: {accuracy: .3f}, ({compare['genotoxicity (experimental)'].isin(['positive', 'negative']).sum()} datapoints)")

# compare predictions with the experimental data, task in vitro, in vitro gene mutation study in mammalian cells
task = 'in vitro, in vitro gene mutation study in mammalian cells'
msk1 = predictions_new['task'] == task
msk2 = exp_data['endpoint'] == 'in vitro gene mutation study in mammalian cells'
compare = predictions_new.loc[msk1].merge(exp_data.loc[msk2], on=['smiles_std'], how='inner')
accuracy = ((compare['genotoxicity (predicted)']==compare['genotoxicity (experimental)']) & compare['genotoxicity (experimental)'].isin(['positive', 'negative'])).sum()/compare['genotoxicity (experimental)'].isin(['positive', 'negative']).sum()
print(f"model accuracy for task {task}: {accuracy: .3f}, ({compare['genotoxicity (experimental)'].isin(['positive', 'negative']).sum()} datapoints)")

# compare predictions with the experimental data, task in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test
task = 'in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test'
msk1 = predictions_new['task'] == task
msk2 = exp_data['endpoint'] == 'in vitro chromosome aberration study in mammalian cells'
compare = predictions_new.loc[msk1].merge(exp_data.loc[msk2], on=['smiles_std'], how='inner')
accuracy = ((compare['genotoxicity (predicted)']==compare['genotoxicity (experimental)']) & compare['genotoxicity (experimental)'].isin(['positive', 'negative'])).sum()/compare['genotoxicity (experimental)'].isin(['positive', 'negative']).sum()
print(f"model accuracy for task {task}: {accuracy: .3f}, ({compare['genotoxicity (experimental)'].isin(['positive', 'negative']).sum()} datapoints)")

# compare predictions with the experimental data, task in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay
task = 'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay'
msk1 = predictions_new['task'].str.contains('in vitro gene mutation study in bacteria')
msk2 = exp_data['endpoint'] == 'in vitro gene mutation study in bacteria'
tmp = exp_data.loc[msk2].groupby('smiles_std')['genotoxicity (experimental)'].apply(lambda calls: 'positive' if 'positive' in calls.to_list() else 'negative').reset_index()
compare = predictions_new.loc[msk1].merge(tmp, on=['smiles_std'], how='inner')
accuracy = ((compare['genotoxicity (predicted)']==compare['genotoxicity (experimental)']) & compare['genotoxicity (experimental)'].isin(['positive', 'negative'])).sum()/compare['genotoxicity (experimental)'].isin(['positive', 'negative']).sum()
print(f"model accuracy for task {task}: {accuracy: .3f}, ({compare['genotoxicity (experimental)'].isin(['positive', 'negative']).sum()} datapoints)")



# .. concordance of experimental data
# compare experimental data, task in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test
task = 'in vitro, in vitro micronucleus study, in vitro mammalian cell micronucleus test'
msk1 = training_test_set['task'] == task
msk2 = exp_data['endpoint'] == 'in vitro micronucleus study'
compare = (training_test_set.loc[msk1].rename({'genotoxicity (experimental)': 'genotoxicity (experimental, train/test set)'}, axis='columns')
           .merge(exp_data.loc[msk2].rename({'genotoxicity (experimental)': 'genotoxicity (experimental, ECVAM)'}, axis='columns'), on=['smiles_std'], how='inner'))
accuracy = ((compare['genotoxicity (experimental, train/test set)']==compare['genotoxicity (experimental, ECVAM)']) & compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative'])).sum()/compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative']).sum()
print(f"experimental concordance (accuracy) for task {task}: {accuracy: .3f}, ({compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative']).sum()} datapoints)")

# compare experimental data, task in vitro, in vitro gene mutation study in mammalian cells
task = 'in vitro, in vitro gene mutation study in mammalian cells'
msk1 = training_test_set['task'] == task
msk2 = exp_data['endpoint'] == 'in vitro gene mutation study in mammalian cells'
compare = (training_test_set.loc[msk1].rename({'genotoxicity (experimental)': 'genotoxicity (experimental, train/test set)'}, axis='columns')
           .merge(exp_data.loc[msk2].rename({'genotoxicity (experimental)': 'genotoxicity (experimental, ECVAM)'}, axis='columns'), on=['smiles_std'], how='inner'))
accuracy = ((compare['genotoxicity (experimental, train/test set)']==compare['genotoxicity (experimental, ECVAM)']) & compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative'])).sum()/compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative']).sum()
print(f"experimental concordance (accuracy) for task {task}: {accuracy: .3f}, ({compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative']).sum()} datapoints)")

# compare experimental data, task in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test
task = 'in vitro, in vitro chromosome aberration study in mammalian cells, in vitro mammalian chromosome aberration test'
msk1 = training_test_set['task'] == task
msk2 = exp_data['endpoint'] == 'in vitro chromosome aberration study in mammalian cells'
compare = (training_test_set.loc[msk1].rename({'genotoxicity (experimental)': 'genotoxicity (experimental, train/test set)'}, axis='columns')
           .merge(exp_data.loc[msk2].rename({'genotoxicity (experimental)': 'genotoxicity (experimental, ECVAM)'}, axis='columns'), on=['smiles_std'], how='inner'))
accuracy = ((compare['genotoxicity (experimental, train/test set)']==compare['genotoxicity (experimental, ECVAM)']) & compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative'])).sum()/compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative']).sum()
print(f"experimental concordance (accuracy) for task {task}: {accuracy: .3f}, ({compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative']).sum()} datapoints)")

# compare experimental data, task in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay
task = 'in vitro, in vitro gene mutation study in bacteria, bacterial reverse mutation assay'
msk1 = training_test_set['task'].str.contains('in vitro gene mutation study in bacteria')
msk2 = exp_data['endpoint'] == 'in vitro gene mutation study in bacteria'
tmp1 = training_test_set.loc[msk1].groupby('smiles_std')['genotoxicity (experimental)'].apply(lambda calls: 'positive' if 'positive' in calls.to_list() else 'negative').reset_index()
tmp2 = exp_data.loc[msk2].groupby('smiles_std')['genotoxicity (experimental)'].apply(lambda calls: 'positive' if 'positive' in calls.to_list() else 'negative').reset_index()
compare = (tmp1.rename({'genotoxicity (experimental)': 'genotoxicity (experimental, train/test set)'}, axis='columns')
           .merge(tmp2.rename({'genotoxicity (experimental)': 'genotoxicity (experimental, ECVAM)'}, axis='columns'), on=['smiles_std'], how='inner'))
accuracy = ((compare['genotoxicity (experimental, train/test set)']==compare['genotoxicity (experimental, ECVAM)']) & compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative'])).sum()/compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative']).sum()
print(f"experimental concordance (accuracy) for task {task}: {accuracy: .3f}, ({compare['genotoxicity (experimental, ECVAM)'].isin(['positive', 'negative']).sum()} datapoints)")


# delete after this point
check = compare.merge(exp_data[['smiles_std', 'smiles']].drop_duplicates(), on='smiles_std')
check.to_excel(r'junk/check.xlsx')
check = check.merge(pd.read_excel(r'D:\myApplications\local\2024_01_21_GCN_Muta\data\combined\tabular\genotoxicity_dataset.xlsx'), on='smiles_std', how='inner')
check['Toolbox'] = check['source record ID'].str.contains('QSAR Toolbox')
check['REACH'] = check['source record ID'].str.contains('REACH')
check.groupby(['Toolbox', 'REACH'])['smiles_std'].nunique()