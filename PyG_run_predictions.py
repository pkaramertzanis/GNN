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
import numpy as np

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

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# structures to run predictions for
# structures_smiles = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\training_eval_dataset\tabular/genotoxicity_dataset.xlsx'
# structures_smiles = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\bacterial_mutagenicity_issty_agg\training_eval_dataset\tabular/genotoxicity_dataset.xlsx'
structures_smiles = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_5_strains\inference\bacterial_mutagenicity_issty\training_eval_dataset\tabular/genotoxicity_dataset.xlsx'
structures = pd.read_excel(structures_smiles, usecols=['smiles_std']).dropna().drop_duplicates().rename({'smiles_std': 'smiles'}, axis='columns').reset_index() # this is a dataframe with a single smiles column
# structures = pd.DataFrame({'smiles': ['N#CO', 'CCC']})
# .. append the standardised smiles and mol objects, only valid structures that can be predicted will be kept
structures = pd.concat([structures['smiles'], pd.DataFrame(structures['smiles'].apply(lambda smiles: process_smiles(smiles)).to_list(), columns=['smiles_std', 'mol', 'processing_details'])], axis='columns', sort=False, ignore_index=False)
structures = structures.reset_index(drop=True).reset_index(drop=False).rename({'index': 'i mol'}, axis='columns')
mols = structures['mol'].dropna().drop_duplicates().to_list()


# model fit folder
output_path = Path(fr'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_5_strains')
stopping = 'early_stopping'

# load the node and edge features, feature values and tasks
with (open(output_path / 'feature_task_info.json', 'r') as f):
    all_features_tasks = json.load(f)
    NODE_FEATS, EDGE_FEATS, node_feature_names, edge_attr_names, tasks = (all_features_tasks['node features'],
                                                                          all_features_tasks['edge features'],
                                                                          all_features_tasks['node feature values'],
                                                                          all_features_tasks['edge feature values'],
                                                                          all_features_tasks['tasks'])

# run inference with all final fitted models (predictions)
model_paths = list((output_path).glob(fr'best_configuration_model_fit_{stopping}_*/model.pth'))
all_predictions = []
for i_model, model_path in enumerate(model_paths):
    log.info(f'generating predictions for model: {i_model}')
    # load the model
    net = torch.load(model_path, map_location=device)
    net.to(device)
    # run the predictions
    predictions = eval(mols, net, tasks, NODE_FEATS, EDGE_FEATS, node_feature_names, edge_attr_names, embeddings=False)
    predictions.insert(loc=0, column='model', value=str(model_path))
    all_predictions.append(predictions)
all_predictions = pd.concat(all_predictions, axis='index', sort=False, ignore_index=True)
all_predictions = structures.merge(all_predictions, on='i mol', how='inner')
all_predictions.to_pickle(fr'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_5_strains/inference/bacterial_mutagenicity_issty\predictions_{stopping}.pickle')

# run inference with all final fitted models (embeddings)
all_embeddings = []
for i_model, model_path in enumerate(model_paths):
    log.info(f'generating embeddings for model: {i_model}')
    # load the model
    net = torch.load(model_path, map_location =device)
    net.to(device)
    # run the predictions
    embeddings = eval(mols, net, tasks, NODE_FEATS, EDGE_FEATS, node_feature_names, edge_attr_names, embeddings=True)
    embeddings.insert(loc=0, column='model', value=str(model_path))
    all_embeddings.append(embeddings)
all_embeddings = pd.concat(all_embeddings, axis='index', sort=False, ignore_index=True)
all_embeddings = structures.merge(all_embeddings, on='i mol', how='inner')
all_embeddings.to_pickle(fr'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_5_strains/inference/bacterial_mutagenicity_issty\embeddings_{stopping}.pickle')




