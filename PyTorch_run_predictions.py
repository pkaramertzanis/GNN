# setup logging
import logging
import logger

log = logger.setup_applevel_logger(file_name ='logs/Pytorch_model_run.log', level_stream=logging.DEBUG, level_file=logging.DEBUG)

import torch
import json
from pathlib import Path

from data.combine import process_smiles

import pandas as pd

from models.PyTorch_eval import eval

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
structures_smiles = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP_GNN\Ames_agg_GM_CA_MN\inference\ToxValDB\training_eval_dataset\tabular/genotoxicity_dataset.xlsx'
# structures_smiles = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\inference\morita_2019\training_eval_dataset\tabular/genotoxicity_dataset.xlsx'
# structures_smiles = r'D:\myApplications\local\2024_01_21_GCN_Muta\output\AttentiveFP\Ames_agg_GM_CA_MN\inference\leadscope\training_eval_dataset/tabular/genotoxicity_dataset.xlsx'
structures = pd.read_excel(structures_smiles, usecols=['smiles_std']).dropna().drop_duplicates().rename({'smiles_std': 'smiles'}, axis='columns').reset_index() # this is a dataframe with a single smiles column
# structures = pd.DataFrame({'smiles': ['N#CO']})
# .. append the standardised smiles and mol objects, only valid structures that can be predicted will be kept
structures = pd.concat([structures['smiles'], pd.DataFrame(structures['smiles'].apply(lambda smiles: process_smiles(smiles)).to_list(), columns=['smiles_std', 'mol', 'processing_details'])], axis='columns', sort=False, ignore_index=False)
structures = structures.reset_index(drop=True).reset_index(drop=False).rename({'index': 'i mol'}, axis='columns')
mols = structures['mol'].dropna().drop_duplicates().to_list()


# model fit folder
output_path = Path(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\MN')
# load the fingerprint parameters and task names
with open(output_path/'fingerprint_tasks.json', 'r') as f:
    fingerprint_tasks = json.load(f)
    fingerprint_parameters = fingerprint_tasks['fingerprint parameters']
    tasks = fingerprint_tasks['tasks']
# run predictions with all final fitted models
model_paths = list((output_path).glob(r'best_configuration_model_fit_*/model.pth'))
all_predictions = []
for i_model, model_path in enumerate(model_paths):
    log.info(f'running predictions for model: {i_model}')
    # load the model
    net = torch.load(model_path, map_location=device)
    # run the predictions
    predictions = eval(mols, net, tasks, fingerprint_parameters)
    predictions.insert(loc=0, column='model', value=str(model_path))
    all_predictions.append(predictions)
all_predictions = pd.concat(all_predictions, axis='index', sort=False, ignore_index=True)
all_predictions = structures.merge(all_predictions, on='i mol', how='inner')
all_predictions.to_pickle(r'D:\myApplications\local\2024_01_21_GCN_Muta\output\FFNN\MN\inference\ToxValDB\predictions.pickle')

