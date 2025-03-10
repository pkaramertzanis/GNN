# setup logging
from optuna.trial import TrialState

import logger
import logging
log = logger.setup_applevel_logger(file_name ='logs/classical_ML_model_fit.log', level_stream=logging.DEBUG, level_file=logging.DEBUG)

# import and configure pandas globally
import pandas_config
import pandas as pd
import numpy as np

from pathlib import Path
import pickle
import json
from tqdm import tqdm

# cheminformatics
from cheminformatics.rdkit_toolkit import read_sdf
from rdkit.Chem import AllChem

# machine learning
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, balanced_accuracy_score

from data.combine import create_sdf


# set the model architecture
MODEL_NAME = 'RF'
STUDY_NAME = "Ames_agg" # name of the study in the Optuna sqlit database

# location to store the results
output_path = Path(rf'output/{MODEL_NAME}/{STUDY_NAME}')
output_path.mkdir(parents=True, exist_ok=True)

# set up the dataset
flat_datasets = [
                # r'data/Hansen_2009/tabular/Hansen_2009_genotoxicity.xlsx',
                # r'data/Leadscope/tabular/Leadscope_genotoxicity.xlsx',
                r'data/ECVAM_AmesNegative/tabular/ECVAM_Ames_negative_genotoxicity.xlsx',
                r'data/ECVAM_AmesPositive/tabular/ECVAM_Ames_positive_genotoxicity.xlsx',
                r'data/QSARChallengeProject/tabular/QSARChallengeProject.xlsx',
                r'data/QSARToolbox/tabular/QSARToolbox_genotoxicity.xlsx',
                r'data/REACH/tabular/REACH_genotoxicity.xlsx',
                # r'data/Baderna_2020/tabular/Baderna_2020_genotoxicity.xlsx',
]
task_specifications = [
        #  {'filters': {'assay': ['bacterial reverse mutation assay'], 'cell line/species': [
        #      'Escherichia coli (WP2 Uvr A)',
        #      # 'Salmonella typhimurium (TA 102)',
        #      # 'Salmonella typhimurium (TA 100)',
        #      #  'Salmonella typhimurium (TA 1535)',
        #       # 'Salmonella typhimurium (TA 98)',
        #      # 'Salmonella typhimurium (TA 1537)'
        #                                                                                   ], 'metabolic activation': [
        #       # 'yes',
        #       'no'
        # ]}, 'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay', 'cell line/species', 'metabolic activation']},

      {'filters': {'assay': ['bacterial reverse mutation assay']},
       'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

       # {'filters': {'assay': ['in vitro mammalian cell micronucleus test']},
       #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},
    # #
    #    {'filters': {'assay': ['in vitro mammalian chromosome aberration test']},
    #     'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

    # {'filters': {'assay': ['in vitro mammalian cell gene mutation test using the Hprt and xprt genes']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},
    #
    # {'filters': {'assay': ['in vitro mammalian cell gene mutation test using the thymidine kinase gene']},
    #  'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay']},

   # {'filters': {'endpoint': ['in vitro gene mutation study in mammalian cells']},
   #  'task aggregation columns': ['in vitro/in vivo', 'endpoint']},

]
# task_specifications = [
#      {'filters': {'assay': ['bacterial reverse mutation assay'], },
#       'task aggregation columns': ['in vitro/in vivo', 'endpoint']},
# ]
# task_specifications = [
#     {'filters': {'assay': ['bacterial reverse mutation assay'], 'cell line/species': [#'Escherichia coli (WP2 Uvr A)',
#                                                                                       #'Salmonella typhimurium (TA 102)',
#                                                                                       'Salmonella typhimurium (TA 100)',
#                                                                                       #'Salmonella typhimurium (TA 1535)',
#                                                                                       'Salmonella typhimurium (TA 98)',
#                                                                                       #'Salmonella typhimurium (TA 1537)'
#                                                                                       ], 'metabolic activation': ['yes', 'no']},
#      'task aggregation columns': ['in vitro/in vivo', 'endpoint', 'assay', 'cell line/species']},
#
# ]
training_eval_dataset_path_tabular = Path(output_path / 'training_eval_dataset/tabular')
training_eval_dataset_path_tabular.mkdir(parents=True, exist_ok=True)
training_eval_dataset_path_sdf = Path(output_path / 'training_eval_dataset/sdf')
training_eval_dataset_path_sdf.mkdir(parents=True, exist_ok=True)
outp_sdf = Path(training_eval_dataset_path_sdf/'genotoxicity_dataset.sdf')
outp_tab = Path(training_eval_dataset_path_tabular/'genotoxicity_dataset.xlsx')
tasks = create_sdf(flat_datasets = flat_datasets,
                   task_specifications = task_specifications,
                   outp_sdf = outp_sdf,
                   outp_tab = outp_tab)

# fingerprint parameters
fingerprint_parameters = {'radius': 2,
                          'fpSize': 2048,
                          'type': 'count' # 'binary', 'count'
                          }

# read in the molecular structures from the SDF file, and compute fingerprints
fpgen = AllChem.GetMorganGenerator(radius=fingerprint_parameters['radius'], fpSize=fingerprint_parameters['fpSize'],
                                   countSimulation=False, includeChirality=False)

mols = read_sdf(outp_sdf)
data = []
for i_mol, mol in tqdm(enumerate(mols)):
    tox_data = mol.GetProp('genotoxicity')
    tox_data = pd.DataFrame(json.loads(tox_data))
    # keep only positive and negative calls
    tox_data = tox_data.loc[tox_data['genotoxicity'].isin(['positive', 'negative'])]
    if not tox_data.empty:
        fg = fpgen.GetFingerprint(mol)
        fg_count = fpgen.GetCountFingerprint(mol)
        if fingerprint_parameters['type'] == 'binary':
            tox_data['fingerprint'] = ''.join([str(bit) for bit in fg.ToList()]) #fg.ToBitString()
        elif fingerprint_parameters['type'] == 'count':
            tox_data['fingerprint'] = ''.join([str(min(bit, 9)) for bit in fg_count.ToList()])
        else:
            ex = ValueError(f"unknown fingerprint type: {fingerprint_parameters}")
            log.error(ex)
            raise ex
        tox_data['molecule ID'] = i_mol
        data.append(tox_data)
data = pd.concat(data, axis='index', ignore_index=True, sort=False)

# features (fingerprints)
X = data.groupby('molecule ID')['fingerprint'].first()
X = np.array([[float(bit) for bit in list(fg)] for fg in X.to_list()], dtype=np.float32)
# genotoxicity outcomes for each task, make positive 1 and negative 0
task_outcomes = data.pivot(index='molecule ID',columns='task',values='genotoxicity')
y = task_outcomes.map(lambda genotoxicity: 1 if genotoxicity == 'positive' else 0).to_numpy().ravel()

# set the random number seed
RANDOM_SEED = 1

# Define a pipeline with variance threshold and random forest classifier
pipeline = Pipeline([
    ('var_thresh', VarianceThreshold()),  # remove low-variance features
    ('rf', RandomForestClassifier(random_state=RANDOM_SEED))
])

# Define hyperparameter grid for tuning
param_grid = {
    'var_thresh__threshold': [0.001, 0.01, 0.02, 0.05],
    'rf__n_estimators': [50, 100, 200, 300],
    'rf__max_depth': [None, 5, 10],
    'rf__min_samples_split': [2, 5],
    'rf__class_weight': ['balanced',  {0: 1, 1: 1}]
}

# Perform 5-fold cross-validation using balanced accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

grid_search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1, verbose=3, refit=True
)

# Fit the model
grid_search.fit(X, y)

# Display the best hyperparameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Balanced Accuracy:", grid_search.best_score_)

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_excel(output_path / 'cv_results.xlsx', index=False)


# repeat the model fitting with the best hyperparameters to also compute the sensitivity and specificity
model = pipeline.set_params(**grid_search.best_params_)
fold_metrics = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 0):
    # Split the dataset
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X[test_idx])
    # .. compute the metrics
    sensitivity = recall_score(y_test, y_pred_test)
    specificity = recall_score(y_test, y_pred_test, pos_label=0)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred_test)
    fold = {'fold': fold, 'sensitivity': sensitivity, 'specificity': specificity, 'balanced accuracy': balanced_accuracy}
    log.info(fold)
    fold_metrics.append(fold)
fold_metrics = pd.DataFrame(fold_metrics)
log.info(fold_metrics[['sensitivity', 'specificity', 'balanced accuracy']].apply(['mean', 'std'], axis='index'))
fold_metrics.to_excel(output_path / 'best_model_fold_metrics.xlsx', index=False)
