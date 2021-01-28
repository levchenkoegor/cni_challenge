from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy import linalg
from scipy.sparse import csgraph
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# define paths
project_root = Path()
data_dir = project_root / 'inputdir'
data_dir_train = data_dir / '2019_CNI_TrainingRelease-master'
data_dir_test = data_dir / '2019_CNI_ValidationRelease-master'

# load data
train_paths = sorted(list(data_dir_train.glob('Training/*/timeseries_*.csv')))
train_tss = [pd.read_csv(path) for path in train_paths]
train_phen = pd.read_csv(data_dir_train / 'SupportingInfo' / 'phenotypic_training.csv').replace(
    {"ADHD": 1, "Control": 0, "F": 0, "M": 1})

test_paths = sorted(list(data_dir_test.glob('Validation/*/timeseries_*.csv')))
test_tss = [pd.read_csv(path) for path in test_paths]
test_phen = pd.read_csv(data_dir_test / 'SupportingInfo' / 'phenotypic_validation.csv').replace(
    {"ADHD": 1, "Control": 0, "F": 0, "M": 1})

# calculate corr mat for train and test
train_corr_mats = [df.T.corr() for df in train_tss]
test_corr_mats = [df.T.corr() for df in test_tss]

# replace negative values to zero
train_corr_mats_pos = [df.clip(lower=0) for df in train_corr_mats]
test_corr_mats_pos = [df.clip(lower=0) for df in test_corr_mats]

# laplacian and eigenavalues
train_laplac_mats = [csgraph.laplacian(corr_mat.values, normed=True) for corr_mat in train_corr_mats_pos]
train_eigs = np.concatenate([linalg.eigvals(corr_mat) for corr_mat in train_laplac_mats], axis=0)

test_laplac_mats = [csgraph.laplacian(corr_mat.values, normed=True) for corr_mat in test_corr_mats_pos]
test_eigs = np.concatenate([linalg.eigvals(corr_mat) for corr_mat in test_laplac_mats], axis=0)

# construct x and y for train/test
train_eigs_by_subj = np.vstack(np.split(train_eigs, 200))  # 200 subjects in train
X_train = pd.concat([train_phen, pd.DataFrame(train_eigs_by_subj)], axis=1).drop(columns=['DX', 'Subj'])
y_train = train_phen['DX'].values

test_eigs_by_subj = np.vstack(np.split(test_eigs, 40))  # 40 subjects in test
X_test = pd.concat([test_phen, pd.DataFrame(test_eigs_by_subj)], axis=1).drop(columns=['DX', 'Subj'])
y_test = test_phen['DX'].values

# tune hyper parameters for SVM (~19 min) and save
svm_model = SVC()
param_grid = {'C': [0.01, 0.1, 1, 10, 25, 50, 100],
              'gamma': [0.001, 0.01, 0.1, 0.5, 1],
              'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)
joblib.dump(grid.best_params_, project_root / 'outputdir' / 'gridSearch_best_params')

print(f'The best score: {grid.best_score_},\n'
      f'The best params: {grid.best_params_}')

# train and predict
loaded_params = joblib.load(project_root / 'outputdir' / 'gridSearch_best_params')

svm_model = SVC(**loaded_params)
svm_model.fit(X_train, y_train)
prediction = svm_model.predict(X_test)

# Save the result of prediction to txt file (1 - patient, 0 - control)
np.savetxt(project_root / 'outputdir' / 'classification.txt', prediction)
