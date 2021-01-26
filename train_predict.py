import os

import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.sparse import csgraph
from sklearn.svm import SVC

## Read the paths to the data
# Training data
tr_X_path = r"inputdir\2019_CNI_TrainingRelease-master\Training"
tr_Y_path = r"inputdir\2019_CNI_TrainingRelease-master\SupportingInfo\phenotypic_training.csv"

X_tr_data_aal = [pd.read_csv((a+'\\'+c[1]), header=None) for (a,b,c) in os.walk(tr_X_path) if len(c)>1]
X_tr_data_cc200 = [pd.read_csv((a+'\\'+c[2]), header=None) for (a,b,c) in os.walk(tr_X_path) if len(c)>1]
X_tr_data_ho = [pd.read_csv((a+'\\'+c[3]), header=None) for (a,b,c) in os.walk(tr_X_path) if len(c)>1]

Y_tr_data = pd.read_csv(tr_Y_path)

# Test data
test_X_path = r"inputdir\2019_CNI_ValidationRelease-master\Validation"
test_Y_path = r"inputdir\2019_CNI_ValidationRelease-master\SupportingInfo\phenotypic_validation.csv"

X_test_data_aal = [pd.read_csv((a+'\\'+c[1]), header=None) for (a,b,c) in os.walk(test_X_path) if len(c)>1]
X_test_data_cc200 = [pd.read_csv((a+'\\'+c[2]), header=None) for (a,b,c) in os.walk(test_X_path) if len(c)>1]
X_test_data_ho = [pd.read_csv((a+'\\'+c[3]), header=None) for (a,b,c) in os.walk(test_X_path) if len(c)>1]

Y_test_data = pd.read_csv(test_Y_path)

# Rename string answers to ints
Y_tr_data = Y_tr_data.replace({"ADHD": 1, "Control": 0, "F": 0, "M": 1}).drop(columns='Subj')
Y_test_data = Y_test_data.replace({"ADHD": 1, "Control": 0, "F": 0, "M": 1}).drop(columns='Subj')

# Calculate corr matrix for train and test datasets
corr_mats_aal = [df.T.corr() for df in X_tr_data_aal]
corr_mats_cc200 = [df.T.corr() for df in X_tr_data_cc200]
corr_mats_ho = [df.T.corr() for df in X_tr_data_ho]

corr_mats_test_aal = [df.T.corr() for df in X_test_data_aal]
corr_mats_test_cc200 = [df.T.corr() for df in X_test_data_cc200]
corr_mats_test_ho = [df.T.corr() for df in X_test_data_ho]


# Change to zero negative values for training (for future Laplacian)
for subj in corr_mats_aal:
    np.fill_diagonal(subj.values, 0)
    for n in np.nditer(subj.values, op_flags=['readwrite']):
        if n < 0:
            n[...] = 0

for subj in corr_mats_cc200:
    np.fill_diagonal(subj.values, 0)
    for n in np.nditer(subj.values, op_flags=['readwrite']):
        if n < 0:
            n[...] = 0

for subj in corr_mats_ho:
    np.fill_diagonal(subj.values, 0)
    for n in np.nditer(subj.values, op_flags=['readwrite']):
        if n < 0:
            n[...] = 0

# Change to zero negative values for validation (for future Laplacian)
for subj in corr_mats_test_aal:
    np.fill_diagonal(subj.values, 0)
    for n in np.nditer(subj.values, op_flags=['readwrite']):
        if n < 0:
            n[...] = 0

for subj in corr_mats_test_cc200:
    np.fill_diagonal(subj.values, 0)
    for n in np.nditer(subj.values, op_flags=['readwrite']):
        if n < 0:
            n[...] = 0

for subj in corr_mats_test_ho:
    np.fill_diagonal(subj.values, 0)
    for n in np.nditer(subj.values, op_flags=['readwrite']):
        if n < 0:
            n[...] = 0

# Laplacian and eigenavalues
laplac_mats_aal = [csgraph.laplacian(corr_mat_pat.values, normed=True) for corr_mat_pat in corr_mats_aal]
laplac_mats_cc200 = [csgraph.laplacian(corr_mat_pat.values, normed=True) for corr_mat_pat in corr_mats_cc200]
laplac_mats_ho = [csgraph.laplacian(corr_mat_pat.values, normed=True) for corr_mat_pat in corr_mats_ho]

eigs_aal = [LA.eigvals(corr_mat) for corr_mat in laplac_mats_aal]
eigs_cc200 = [LA.eigvals(corr_mat) for corr_mat in laplac_mats_cc200]
eigs_ho = [LA.eigvals(corr_mat) for corr_mat in laplac_mats_ho]

laplac_mats_test_aal = [csgraph.laplacian(corr_mat_pat.values, normed=True) for corr_mat_pat in corr_mats_test_aal]
laplac_mats_test_cc200 = [csgraph.laplacian(corr_mat_pat.values, normed=True) for corr_mat_pat in corr_mats_test_cc200]
laplac_mats_test_ho = [csgraph.laplacian(corr_mat_pat.values, normed=True) for corr_mat_pat in corr_mats_test_ho]

eigs_test_aal = [LA.eigvals(corr_mat) for corr_mat in laplac_mats_test_aal]
eigs_test_cc200 = [LA.eigvals(corr_mat) for corr_mat in laplac_mats_test_cc200]
eigs_test_ho = [LA.eigvals(corr_mat) for corr_mat in laplac_mats_test_ho]

# Construct dataset to train
X_aal = np.vstack(eigs_aal)
X_cc200 = np.vstack(eigs_cc200)
X_ho = np.vstack(eigs_ho)

X_test_aal = np.vstack(eigs_test_aal)
X_test_cc200 = np.vstack(eigs_test_cc200)
X_test_ho = np.vstack(eigs_test_ho)

X_train = pd.concat([Y_tr_data, pd.DataFrame(X_aal), pd.DataFrame(X_cc200), pd.DataFrame(X_ho)], axis=1).drop(columns='DX')
y_train= Y_tr_data['DX'].values

X_test = pd.concat([Y_test_data, pd.DataFrame(X_test_aal), pd.DataFrame(X_test_cc200), pd.DataFrame(X_test_ho)], axis=1).drop(columns='DX')
y_test = Y_test_data['DX'].values

# Train with optimal hyperparameters (was calculated in other script)
svm_model = SVC(C=25, gamma=0.001, kernel='poly')
svm_model.fit(X_train, y_train)
pred_svm = svm_model.predict(X_test)

# Save the result of prediction to txt file (1 - patient, 0 - control)
np.savetxt(os.getcwd()+'\outputdir\classification.txt', pred_svm)
