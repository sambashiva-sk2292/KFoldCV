import numpy as np
import sklearn
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import sys
import scipy
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os.path
import copy
import warnings
import statistics
warnings.simplefilter('error') # treat warnings as errors

from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=24)

def KFoldCV(X_mat,y_vec,ComputePredictions,num_folds=5):
    error_vec=list()
    X_subsets=list()
    Y_subsets=list()
    num_rows = X_mat.shape[0]
    for i in range(num_folds):
        start = int(num_rows*i*(1/num_folds))
        end = int(num_rows*(i+1)*(1/num_folds))
        X_subsets.append(np.copy(X_mat[start:end]))
        Y_subsets.append(np.copy(y_vec[start:end]))
    for i in range(num_folds):
        X_train=copy.deepcopy(X_subsets)
        del X_train[i]
        X_train=np.concatenate(X_train)
        X_new=X_subsets[i]
        y_train=copy.deepcopy(Y_subsets)
        del y_train[i]
        y_train=np.concatenate(y_train)
        y_new=Y_subsets[i]
        pred_new=ComputePredictions(X_train,y_train,X_new)
        error_vec.append(100 * (np.mean(y_new[:, 0] != pred_new)))
    return error_vec

def ComputePredictions(X_train, y_train, X_new, num_neighbors=20):
    nneighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X_train)
    distances, indicies = nneighbors.kneighbors(X_new)
    pred_new = list()
    for i in range(X_new.shape[0]):
        if (y_train[indicies[i]] == 1).sum() > (num_neighbors / 2):
            pred_new.append(1)
        else:
            pred_new.append(0)
    pred_new = np.array(pred_new)
    return pred_new

def NearestNeighborsCV(X_mat,y_vec,X_new,num_folds=5,max_neighbors=20):
    num_rows = X_mat.shape[0]
    validation_fold_vec = np.repeat(np.arange(1,num_folds), num_rows, axis = 0)
    error_mat = np.zeros(shape = (num_folds, max_neighbors))
    error_mat = error_mat.transpose
    for out_index in range(max_neighbors):
        error_vec = KFoldCV(X_mat, y_vec, ComputePredictions, num_folds)
        for inner_index in range(num_folds):
            error_mat[out_index, inner_index] = error_vec[inner_index]
    mean_error_vec = np.zeros(shape = (1,max_neighbors))
    for index in range(max_neighbors)
        mean_error_vec[index] = statistics.mean(error_mat[index, : ])
    best_neighbors = min(mean_error_vec)

def Parse(fname, seed):
    all_rows = []
    with open(fname) as fp:
        for line in fp:
            row = line.split(' ')
            all_rows.append(row)
    temp_ar = np.array(all_rows, dtype=float)
    temp_ar = temp_ar.astype(float)
    # standardize each column to have ¦Ì = 0 and ¦Ò^(2) = 1
    # in other words convert all elements to z-scores for each column
    for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
        std = np.std(temp_ar[:, col])
        if(std == 0):
            print("col " + str(col) + " has an std of 0")
        temp_ar[:, col] = stats.zscore(temp_ar[:, col])
    np.random.seed(seed)
    np.random.shuffle(temp_ar) # shuffle rows, set of columns remain the same
    return temp_ar

if len(sys.argv) < 4:
    help_str = """Execution example: python3 main.py <stepSize> <maxiterations> <seed>
stepSize must be a float
maxiterations must be an int
seed must be an int
"""
    print(help_str)
    exit(0)

stepSize = float(sys.argv[1])
maxiterations = int(sys.argv[2])
seed = int(sys.argv[3])
temp_ar = Parse("spam.data", seed)

# temp_ar is randomly shuffled at this point
num_rows = temp_ar.shape[0]

X_mat = temp_ar[:, 0:-1] # m x n
X_mat = X_mat.astype(float)
y_vec = np.array([temp_ar[:, -1]]).T # make it a row vector, m x 1
y_vec = y_vec.astype(int)

error_vec = KFoldCV(X_mat,y_vec,ComputePredictions)
print(str(error_vec))

# print("error %: " + str(100 * (np.mean(y_new[:, 0] != pred_new))))
# import pdb; pdb.set_trace()
