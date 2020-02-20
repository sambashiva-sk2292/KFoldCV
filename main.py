# This Python file uses the following encoding: utf-8

import os, sys
import numpy as np
import sklearn
import random
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

# TODO: Use the fold_vec and validation_fold_vec

# These are row vectors which tell you what fold each row is in
# This will make it easier to check after the fact and is what is specified
# in the rubric

# in NearestNeighborCV: with the X_mat, y_vec, best_k, test with X_new at the end
# return what is specified in rubric

def KFoldCV(X_mat,y_vec,ComputePredictions,fold_vec, num_neighbors):
    error_vec = list()
    X_subsets = list()
    y_subsets = list()
    num_rows = X_mat.shape[0]
    num_folds = fold_vec.max()
    for i in range(1, num_folds + 1):
        row_nums = list()
        for j in range(fold_vec.shape[0]):
            if(i == fold_vec[j]):
                row_nums.append(j)
        X_subsets.append(np.copy(X_mat[row_nums]))
        y_subsets.append(np.copy(y_vec[row_nums]))
    for i in range(num_folds):
        X_train=copy.deepcopy(X_subsets)
        #print(X_train[i])  
        del X_train[i]
        X_train=np.concatenate(X_train)
        X_new=X_subsets[i]
        y_train=copy.deepcopy(y_subsets)
        del y_train[i]
        y_train=np.concatenate(y_train)
        y_new=y_subsets[i]
        pred_new=ComputePredictions(X_train,y_train,X_new,num_neighbors)
        error_vec.append(100 * (np.mean(y_new[:, 0] != pred_new)))
    return error_vec

# this is the default one
def ComputePredictions(X_train, y_train, X_new, num_neighbors):
    nneighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X_train)
    distances, indices = nneighbors.kneighbors(X_new)
    pred_new = list()
    for i in range(X_new.shape[0]):
        if (y_train[indices[i]] == 1).sum() > (num_neighbors / 2):
            pred_new.append(1)
        else:
            pred_new.append(0)
    pred_new = np.array(pred_new)
    return pred_new

# this is the best one
##def Best_ComputePredictions(X_train, y_train, X_new):
## get the best k neighbour by NearestNeighborsCV
## use default ComputePredictions to get prediction

# this is the overfit one
##def Over_ComputePredictions(X_train, y_train, X_new):
## get the best k neighbour + 1 by NearestNeighborsCV
## use default ComputePredictions to get prediction

# this is the underfit one
##def Under_ComputePredictions(X_train, y_train, X_new):
## get the most frequent one as prediction

def NearestNeighborsCV(X_mat,y_vec,num_folds=5,max_neighbors=20):
    num_rows = X_mat.shape[0]
    validation_fold_vec = np.repeat(np.arange(1,num_folds+1), num_rows/5, axis = 0)
    random.shuffle(validation_fold_vec)
    print(validation_fold_vec[4])
    error_mat = list();
    for i in range(max_neighbors):
        error_vec = KFoldCV(X_mat, y_vec, ComputePredictions, validation_fold_vec, i + 1)
        print(error_vec)
        error_mat.append(error_vec)
    print(error_mat)

    mean_error_vec = list()
    for index in range(max_neighbors):
        mean_error_vec.append(statistics.mean(error_mat[index]))
    best_k = min(mean_error_vec)
    k = mean_error_vec.index(min(mean_error_vec)) + 1
    print(mean_error_vec)
    print(best_k)
    print(k)
    # may want to return X_mat or something to get the mean error for each fold
    # we'll see once we start graphing
    return pred_new, mean_error_vec,best_k,k

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

num_folds = int(sys.argv[1])
max_neighbors = int(sys.argv[2])
seed = int(sys.argv[3])
temp_ar = Parse("spam.data", seed)
# temp_ar is randomly shuffled at this point
num_rows = temp_ar.shape[0]

X_mat = temp_ar[:, 0:-1] # m x n
X_mat = X_mat.astype(float)
y_vec = np.array([temp_ar[:, -1]]).T # make it a row vector, m x 1
y_vec = y_vec.astype(int)

# print("error %: " + str(100 * (np.mean(y_new[:, 0] != pred_new))))
# import pdb; pdb.set_trace()

#Use NearestNeighborsCV with the whole data set as the training inputs (X_mat/y_vec). 
X_New = np.zeros(shape = (1, num_rows))
y_Prediction,mean_validation_error,min_error,best_neighbour = NearestNeighborsCV(X_mat,y_vec,5,20)

print (mean_validation_error)
print (min_error)
print (best_neighbour)


















