# This Python file uses the following encoding: iso-8859-1

import os, sys
import numpy as np
import sklearn
import random
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
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
figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=24)

def KFoldCV(X_mat,y_vec,function,fold_vec, num_neighbors):
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
        del X_train[i]
        X_train=np.concatenate(X_train)
        X_new=X_subsets[i]
        y_train=copy.deepcopy(y_subsets)
        del y_train[i]
        y_train=np.concatenate(y_train)
        y_new=y_subsets[i]
        # three times?
        pred_new=function(X_train,y_train,X_new,num_neighbors)
        pred_new=function(X_train,y_train,X_new,5,20)
        pred_new=function(X_train,y_train,X_new)

        error_vec.append(100 * (np.mean(y_new[:, 0] != pred_new)))
    error_vec = np.array(error_vec)
    return error_vec

def ComputePredictions(X_train, y_train, X_new, num_neighbors):
    nneighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X_train)
    distances, indices = nneighbors.kneighbors(X_new)
    pred_new = list()
    for i in range(X_new.shape[0]):
        count = (y_train[indices[i]] == 1).sum()
        if count > (num_neighbors / 2):
            pred_new.append(1)
        elif count == (num_neighbors / 2):
            coin_flip = np.random.randint(0, 2, 1)[0]
            if(coin_flip == 1):
                pred_new.append(1)
            else:
                pred_new.append(0)
        else:
            pred_new.append(0)
    pred_new = np.array(pred_new)
    return pred_new

def NearestNeighborsCV(X_mat,y_vec,X_new,num_folds=5,max_neighbors=20):
    num_rows = X_mat.shape[0]
    validation_fold_vec = np.random.randint(1, num_folds + 1, num_rows)
    # validation_fold_vec = np.repeat(np.arange(1,num_folds+1), num_rows/5, axis = 0)
    random.shuffle(validation_fold_vec)
    # error_mat is a num_fold x max_neighbors matrix
    error_mat = np.zeros((num_folds, max_neighbors), dtype=float)
    for i in range(max_neighbors):
        error_vec = KFoldCV(X_mat, y_vec, ComputePredictions, validation_fold_vec, i + 1)
        error_mat[:, i] = error_vec
    mean_error_vec = list()
    for index in range(max_neighbors):
        mean_error_vec.append(statistics.mean(error_mat[:, index]))
    min_error = min(mean_error_vec)
    best_neighbours = mean_error_vec.index(min(mean_error_vec)) + 1
    print("mean_error_vec = " + str(mean_error_vec))
    print("min_error = " + str(min_error))
    print("best_neighbours = " + str(best_neighbours))
    # what train test split do we do here?
    pred_new = np.array([])
    if(X_new.shape[0] != 0):
        pred_new = ComputePredictions(X_mat, y_vec, X_new, best_neighbours)
    # may want to return X_mat or something to get the mean error for each fold
    # we'll see once we start graphing
    return pred_new, mean_error_vec, min_error, best_neighbours

def OneNearestNeighbors(X_mat, y_vec, X_new):
    pred_new = ComputePredictions(X_mat, y_vec, X_new, 1)
    return pred_new

def Baseline(X_mat, y_vec, X_new):
    pred_new = np.zeros((X_new.shape[0],))
    if (y_vec == 1).sum() > (y_vec.shape[0] / 2):
        pred_new = np.where(pred_new == 0, 1, pred_new)
    return pred_new

def Parse(fname):
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
      atter(test_error_vec, 4 * [str(best_neighbours) + '-NearestNeighbors'])
plt.xlabel("Mean Validation Error %")
plt.ylabel("algorithm")
plt.savefig("test_errors.png")
plt.clf()

# a table of counts with a row for each fold
X_subsets = list()
y_subsets = list()
num_rows = X_mat.shape[0]
for i in range(1, 4 + 1):
    row_nums = list()
    for j in range(test_fold_vec.shape[0]):
        if(i == test_fold_vec[j]):
            row_nums.append(j)
    X_subsets.append(np.copy(X_mat[row_nums]))
    y_subsets.append(np.copy(y_vec[row_nums]))
print('            y')
print('  {0: >10} {1: >4} {2: >4}'.format('set', '0', '1'))
for i in range(num_folds):
    print('  {0: >10} {1: >4} {2: >4}'.format('Fold'+str(i),                                     
                                          str(((y_subsets[i] == 0).sum()),
                                          str((y_subsets[i] == 1).sum())))

    
# IT WILL USE THE ENTIRE DATASET
# WE SPLIT IT INTO FOUR PARTS AND USE ALL THREE ALGORITHMS
# YOU SHOULD ACCOUNT FOR EDGE CASE WHERE 1s equal 0s
# IT'LL BE EASY TO JUST RANDOMLY PICK A WINNER

# 5 20 5 gives us k = 1
# 5 20 1 gives us k = 1
# 10 20 2 gives us k = 1
# 8 20 2 ives us k = 3  temp_ar[:, col] = stats.zscore(temp_ar[:, col])
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
np.random.seed(seed)
temp_ar = Parse("spam.data")
# temp_ar is randomly shuffled at this point
num_rows = temp_ar.shape[0]

X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T # make it a row vector, m x 1
y = y.astype(int)

# print("error %: " + str(100 * (np.mean(y_new[:, 0] != pred_new))))
# import pdb; pdb.set_trace()

pred_new,mean_validation_error,min_error,best_neighbours = NearestNeighborsCV(X, y, np.array([]), 5, 20)
x = [i for i in range(1, len(mean_validation_error) + 1)]
plt.plot(x, mean_validation_error, c="red", linewidth=3, label='validation')
plt.scatter(best_neighbours, min_error, marker='o', edgecolors='r', s=160, facecolor='none', linewidth=3, label='minimum')
plt.xlabel("Number of Neighbors")
plt.ylabel("Mean Validation Error %")
plt.legend()
plt.savefig("validation_error.png")
plt.clf()

num_rows = X.shape[0]
# create random fold vec
test_fold_vec = np.random.randint(1, 4 + 1, num_rows) 
test_error_vec = KFoldCV(X, y, ComputePredictions, test_fold_vec, best_neighbours)
plt.scatter(test_error_vec, 4 * [str(best_neighbours) + '-NearestNeighbors'])
plt.xlabel("Mean Validation Error %")
plt.ylabel("algorithm")
plt.savefig("test_errors.png")
plt.clf()

# a table of counts with a row for each fold
X_subsets = list()
y_subsets = list()
num_rows = X_mat.shape[0]
for i in range(1, 4 + 1):
    row_nums = list()
    for j in range(test_fold_vec.shape[0]):
        if(i == test_fold_vec[j]):
            row_nums.append(j)
    X_subsets.append(np.copy(X_mat[row_nums]))
    y_subsets.append(np.copy(y_vec[row_nums]))
print('            y')
print('  {0: >10} {1: >4} {2: >4}'.format('set', '0', '1'))
for i in range(num_folds):
    print('  {0: >10} {1: >4} {2: >4}'.format('Fold'+str(i),                                     
                                          str(((y_subsets[i] == 0).sum()),
                                          str((y_subsets[i] == 1).sum())))

# IT WILL USE THE ENTIRE DATASET
# WE SPLIT IT INTO FOUR PARTS AND USE ALL THREE ALGORITHMS
# YOU SHOULD ACCOUNT FOR EDGE CASE WHERE 1s equal 0s
# IT'LL BE EASY TO JUST RANDOMLY PICK A WINNER

# 5 20 5 gives us k = 1
# 5 20 1 gives us k = 1
# 10 20 2 gives us k = 1
# 8 20 2 ives us k = 3
