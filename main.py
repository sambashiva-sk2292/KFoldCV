# This Python file uses the following encoding: iso-8859-1

import os, sys
import numpy as np
import sklearn
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
figure(num=None, figsize=(30, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=24)
from gradient import Gradient, GradientDescent, MeanLogisticLoss

def KFoldCV(X_mat,y_vec,my_function,fold_vec, num_neighbors):
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
        pred_new = None
        if(my_function.__name__ == 'ComputePredictions'):
            pred_new=my_function(X_train,y_train,X_new,num_neighbors)
        elif(my_function.__name__ == 'NearestNeighborsCV'):
            pred_new=my_function(X_train,y_train,X_new)
            pred_new=pred_new[0]
        elif(my_function.__name__ == 'Baseline'):
            pred_new=my_function(X_train,y_train,X_new)
        elif(my_function.__name__ == 'OneNearestNeighbors'):
            pred_new=my_function(X_train,y_train,X_new)
        elif(my_function.__name__ == 'GradientDesent'):
            weight_mat = my_function(X_train,y_train,0.1,800)
            pred_new = np.matmul(X_new, weight_mat)
        else:
            raise Exception("Unexpected function")
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
    np.random.shuffle(validation_fold_vec)
    error_mat = np.zeros((num_folds, max_neighbors), dtype=float)
    for i in range(max_neighbors):
        error_vec = KFoldCV(X_mat, y_vec, ComputePredictions, validation_fold_vec, i + 1)
        error_mat[:, i] = error_vec
    mean_error_vec = list()
    for index in range(max_neighbors):
        mean_error_vec.append(statistics.mean(error_mat[:, index]))
    min_error = min(mean_error_vec)
    best_neighbours = mean_error_vec.index(min(mean_error_vec)) + 1
    pred_new = np.array([])
    if(X_new.shape[0] != 0):
        pred_new = ComputePredictions(X_mat, y_vec, X_new, best_neighbours)
    return pred_new, mean_error_vec, min_error, best_neighbours, error_mat

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
    for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
        std = np.std(temp_ar[:, col])
        if(std == 0):
            print("col " + str(col) + " has an std of 0")
        temp_ar[:, col] = stats.zscore(temp_ar[:, col])
    return temp_ar

if len(sys.argv) < 4:
    help_str = """Execution example: python3 main.py <No.of folds k> <No. of Nearest neighbors> <seed>
Folds must be a float
NN must be an int
seed must be an int
"""
    print(help_str)
    exit(0)

num_folds = int(sys.argv[1])
max_neighbors = int(sys.argv[2])
seed = int(sys.argv[3])
np.random.seed(seed)
temp_ar = Parse("spam.data")

X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T 
y = y.astype(int)

import pdb; pdb.set_trace()

pred_new,mean_validation_error,min_error,best_neighbours, error_mat = NearestNeighborsCV(X, y, np.array([]), num_folds, max_neighbors)

error_fold_1 = error_mat[0]
min_error_1 = min(error_fold_1)
best_neighbour_1 = np.where(error_fold_1 == np.amin(error_fold_1))[0][0] + 1

error_fold_2 = error_mat[1]
min_error_2 = min(error_fold_2)
best_neighbour_2 = np.where(error_fold_2 == np.amin(error_fold_2))[0][0] + 1

error_fold_3 = error_mat[2]
min_error_3 = min(error_fold_3)
best_neighbour_3 = np.where(error_fold_3 == np.amin(error_fold_3))[0][0] + 1

error_fold_4 = error_mat[3]
min_error_4 = min(error_fold_4)
best_neighbour_4 = np.where(error_fold_4 == np.amin(error_fold_4))[0][0] + 1

error_fold_5 = error_mat[4]
min_error_5 = min(error_fold_5)
best_neighbour_5 = np.where(error_fold_5 == np.amin(error_fold_5))[0][0] + 1

x = [i for i in range(1, len(mean_validation_error) + 1)]
plt.plot(x, mean_validation_error, c="red", linewidth=3, label='mean_validation')
plt.scatter(best_neighbours, min_error, marker='o', edgecolors='r', s=160, facecolor='none', linewidth=3, label='minimum')
plt.xlabel("Number of Neighbors")
plt.ylabel("Mean Validation Error %")
plt.legend()
plt.tight_layout()
plt.savefig("validation_error.png")
plt.clf()


plt.plot(x, mean_validation_error, c="red", linewidth=6, label='mean_validation')
plt.scatter(best_neighbours, min_error, marker='o', edgecolors='r', s=160, facecolor='none', linewidth=3, label='minimum')

plt.plot(x, error_fold_1, c="black", linewidth=4, label='fold1_validation')
plt.scatter(best_neighbour_1, min_error_1, marker='o', edgecolors='black', s=150, facecolor='none', linewidth=3, label='minimum')

plt.plot(x, error_fold_2, c="blue", linewidth=4, label='fold2_validation')
plt.scatter(best_neighbour_2, min_error_2, marker='o', edgecolors='blue', s=150, facecolor='none', linewidth=3, label='minimum')

plt.plot(x, error_fold_3, c="yellow", linewidth=4, label='fold3_validation')
plt.scatter(best_neighbour_3, min_error_3, marker='o', edgecolors='yellow', s=150, facecolor='none', linewidth=3, label='minimum')

plt.plot(x, error_fold_4, c="pink", linewidth=4, label='fold4_validation')
plt.scatter(best_neighbour_4, min_error_4, marker='o', edgecolors='pink', s=150, facecolor='none', linewidth=3, label='minimum')

plt.plot(x, error_fold_5, c="green", linewidth=4, label='fold5_validation')
plt.scatter(best_neighbour_5, min_error_5, marker='o', edgecolors='green', s=150, facecolor='none', linewidth=3, label='minimum')
plt.xlabel("Number of Neighbors")
plt.ylabel("Mean Validation Error %")
plt.legend(ncol=2,fontsize=15)
plt.tight_layout()
plt.savefig("extra_validation_error.png")
plt.clf()

num_rows = X.shape[0]
test_fold_num = 4
test_fold_vec = np.random.randint(1, test_fold_num + 1, num_rows)
test_accuracy_vec_baseline = 100 - KFoldCV(X, y, Baseline, test_fold_vec, best_neighbours)
plt.scatter(test_accuracy_vec_baseline, test_fold_num * ['Baseline'], color='black')
test_accuracy_vec_nncv = 100 - KFoldCV(X, y, NearestNeighborsCV, test_fold_vec, best_neighbours)
plt.scatter(test_accuracy_vec_nncv, test_fold_num * [str(best_neighbours) + '-NN'], color='red')
test_accuracy_vec_onn = 100 - KFoldCV(X, y, OneNearestNeighbors, test_fold_vec, best_neighbours)
plt.scatter(test_accuracy_vec_onn, test_fold_num * ['OneNN'], color='blue')
plt.xlabel("accuracy.percent %")
plt.ylabel("algorithm")
plt.tight_layout()
plt.savefig("test_accuracy.png")
plt.clf()

auc = {'baseline': [], 'k-nn': [], 'one-nn': []}
for i in range(test_fold_num):
    test_indices = np.where(test_fold_vec == i + 1)[0]
    train_indices = np.where(test_fold_vec != i + 1)[0]
    train_x = X[train_indices]
    train_y = y[train_indices]
    test_x = X[test_indices]
    test_y = y[test_indices]
    baseline_preds = Baseline(train_x, train_y, test_x)
    k_nn_preds = ComputePredictions(train_x, train_y, test_x, best_neighbours)
    one_nn_preds = ComputePredictions(train_x, train_y, test_x, 1)
    
    fpr, tpr, baseline_thresholds = metrics.roc_curve(test_y, baseline_preds, pos_label=1)
    plt.plot(fpr, tpr, linestyle='dotted', label="baseline" if i == 0 else "", color='black', linewidth=3, alpha=0.5)
    auc['baseline'].append(metrics.roc_auc_score(test_y, baseline_preds))
    
    fpr, tpr, k_nn_thresholds = metrics.roc_curve(test_y, k_nn_preds, pos_label=1)
    plt.plot(fpr, tpr, label="k-nn" if i == 0 else "", color='red', linewidth=3, alpha=0.5)
    auc['k-nn'].append(metrics.roc_auc_score(test_y, k_nn_preds))
    
    fpr, tpr, one_nn_thresholds = metrics.roc_curve(test_y, one_nn_preds, pos_label=1)
    plt.plot(fpr, tpr, linestyle='dashed', label="one-nn" if i == 0 else "", color='blue', linewidth=3, alpha=0.5)
    auc['one-nn'].append(metrics.roc_auc_score(test_y, one_nn_preds))
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.clf()

plt.xlabel("area")
plt.ylabel("algorithm")
for algorithm in auc:
    for value in auc[algorithm]:
        my_color = None
        if(algorithm == 'baseline'):
            my_color = 'black'
        elif(algorithm == 'k-nn'):
            my_color = 'red'
        elif(algorithm == 'one-nn'):
            my_color = 'blue'
        plt.scatter(value, algorithm, color=my_color)
plt.tight_layout()
plt.savefig("auc_plot.png")
plt.clf()

# a table of counts with a row for each fold
X_subsets = list()
y_subsets = list()
for i in range(1, test_fold_num + 1):
    row_nums = list()
    for j in range(test_fold_vec.shape[0]):
        if(i == test_fold_vec[j]):
            row_nums.append(j)
    X_subsets.append(np.copy(X[row_nums]))
    y_subsets.append(np.copy(y[row_nums]))

print('            y')
print('  {0: >10} {1: >4} {2: >4}'.format('set', '0', '1'))
for i in range(test_fold_num):
    zero_count = (y_subsets[i] == 0).sum()
    one_count = (y_subsets[i] == 1).sum()
    print('  {0: >10} {1: >4} {2: >4}'.format('Fold' + str(i),
                                              str(zero_count),
                                              str(one_count)))
