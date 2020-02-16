import numpy as np
import sklearn
from sklearn import metrics
import sys
import scipy
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os.path

import warnings
warnings.simplefilter('error') # treat warnings as errors

from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
matplotlib.rc('font', size=24)

def Parse(fname, seed):
    if(fname == 'spam.data'):
        all_rows = []
        with open('spam.data') as fp:
            for line in fp:
                row = line.split(' ')
                all_rows.append(row)
        temp_ar = np.array(all_rows, dtype=float)
        temp_ar = temp_ar.astype(float)
    elif(fname == 'SAheart.data'):
        all_rows = []
        with open('SAheart.data') as fp:
            for line in fp:
                row = line.split(',')
                all_rows.append(row)
        all_rows = all_rows[1:]
        all_rows=np.array(all_rows)
        all_rows[all_rows == "Present"] = "1"
        all_rows[all_rows == "Absent"] = "0"
        all_rows= all_rows[:,1:]
        temp_ar = np.array(all_rows, dtype=float)
    elif(fname == 'zip.train'):
        all_rows = []
        with open('zip.train') as fp:
            for line in fp:
                line= line.strip()
                row = line.split(' ')
                all_rows.append(row)
        all_rows=np.array(all_rows)
        all_rows=all_rows[(all_rows[:,0] == "0.0000") |  (all_rows[:,0] == "1.0000")]
        all_rows[:,[0,256]]= all_rows[:,[256,0]]
        temp_ar = np.array(all_rows, dtype=float)
        # remove any cols with only 1 unique element, it'll cause errors
        useless_cols = []
        for col in range(temp_ar.shape[1] - 1): # iterate through all but the last column
            if(np.unique(temp_ar[:, col]).shape[0] == 1):
                useless_cols.append(col)
        temp_ar = np.delete(temp_ar, [useless_cols], 1)
    else:
        raise Exception("Unknown dataset")
    # standardize each column to have μ = 0 and σ^(2) = 1
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

stepSize = float(sys.argv[2])
maxiterations = int(sys.argv[3])
seed = int(sys.argv[4])
temp_ar = Parse("spam.data", seed)

# temp_ar is randomly shuffled at this point
num_rows = temp_ar.shape[0]

X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T # make it a row vector, m x 1
y = y.astype(int)

train_X = X[0: int(num_rows * 0.6)]                        # slice of 0% to 60%
train_y = y[0: int(num_rows * 0.6)]                        # slice of 0% to 60%
test_X = X[int(num_rows * 0.6): int(num_rows * 0.8)]       # slice of 60% to 80%
test_y = y[int(num_rows * 0.6): int(num_rows * 0.8)]       # slice of 60% to 80%
validation_X = X[int(num_rows * 0.8):]                     # slice of 80% to 100%
validation_y = y[int(num_rows * 0.8):]                     # slice of 80% to 100%

print('            y')
print('  {0: >10} {1: >4} {2: >4}'.format('set', '0', '1'))
print('  {0: >10} {1: >4} {2: >4}'.format('test',
                                          str((test_y == 0).sum()),
                                          str((test_y == 1).sum())))
print('  {0: >10} {1: >4} {2: >4}'.format('train',
                                          str((train_y == 0).sum()),
                                          str((train_y == 1).sum())))
print('  {0: >10} {1: >4} {2: >4}'.format('validation',
                                          str((validation_y == 0).sum()),
                                          str((validation_y == 1).sum())))
