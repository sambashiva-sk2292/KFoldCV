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
    all_rows = []
    with open(fname) as fp:
        for line in fp:
            row = line.split(' ')
            all_rows.append(row)
    temp_ar = np.array(all_rows, dtype=float)
    temp_ar = temp_ar.astype(float)
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

stepSize = float(sys.argv[1])
maxiterations = int(sys.argv[2])
seed = int(sys.argv[3])
temp_ar = Parse("spam.data", seed)

# temp_ar is randomly shuffled at this point
num_rows = temp_ar.shape[0]

X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T # make it a row vector, m x 1
y = y.astype(int)

print('            y')
print('  {0: >10} {1: >4} {2: >4}'.format('set', '0', '1'))
print('  {0: >10} {1: >4} {2: >4}'.format('dataset',
                                          str((y == 0).sum()),
                                          str((y == 1).sum())))

