#################################
#          IML_T1B              #
#################################
#
# File Name: main.py
# Course: 252-0220-00L Introduction to Machine Learning
#
# Authors: Adrian Esser (aesser@student.ethz.ch)
#          Abdelrahman-Shalaby (shalabya@student.ethz.ch)

import numpy as np
import sys
import os
import csv
import itertools

from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error
from helpers import *

np.set_printoptions(suppress=True)

# Import training data
data = np.genfromtxt('train.csv', delimiter=',')
data = np.delete(data, 0, 0) # remove first row
data = np.matrix(data)

# Extract data into matrices
X = data[:,2:] # get third column to end
y = data[:,1] # get second column
d = np.shape(X)[1] # number of parameters (should be 5)
n = np.shape(X)[0] # number of data points 


# Build giant data matrix
A = np.concatenate( (X, np.square(X), np.exp(X), np.cos(X), np.ones((n,1))), axis=1 )
dA = np.shape(A)[1] # number of features for model (should be 21)


Lam1 = np.logspace(-3, 4, 8)
Lam2 = np.logspace(-3, 4, 8)
Lam3 = np.logspace(-3, 4, 8)
Lam4 = np.logspace(-3, 4, 8)


lambda_combs = list(itertools.product(Lam1, Lam2, Lam3, Lam4))
rms_min = float('inf')
errors = []
opt_weights = []
opt_lambda = []

k = 20 # cross validation

for tup in lambda_combs:
    
    # compute the weight matrix
    Lambda = lambda_matrix(np.array(tup))

    kf = KFold(n_splits=k, shuffle=False) # define the split, use no shuffling
    rms_vect = []

    for train_index, test_index in kf.split(A):
        # Extract the training and test data
        A_train, A_test = A[train_index,:], A[test_index,:]
        y_train, y_test = y[train_index,:], y[test_index,:]
            

        # compute LS estimates with regularization matrix
        theta_ls_ridge = np.linalg.inv(A_train.T*A_train + Lambda)*A_train.T*y_train

        # Compute predictions
        y_pred = A_test*theta_ls_ridge

        # And the RMSE
        rms_err = mean_squared_error(y_test, y_pred)**0.5 
        rms_vect.append(rms_err)
       
    
    rms_mean = np.mean(rms_err) # compute average RMS error
    errors.append(rms_mean) # keep track of all mean errors
    if rms_mean < rms_min:
        # compute the weights with ALL of the data
        opt_weights = np.linalg.inv(A.T*A + Lambda)*A.T*y
        opt_lambda = tup
        rms_min = rms_mean

print("Optimal Weights: \n", opt_weights)
print("Optimal Lambda: ", opt_lambda)
print(rms_min)



#############################
#   Write Ouput to File     #
#############################
np.savetxt('final_results.csv', opt_weights, fmt='%.12f', newline='\n', comments='')






