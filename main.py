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

# TODO: Eventually get rid of this. Just curious to see largest 
#       and smallest value 
print(np.max(np.max(X)))
print(np.min(np.min(X)))

# Build giant data matrix
A = np.concatenate( (X, np.square(X), np.exp(X), np.cos(X), np.ones((n,1))), axis=1 )
dA = np.shape(A)[1] # number of features for model (should be 21)

# TODO: And how they change after computing the features...
print(np.max(np.max(A)))
print(np.min(np.min(X)))
print("\n")

# For a benchmark, we will compute the parameter set for the 
# simplest case. Plain Least Squares, no cross-validation, no regularization. 
# We will use the RMS error as a benchmark to judge how well
# the more complicated algorithms are performing. 
theta_ls = np.linalg.inv(A.T * A)*A.T*y
y_pred_ls = A*theta_ls
rmse_ls = mean_squared_error(y, y_pred_ls)**0.5 

print("LS Parameters: \n", theta_ls) 
print("\n")
print("RMS Error: ", rmse_ls)
print("\n")

# Idea. So basically we've got two methods to work with here. The cross-validation 
# (using something like KFolds) prevents overfitting of the data, while the regularization 
# parameter tempers the model complexity (by keeping the size of the parameters in check)
# 
# So basically we've got two choices. We need to choose an appropriate 'k' for the 
# KFold cross validation. It may be worth trying LOOCV, but that could be too 
# expensive. We should compare it agains other options (5, 10, etc...)
#
# We also need to choose regularization parameters. My theory is that we should 
# actually try to choose a different regularization parameter for each of the 
# types of features, because the exponential terms are very different from the
# cosine terms, and we don't want the regularization parameter to trade off
# between these two terms (like say we find lambda to be small because the exponential
# terms are large, but then this reduces the predictive power of the cosine terms). 
#
# This is complicated, but we can perform a search over 5 different lambdas
# for each regularization parameter. Question is, can we also couple this 
# with a search over the different KFold values? Or will this 
# become computationally intractable? We need to think about this. 

# First let's iterate over potential regularization parameters (no cross validation)

# First Attempt
Lam1 = np.logspace(-3, 4, 8)
Lam2 = np.logspace(-3, 4, 8)
Lam3 = np.logspace(-3, 4, 8)
Lam4 = np.logspace(-3, 4, 8)
Lam5 = np.logspace(-3, 4, 8)

# Second Attempt
#Lam1 = np.logspace(-3, 2, 6)
#Lam2 = np.logspace(-3, 2, 6)
#Lam3 = np.logspace(-3, 2, 6)
#Lam4 = np.logspace(-3, 2, 6)
#Lam5 = np.logspace(-3, 2, 6)

#sys.exit()

lambda_combs = list(itertools.product(Lam1, Lam2, Lam3, Lam4, Lam5))
errors = []
rms_min = float('inf')
opt_weights = []
opt_lambda = []

for tup in lambda_combs:
    #print(np.array(tup))
    #print(lambda_matrix(np.array(tup)))
    
    # compute the weight matrix
    Lambda = lambda_matrix(np.array(tup))
   
    # TODO: Replace this with proper cross-validation. I'm just
    #       doing it this was as an initial test to see if
    #       the regularization can do better when chunked
    A_train, A_test = np.split(A, [600]) # split @ 600 (ie 600 train, 300 test)
    y_train, y_test = np.split(y, [600])

    # compute LS estimates with regularization matrix
    theta_ls_ridge = np.linalg.inv(A_train.T*A_train + Lambda)*A_train.T*y_train

    # Compute predictions
    y_pred = A_test*theta_ls_ridge

    # And the RMSE
    rms_err = mean_squared_error(y_test, y_pred)**0.5 
    errors.append(rms_err)
    #print(tup, " - Error: ", rms_err)

    if rms_err < rms_min:
        opt_weights = theta_ls_ridge
        opt_lambda = tup
        rms_min = rms_err

#print(errors)
print("Maximum Error: ", np.max(errors))
print("Associated Tuple: ", lambda_combs[np.argmax(errors)])
print("\n")
print("Minimum Error: ", np.min(errors))
print("Associated Tuple: ", lambda_combs[np.argmin(errors)])

print("Optimal Weights: \n", opt_weights)
print("Optimal Lambda: ", opt_lambda)
print(rms_min)

#############################
#   Write Ouput to File     #
#############################
#np.savetxt('results.csv', rms_vect, fmt='%.12f', newline='\n', comments='')




