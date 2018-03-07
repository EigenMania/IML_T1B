import numpy as np
import sys
import os
from sklearn.metrics import mean_squared_error

def lambda_matrix(lambda_vect):
    v = np.reshape(np.tile(lambda_vect, (4,1)).T, (-1,1))
    v = np.append(v, 0) # add a zero so that we don't regularize bias term
    V = np.diag(v) # convert to diagonal matrix
    return V


