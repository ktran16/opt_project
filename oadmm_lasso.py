import numpy as np
from numpy.linalg import norm, cholesky
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time


# supporting functions
def objective_function():
    return None

def soft_threshold(k, gamma): # shrinkage operator function
  ''' 
  funtion to update z
  '''
  if k < - gamma:
    return k + gamma
  
  elif k > gamma:
    return k - gamma 
  
  else:
    return 0


def factor():
    return None


# main function of Online ADMM for Lasso
def oadmm_lasso():
    return None


    


