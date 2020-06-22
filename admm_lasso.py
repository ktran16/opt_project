import numpy as np
from numpy.linalg import norm, cholesky
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time


# supporting functions

def objective_function(A, b, lamda, x, z):
    return 1/2 * np.square(A.dot(x) - b).sum() + lamda * norm(z, 1)

def shrinkage(x, kappa):
    ''' 
    function to update z
    '''
    return np.maximum(0., x - kappa) - np.maximum(0., -x - kappa) # soft thresholding

def factor(A, rho):
    m, n = A.shape
    if m >= n:
        L = cholesky(A.T.dot(A) + rho * sparse.eye(n))
    else:
        L = cholesky(sparse.eye(m) + 1. / rho * (A.dot(A.T)))
    
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L, U

# main function of ADMM Lasso
def admm_lasso(A, b, lamda, rho, rel_par, max_iters, abstol, reltol, QUIET = True):
    '''
    minimize 1/2*|| Ax - b ||_2^2 + lamda * || x ||_1
    A: input features or independent variables m * n
    b: label feature m * 1
    lamda (λ > 0) is a scalar regularization parameter that is usually chosen by cross-validation
    rho: augmented Lagrangian parameter. (ρ)
    rel_par: over-relaxation parameter (typical values are between 1.0 and 1.8)
    max_iter: maximum number of iteration
    abstol:
    reltol: 

    OUTPUT:
    vector x: n * 1
    '''
    if not QUIET:
        tic = time.time()

    m, n = A.shape
    
    #save a matrix-vector multiply
    Atb = A.T.dot(b)
    
    # initializing parameters
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    u = np.zeros((n, 1))

    # L is a lower triangular matrix, U is an upper triangular matrix
    # cache the (Cholesky) factorization
    L, U = factor(A, rho)

    # Create a dictionary h to save states
    h = {}
    # initilizing values
    h['obj_val'] = np.zeros(max_iters)  # objective function value
    h['r_norm'] = np.zeros(max_iters)  # primal residual
    h['s_norm'] = np.zeros(max_iters)  # dual residual
    h['eps_pri'] = np.zeros(max_iters) # epsilon primal
    h['eps_dual'] = np.zeros(max_iters) # epsilon dual
    '''                                   
    eps_primal > 0 and eps_dual > 0 are feasibility tolerances for the primal and dual feasibility conditions
    ''' 

    # iterations
    for k in range(max_iters):
        # x update
        q = np.array(Atb) + rho * (z - u)[0] # temporary value
        ''' v = z - u
        '''
        if m >= n:
            x = spsolve(U, spsolve(L, q))[..., np.newaxis] # numpy.newaxis is used to increase the dimension of the existing array by one more dimension, when used once.
        else:
            ULAq = spsolve(U, spsolve(L, A.dot(q)))[..., np.newaxis]
            x = (q * 1. / rho) - ((A.T.dot(ULAq)) * 1. / (rho ** 2))

        # z update with relaxation
        z_old = np.copy(z)
        x_hat = rel_par * x + (1. - rel_par) * z_old
        z = shrinkage(x_hat + u, lamda / rho)

        # u update
        u += (x_hat - z)

        # diagnostics, reporting, termination checks
        h['obj_val'][k] = objective_function(A, b, lamda, x, z)
        h['r_norm'][k] = norm(x - z)
        h['s_norm'][k] = norm(-rho * (z - z_old))
        h['eps_pri'][k] = np.sqrt(n) * abstol + reltol * np.maximum(norm(x), norm(-z))
        h['eps_dual'][k] = np.sqrt(n) * abstol + reltol * norm(rho * u)

        if (h['r_norm'][k] , h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
            break

    if not QUIET:
        toc = time.time() - tic
        print('Elapsed time is {:.2f} seconds'.format(toc))    

    return z.ravel(), h 
