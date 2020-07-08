import numpy as np


def fobos_lasso(A, b, w0, eta, mu, max_iters):
    w = np.zeros((len(w0), max_iters+1))
    #print(w[0], w0)
    w.T[0] = w0
    for i in range(max_iters):
        print(i)
        gf = np.sum([np.asscalar((A[j].dot(w.T[i])-b[j]))*(A[j]) for j in range(len(w.T[i]))])
        wi_12 = w.T[i]-eta*gf
        wi_1 = np.zeros(len(wi_12))
        for j in range(len(wi_12)):
            wi_1j = np.sign(wi_12[j])*(np.abs(wi_12[j])-mu)
            wi_1[j] = (wi_1j)
        w.T[i+1] = wi_1
    return w
