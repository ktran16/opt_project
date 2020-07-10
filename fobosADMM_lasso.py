import numpy as np


def fobos_lasso(A, b, w0, eta, mu, max_iters):
    w = np.zeros((len(w0), max_iters+1))
    res = np.zeros(max_iters+1)
    w.T[0] = w0
    t = A_n.dot(w.T[0]).T-b.T
    res[0] = (1/2)*(t.dot(t.T) + mu*np.sum(w.T[0])
    for i in range(max_iters):
        #print(i)
        gf = np.sum([np.asscalar((A[j].dot(w.T[i])-b[j]))*(A[j]) for j in range(len(w.T[i]))])
        wi_12 = w.T[i]-eta*gf
        wi_1 = np.zeros(len(wi_12))
        for j in range(len(wi_12)):
            wi_1j = np.sign(wi_12[j])*(np.abs(wi_12[j])-mu)
            wi_1[j] = (wi_1j)
        w.T[i+1] = wi_1
        t = A_n.dot(w.T[i+1]).T-b.T
        res[i+1] = (1/2)*(t.dot(t.T) + mu*np.sum(w.T[i])
    return w, res
