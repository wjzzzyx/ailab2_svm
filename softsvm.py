from math import exp
import numpy as np
from sklearn import svm
from scipy.linalg import *
from cvxopt import solvers, matrix
from getFeature import X, y

'''
def softsvm(traindata, trainlabel, testdata, testlabel, sigma, C):
    n = (traindata.shape)[0]
    trainlabel = trainlabel.reshape(n, 1)

    K = KMatrix(traindata, sigma)
    P = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(n):
            if trainlabel[i] == trainlabel[j]:
                P[i, j] = K[i, j]
            else:
                P[i, j] = -K[i, j]
    P = matrix(P)
    q = np.full((n, 1), -1)
    G = matrix(np.row_stack((-np.identity(n), np.identity(n))))
    h = np.row_stack((np.zeros((n, 1)), np.full((n, 1), C)))
    A = matrix(trainlabel.T)
    b = 0

    alpha = solvers.qp(P, q, G, h, A, b)
    print alpha

def KMatrix(X, sigma):
    n = X.shape[0]
    if sigma == 0:
        return X.dot(X.T)
    else:
        K = np.zeros(n, n)
        for i in xrange(n):
            for j in xrange(n):
                t = norm((X.getrow(i) - X.getrow(j)).todense())
                K[i, j] = exp(- t**2 / sigma**2)
'''

def softsvm(traindata, trainlabel, testdata, testlabel, sigma, C):
    if sigma == 0:
        clf = svm.SVC(kernel='linear', C=C)
    else:
        clf = svm.SVC(kernel='rbf')
    clf.fit(traindata, trainlabel)
    y_pred = clf.predict(testdata)

    n1 = testdata.shape[0]
    correct_count = 0
    for i in xrange(n1):
        if y_pred[i] == testlabel[i]:
            correct_count += 1

    print correct_count
    return y_pred, (correct_count + 0.0) / n1

n = (X.shape)[0]
softsvm(X[0 : int(0.8 * n)][:], y[0 : int(0.8 * n)], X[int(0.8 * n) : n][:], y[int(0.8 * n) : n], 0, 1)