#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:06:50 2020
COMP0078 Supervised Learning
Question 1
@author: Anthony, Douglas
"""


import numpy             as np
import matplotlib.pyplot as plt


plt.close('all')

def transform(X, k):
    # Transforms the data set X to k dimension features by feature map phi(x) = x^{k - 1}
    m   = X.shape[0]
    Phi = np.zeros((m, k))
    
    for i in range(k):
        Phi[:, i] = X[:, 0] ** i
    
    return Phi

def fit(X, y):
    # fit data and output the fitted w
    # w = (X^{T}X)^{-1}X^T y
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

def predict(X, w):
    # predict y from data and w
    # y^{hat} = Xw
    return X.dot(w)
    
m = 4
k = 4

X = [[1],
     [2],
     [3],
     [4]]
y = [3, 2, 0, 5]

X = np.array(X)
y = np.array(y)


plt.figure()
for i in range(1,5):
    Phi = transform(X, i)
    w = fit(Phi, y)
    
    # Calculate mean squared error
    # MSE = (Xw - y)^T(Xw - y)
    mse = 1/m * np.sum((predict(Phi, w) - y) ** 2) 
    print('k = %d, MSE = %.2f, w =' % (i, mse), w)
    
    # Generate points for plotting
    plotx = np.linspace(0, 5, 50)
    plotPhi = transform(plotx[:, np.newaxis], i)
    ploty = predict(plotPhi, w)
    
    plt.plot(plotx, ploty, label='k = %d'% i)

plt.plot(X.ravel(), y, 'bo', label='data')

plt.axis([0, 5, -5, 8])
plt.legend()
    
plt.show()