#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 03:07:59 2020
COMP0078 Supervised Learning
Question 3
@author: Anthony, Douglas
"""


import numpy             as np
import matplotlib.pyplot as plt


plt.close('all')
np.random.seed(0)

def transform(X, k):
    # Transforms the data to a new feature space
    m = X.shape[0]
    Phi = np.zeros((m, k))
    
    for i in range(k):
        # phi(x) = sin(i pi x)
        Phi[:, i] = np.sin((i+1) * np.pi * X[:, 0])
    
    return Phi

def fit(X, y):
    # fit data and output the fitted w
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

def predict(X, w):
    # predict y from data and w
    return X.dot(w)

def g(x, sigma):
    # Generates data
    m = x.shape[0]
    eps = np.random.randn(m) * sigma
    return np.sin(2 * np.pi * x) ** 2 + eps

ks = [i for i in range(1, 19)]

# Plot log training error versus polynomial dimensions
m     = 30
sigma = 0.07

X = np.random.rand(m, 1)
y = g(X.ravel(), sigma)

ws    = [] # trained w
logte = [] # log training errors
for k in ks:
    Phi = transform(X, k)
    w   = fit(Phi, y)
    ws.append(w)
    
    te = 1/m * np.sum((predict(Phi, w) - y) ** 2) # calculate training error
    logte.append(np.log(te))

# Plot log training error versus k
plt.figure()
plt.plot(ks, logte)
plt.xlabel('k')
plt.ylabel('$\log (te_k)$')
plt.title('log of training error versus k')

# Plot log testing error versus polynomial dimensions
m_test     = 1000
sigma_test = 0.07

X_test = np.random.rand(m_test, 1)
y_test = g(X_test.ravel(), sigma_test)

logtse = [] # log testing errors
for k in ks:
    Phi_test = transform(X_test, k)
    tse      = 1/m_test * np.sum((predict(Phi_test, ws[k-1]) - y_test) ** 2) # calculate testing error
    logtse.append(np.log(tse))

# Plot log testing error versus k
plt.figure()
plt.plot(ks, logtse)
plt.xlabel('k')
plt.ylabel('$\log (tse_k)$')
plt.title('log of testing error versus k')

# Repeat training and testing for 100 times
te_avg  = np.zeros(len(ks))
tse_avg = np.zeros(len(ks))

for i in range(100):    
    # Generate new training data
    X = np.random.rand(m, 1)
    y = g(X.ravel(), sigma)
    
    ws = [] # trained w
       
    for k in ks:
        Phi = transform(X, k)
        w   = fit(Phi, y)
        ws.append(w)
        
        te = 1/m * np.sum((predict(Phi, w) - y) ** 2) # calculate training error
        te_avg[k-1] += te
    
    # Generate new testing data
    X_test = np.random.rand(m_test, 1)
    y_test = g(X_test.ravel(), sigma_test)
    
    for k in ks:
        Phi_test = transform(X_test, k)
        tse      = 1/m_test * np.sum((predict(Phi_test, ws[k-1]) - y_test) ** 2) # calculate testing error
        tse_avg[k-1] += tse
    
te_avg  /= 100
tse_avg /= 100

logte_avg  = np.log(te_avg)
logtse_avg = np.log(tse_avg)

# Plot average training error
plt.figure()
plt.plot(ks, logte_avg)
plt.xlabel('k')
plt.ylabel('$\log (te_k)$')
plt.title('Log average training error')

# Plot average testing error
plt.figure()
plt.plot(ks, logtse_avg)
plt.xlabel('k')
plt.ylabel('$\log (tse_k)$')
plt.title('Log average testing error')

plt.show()