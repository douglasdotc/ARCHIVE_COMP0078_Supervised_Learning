#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:56:07 2020
COMP0078 Supervised Learning
Question 5 (a - c)
@author: Anthony, Douglas
"""


import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

np.random.seed(0)

"""
def kernel2(Xi, Xj, sigma):
    # Calculates a Gaussian kernel matrix given data and sigma
    mi  = Xi.shape[0]
    mj = Xj.shape[0]
    K = np.zeros((mi, mj))
    
    for i in range(mi):
        for j in range(mj):
            K[i, j] = np.sum((Xi[i, :] - Xj[j, :]) ** 2)
    
    K = np.exp(-K/(2 * sigma**2))
    
    return K
"""

def kernel(Xi, Xj, sigma):
    # Calculates a Gaussian kernel matrix given data and sigma
    # K(x_i, x_j) = exp(-norm(x_i - x_j)^2/(2*sigma^2))
    mi = Xi.shape[0]
    mj = Xj.shape[0]
    
    xi2 = np.sum(Xi ** 2, axis=1)
    xj2 = np.sum(Xj ** 2, axis=1)
    
    xi2 = xi2.reshape((mi, 1))
    xj2 = xj2.reshape((1, mj))
    
    K = xi2 + xj2 - 2 * Xi.dot(Xj.T)
    K = np.exp(-K/(2 * sigma**2))

    return K

def fit(K, y, gamma):
    # fit kernel and output the fitted alpha
    m         = K.shape[0]
    assert m == K.shape[1]
    
    # alpha = (K + gamma*l*I)^{-1}*y
    alpha = np.linalg.inv(K + gamma * m * np.eye(m)).dot(y)
    return alpha

def predict(K, alpha):
    # predict y from given kernel and alpha
    # y_test = K^T*alpha
    return K.T.dot(alpha)

def MSE(y_pred, y):
    # Calculates the mean squared error between predicted y and actual y
    return np.mean((y_pred - y) ** 2)

def kFold(data, fold, folds):
    # Return split data sets for k-fold validation
    
    if fold > folds - 1:
        raise Exception('fold number exceeds total folds available')
    
    m        = data.shape[0]
    size     = m // folds
    data_val = data[fold*size:(fold + 1)*size]

    if fold == 0:
        # first fold
        data_train = data[(fold+1)*size:]

    elif fold == folds - 1:
        # last fold
        data_train = data[:fold*size]

    else:
        # middle fold
        data_train1 = data[:fold*size]
        data_train2 = data[(fold+1)*size:]
        data_train  = np.concatenate((data_train1, data_train2), axis=0)
    
    return data_train, data_val

#path     = '/Users/DouglasChiang/Google Drive_HNC/_My Study/_UCL_Master_Robotics and Computation/Courses/T1/COMP0078/Assignments/CW1/code/Boston-filtered.csv'
path     = 'Boston-filtered.csv'
raw_data = pd.read_csv(path).to_numpy() # Load data from csv
m, n     = raw_data.shape

m_train = 2 * m //3 # number of train data
m_test  = m - m_train # number of test data
folds   = 5

gammas = [2 ** i for i in range(-40, -25)]
sigmas = [2 ** (i/2) for i in range(14, 27)]

# Shuffle order of data
data = np.random.permutation(raw_data)

# Create train test split
train_data = data[:m_train, :]
test_data  = data[m_train:, :]

# Extract X and y
X_train = train_data[:, :-1]
X_test  = test_data[:, :-1]
y_train = train_data[:, -1]
y_test  = test_data[:, -1]


# Generate dummy best cross validation error
alpha      = np.zeros(X_train.shape[0])
K          = kernel(X_train, X_train, sigma=1)
cve_best   = MSE(predict(K, alpha), y_train)
gamma_best = -1
sigma_best = -1

# Cross validation errors
cves       = []
for sigma in sigmas:
    for gamma in gammas:
        cve = 0
        # five fold cross-validation
        for fold in range(folds):
            # five fold split
            train_fold, val_fold = kFold(train_data, fold, folds)
            
            X_train_fold = train_fold[:, :-1]
            y_train_fold = train_fold[:, -1]
            X_val_fold   = val_fold[:, :-1]
            y_val_fold   = val_fold[:, -1]
            
            # Kernel ridge regression
            K     = kernel(X_train_fold, X_train_fold, sigma)
            alpha = fit(K, y_train_fold, gamma)
            K_val = kernel(X_train_fold, X_val_fold, sigma)
            mse   = MSE(predict(K_val, alpha), y_val_fold)
            
            cve  += mse
        
        # Calculate cross validation error
        cve /= folds
        cves.append(cve)
        # Store best parameters
        if cve < cve_best:
            cve_best   = cve
            gamma_best = gamma
            sigma_best = sigma
        

# Use best parameters to train and test
K          = kernel(X_train, X_train, sigma_best)
K_test     = kernel(X_train, X_test, sigma_best)

alpha_best = fit(K, y_train, gamma_best)

# Find mean squared error on train and test
mse_train = MSE(predict(K, alpha_best), y_train)
mse_test  = MSE(predict(K_test, alpha_best), y_test)

print('best gamma = 2^%d, best sigma = 2^%.2f' % (np.log2(gamma_best), np.log2(sigma_best)))
print('best parameters train MSE = %.3f' % (mse_train))
print('best parameters test MSE = %.3f' % (mse_test))

# Plot of cross validation error as a function of gamma and sigma
cves    = np.array(cves).reshape(len(sigmas), len(gammas))
logcves = np.log(cves)
plt.figure()
for sigma, logcve_sigma in zip(sigmas, logcves):
    plt.plot(np.log2(gammas), logcve_sigma, label='$\log_2 \sigma = %.1f$' % np.log2(sigma))

plt.xlim([-40, -26])
plt.legend()
plt.xlabel('$\log_2 \gamma$')
plt.ylabel('log of cross validation error')

plt.show()