#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 02:32:58 2020
COMP0078 Supervised Learning
Question 5d
@author: Anthony, Douglas
"""


import pandas as pd
import numpy  as np

np.random.seed(0)

def fit(X, y):
    # fit data and output the fitted w
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w

def predict(X, w):
    # predict y from data and w
    return X.dot(w)

def MSE(y_pred, y):
    # Calculates the mean squared error between predicted y and actual y
    return np.mean((y_pred - y) ** 2)

path     = '/Users/DouglasChiang/Google Drive_HNC/_My Study/_UCL_Master_Robotics and Computation/Courses/T1/COMP0078/Assignments/CW1/code/Boston-filtered.csv'
#path     = 'Boston-filtered.csv'
raw_data = pd.read_csv(path).to_numpy() # Load data from csv
m, n     = raw_data.shape

m_train = 2 * m //3 # number of train data
m_test  = m - m_train # number of test data



print('Method \t\t\t\t\t\t MSE train \t\t MSE test')

##### Naive regression
mse_trains = []
mse_tests  = []
for i in range(20):
    # Shuffle order of data
    data = np.random.permutation(raw_data)
    
    # Create train test split
    train_data = data[:m_train, :]
    test_data  = data[m_train:, :]
    
    # Extract X and y
    X_train = np.ones((m_train, 1))
    y_train = train_data[:, -1]
    X_test  = np.ones((m_test, 1))
    y_test  = test_data[:, -1]

    # Linear regression
    w = fit(X_train, y_train)
    
    # Find mean squared error for train and test
    mse_train = MSE(predict(X_train, w), y_train)
    mse_test  = MSE(predict(X_test, w), y_test)
    
    mse_trains.append(mse_train)
    mse_tests.append(mse_test)

mse_train_mean = np.mean(mse_trains)
mse_train_std  = np.std(mse_trains)
mse_test_mean  = np.mean(mse_tests)
mse_test_std   = np.std(mse_tests)
print('Naive Regression \t\t\t %.2f +- %.2f \t %.2f +- %.2f'
      % (mse_train_mean, mse_train_std, mse_test_mean, mse_test_std))


##### Linear regression with single attributes
mse_trains = []
mse_tests  = []
for i in range(20):
    for j in range(n-1):
        # Shuffle order of data
        data = np.random.permutation(raw_data)
        
        # Create train test split
        train_data = data[:m_train, :]
        test_data  = data[m_train:, :]
        
        # Extract X and y
        X_train = train_data[:, j, np.newaxis]
        X_test  = test_data[:, j, np.newaxis]
        y_train = train_data[:, -1]
        y_test  = test_data[:, -1]
        
        # Add bias term
        X_train = np.hstack((X_train, np.ones((m_train, 1))))
        X_test  = np.hstack((X_test, np.ones((m_test, 1))))
        
        # Linear regression
        w = fit(X_train, y_train)
        
        # Find mean squared error for train and test
        mse_train = MSE(predict(X_train, w), y_train)
        mse_test  = MSE(predict(X_test, w), y_test)
        
        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

mse_trains = np.array(mse_trains).reshape(20, n-1)
mse_tests  = np.array(mse_tests).reshape(20, n-1)

mse_train_means = np.mean(mse_trains, axis=0)
mse_train_stds  = np.std(mse_trains, axis=0)
mse_test_means  = np.mean(mse_tests, axis=0)
mse_test_stds   = np.std(mse_tests, axis=0)

for i in range(n-1):
    print('Linear Regression %d \t\t\t %.2f +- %.2f \t %.2f +- %.2f'
          % (i+1, mse_train_means[i], mse_train_stds[i], mse_test_means[i], mse_test_stds[i]))


##### Linear regression with all attributes
mse_trains = []
mse_tests  = []
for i in range(20):
    # Shuffle order of data
    data = np.random.permutation(raw_data)
    
    # Create train test split
    train_data = data[:m_train, :]
    test_data  = data[m_train:, :]
    
    # Extract X and y
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test  = test_data[:, :-1]
    y_test  = test_data[:, -1]
    
    # Add bias term
    X_train = np.hstack((X_train, np.ones((m_train, 1))))
    X_test  = np.hstack((X_test, np.ones((m_test, 1))))
    
    # Linear regression
    w = fit(X_train, y_train)
    
    # Find mean squared error for train and test
    mse_train = MSE(predict(X_train, w), y_train)
    mse_test  = MSE(predict(X_test, w), y_test)
    
    mse_trains.append(mse_train)
    mse_tests.append(mse_test)

mse_train_mean = np.mean(mse_trains)
mse_train_std  = np.std(mse_trains)
mse_test_mean  = np.mean(mse_tests)
mse_test_std   = np.std(mse_tests)
print('Linear Regression all\t\t\t %.2f +- %.2f \t %.2f +- %.2f'
      % (mse_train_mean, mse_train_std, mse_test_mean, mse_test_std))


##### Kernel ridge regression
def kernel(Xi, Xj, sigma):
    # Calculates a Gaussian kernel matrix given data and sigma
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
    m = K.shape[0]
    
    assert m == K.shape[1]
    
    alpha = np.linalg.inv(K + gamma * m * np.eye(m)).dot(y)
    return alpha

def predict(K, alpha):
    # predict y from given kernel and alpha
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
    data_val = data[fold*size:(fold+1)*size]

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
        data_train = np.concatenate((data_train1, data_train2), axis=0)
    
    return data_train, data_val

folds      = 5
gammas     = [2 ** i for i in range(-40, -25)]
sigmas     = [2 ** (i/2) for i in range(14, 27)]
mse_trains = []
mse_tests  = []
for i in range(20):
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
    cves       = []

    for sigma in sigmas:
        for gamma in gammas:
            cve = 0
            # five fold cross-validation
            for fold in range(folds):
                # five fold split
                train_fold, val_fold = kFold(train_data, fold, folds)
                
                X_train_fold = train_fold[:, :-1]
                X_val_fold   = val_fold[:, :-1]
                y_train_fold = train_fold[:, -1]
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
    
    mse_trains.append(mse_train)
    mse_tests.append(mse_test)

mse_train_mean = np.mean(mse_trains)
mse_train_std  = np.std(mse_trains)
mse_test_mean  = np.mean(mse_tests)
mse_test_std   = np.std(mse_tests)

print('Kernel Ridge Regression\t\t %.2f +- %.2f \t %.2f +- %.2f'
      % (mse_train_mean, mse_train_std, mse_test_mean, mse_test_std))