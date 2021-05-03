#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 03:38:26 2020
COMP0078 Supervised Learning
Question 4
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


#path     = '/Users/DouglasChiang/Google Drive_HNC/_My Study/_UCL_Master_Robotics and Computation/Courses/T1/COMP0078/Assignments/CW1/code/Boston-filtered.csv'
path     = 'Boston-filtered.csv'
raw_data = pd.read_csv(path).to_numpy() # Load data from csv
m, n     = raw_data.shape

m_train = 2 * m //3 # number of train data
m_test  = m - m_train # number of test data


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
    X_test  = np.ones((m_test, 1))
    y_train = train_data[:, -1]
    y_test  = test_data[:, -1]

    # Linear regression
    w = fit(X_train, y_train)
    
    # Find mean squared error for train and test
    mse_train = MSE(predict(X_train, w), y_train)
    mse_test  = MSE(predict(X_test, w), y_test)
    
    mse_trains.append(mse_train)
    mse_tests.append(mse_test)

print('')
print('Naive regression:')
print('MSE on train = %.3f' % np.mean(mse_trains))
print('MSE on test = %.3f' % np.mean(mse_tests))


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
        
        # !!!!!!!!!!Add bias term
        X_train = np.hstack((X_train, np.ones((m_train, 1))))
        X_test  = np.hstack((X_test, np.ones((m_test, 1))))
        
        # Linear regression
        w = fit(X_train, y_train)
        
        # Find mean squared error for train and test
        mse_train = MSE(predict(X_train, w), y_train)
        mse_test  = MSE(predict(X_test, w), y_test)
        
        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

print('')
print('Linear regression on single attributes:')
print('MSE on train = %.3f' % np.mean(mse_trains))
print('MSE on test = %.3f' % np.mean(mse_tests))


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
    X_test  = test_data[:, :-1]
    y_train = train_data[:, -1]
    y_test  = test_data[:, -1]
    
    # !!!!!!!!!!Add bias term
    X_train = np.hstack((X_train, np.ones((m_train, 1))))
    X_test  = np.hstack((X_test, np.ones((m_test, 1))))
    
    # Linear regression
    w = fit(X_train, y_train)
    
    # Find mean squared error for train and test
    mse_train = MSE(predict(X_train, w), y_train)
    mse_test  = MSE(predict(X_test, w), y_test)
    
    mse_trains.append(mse_train)
    mse_tests.append(mse_test)

print('')
print('Linear regression on all attributes:')
print('MSE on train = %.3f' % np.mean(mse_trains))
print('MSE on test = %.3f' % np.mean(mse_tests))