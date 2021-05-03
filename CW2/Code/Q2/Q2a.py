# ==========================
# COMP0078 Coursework 2 Q2.1
# Douglas Chiang
# 15055142
# ==========================
import time
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

def data_generate(m, n, winnow = False):
    """
    Generate data set X (size m x n) and label y (m x 1)
    """
    if winnow == False:
        X = np.random.choice([-1,1], (m, n))
        y = X[:,0]
    else:
        X = np.random.choice([0,1], (m, n))
        y = X[:,0]
    return X, y

#-----------1NN-----------------------------------
def One_NN(X, y, X_test):
    """
    Input:
    X: full dataset (m x n)
    y; labels (m x 1)
    X_test: data to be tested
    
    Output:
    y_pred: predictions
    """
    y_pred = np.zeros((X_test.shape[0],))
    for x_test_idx in range(X_test.shape[0]):
        X_test_expand      = np.full((X.shape[0], X_test.shape[1]), X_test[x_test_idx])
        # 1NN take the label of the closest sample.
        # If a sample is closest, it has the highest number of features agree with the testing sample:
        y_pred[x_test_idx] = y[np.argmax(np.sum(X_test_expand == X, axis = 1))]
    return y_pred
#-----------1NN-----------------------------------

#-----------Least Square--------------------------
def Least_Square(X, y):
    """
    Input:
    X: training dataset (m x n)
    y: training label (m x 1)

    Output:
    w: weight
    """
    #w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    w = np.linalg.pinv(X).dot(y)
    return w

def Least_Square_Predict(X, w):
    y_pred = X.dot(w)
    y_pred = (y_pred >= 0)*2 - 1
    return y_pred
#-----------Least Square--------------------------

#-----------Winnow--------------------------------
def Winnow(X, y):
    theta = X.shape[1]
    w = np.ones((theta,))
    for x_i in range(y.shape[0]):
        if w.dot(X[x_i]) >= theta:
            y_pred_i = 1
        else:
            y_pred_i = 0
        
        if y[x_i] == 1 and y_pred_i == 0:
            np.multiply(w, 2, out = w, where = (X[x_i] > 0))
        elif y[x_i] == 0 and y_pred_i == 1:
            np.divide(w, 2, out = w, where = (X[x_i] > 0))
    return w

def Winnow_Predict(X, w):
    y_pred = (np.matmul(X, w) >= n).astype(int)
    return y_pred
#-----------Winnow--------------------------------

#-----------Perceptron----------------------------
def Perceptron(X, y):
    """
    Input:
    X: training dataset (m (samples) x n (features))
    y: true labels

    Output:
    w: weight vector (n x 1)
    """
    m, n = X.shape
    w = np.zeros((n,))
    for i in range(m):
        # Predict
        y_pred = (w.dot(X[i,:]) > 0)*2 - 1
        # Update w:
        if y[i] != y_pred:
            w += y[i]*X[i,:]
    return w

def Perceptron_Predict(X, w):
    return (w.dot(X.T) > 0)*2 - 1

#-----------Perceptron----------------------------

if __name__ == "__main__":
    m_mem_mean = []
    m_mem_std  = []
    n_mem = []

    for n in range(1, 100):
        validate_X, validate_Y = data_generate(10000, n, winnow = False) #!!!
        m_list = []
        for trial in range(10):
            for m in range(1, 10000):
                train_X, train_Y = data_generate(m, n, winnow = False) #!!!
                
                # Algorithm
                #w = Winnow(train_X, train_Y)
                #w = Least_Square(train_X, train_Y)
                w = Perceptron(train_X, train_Y)

                # Prediction
                #prediction_Y = Winnow_Predict(validate_X, w)
                #prediction_Y = Least_Square_Predict(validate_X, w)
                prediction_Y = Perceptron_Predict(validate_X, w)

                # 1NN ONLY:
                # prediction_Y = One_NN(train_X, train_Y, validate_X)

                error_rate = np.sum(prediction_Y != validate_Y) / len(prediction_Y)
                if error_rate <= 0.1:
                    m_list.append(m)
                    break
        m_mem_mean.append(np.mean(m_list))
        m_mem_std.append(np.std(m_list))
        n_mem.append(n)
        last_m = int(np.mean(m_list))
        print(f"m mean = {np.mean(m_list)}, m std = {np.std(m_list)}, n = {n}")
    
    print(f"m_MEAN: {m_mem_mean}")
    print(f"m STD: {m_mem_std}")
    print(f"n: {n_mem}")