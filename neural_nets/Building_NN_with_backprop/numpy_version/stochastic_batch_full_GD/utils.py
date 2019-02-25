import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def get_data():
    df = pd.read_csv('/home/student401/study/data_sets/mnist/train.csv')
    data = df.values

    X = data[:, 1:]
    Y = data[:, 0]
    
    return X, Y

def one_hot_encoding(Y):
    N = len(Y)
    D = len(set(Y))
    new_Y = np.zeros((N, D))
    
    for i in range(N):
        new_Y[i, Y[i]] = 1
        
    return new_Y

def get_preprocessed_data():
    X, Y = get_data()
    X = (X - X.mean()) / X.std()
    Y = one_hot_encoding(Y)
    X, Y = shuffle(X, Y)
    return X, Y

def error_rate(Y, T):
    return np.mean(Y != T)