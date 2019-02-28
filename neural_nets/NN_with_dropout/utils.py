import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
plt.switch_backend('agg')

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

def create_graph(value, name, y_label, x_label):
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(value)
    axes.set_ylabel(y_label)
    axes.set_xlabel(x_label)
    
    fig.savefig(name)