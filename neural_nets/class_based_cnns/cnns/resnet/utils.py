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
    X = X / 255
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

def rearrange(X):
    # input is (N, 784)
    # output is (N, 28, 28)
    new_X = np.zeros((X.shape[0], 1, 28, 28))
    for pixels_row, new_X_mat in zip(X, new_X):
        for j in range(28):
            new_X_mat[0, j] += pixels_row[j*28:(j+1)*28]

    return new_X.astype(np.float32)

def rearrange_tf(X):
    # input is (N, 784)
    # output is (N, 28, 28, 1)
    new_X = np.zeros((X.shape[0], 28, 28, 1))
    for j in range(28):
        new_X[:,j,:,0] += X[:,j*28:(j+1)*28]
        
    return new_X
    
    
def get_preprocessed_image_data():
    X, Y = get_preprocessed_data()
    X = rearrange(X)
    return X, Y

def get_preprocessed_image_data_tf():
    X, Y = get_preprocessed_data()
    X = rearrange_tf(X)
    return X, Y