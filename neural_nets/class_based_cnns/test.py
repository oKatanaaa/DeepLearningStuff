from cnns.resnet import layers
from cnns.resnet.DenNet import DenNet
import numpy as np
from utils import get_dendrite_preprocessed_train_data_tf, error_rate, create_graph
from sklearn.utils import shuffle
from utils import get_preprocessed_image_data_tf, error_rate, create_graph

def main():
    X, Y = get_dendrite_preprocessed_train_data_tf()
    # Get training pieces
    # x 0
    # 0 0
    X1, Y1 = X[:-1000, 0:100, 0:100], Y[:-1000]
    # 0 x
    # 0 0
    X2, Y2 = X[:-1000, 100:200, 0:100], Y[:-1000]
    # 0 0
    # x 0
    X3, Y3 = X[:-1000, 0:100, 100:200], Y[:-1000]
    # 0 0
    # 0 x
    X4, Y4 = X[:-1000, 100:200, 100:200], Y[:-1000]
    
    # Stack all the training Xs and Ys
    Xtrain, Ytrain = np.vstack([X1, X2, X3, X4]), np.vstack([Y1, Y2, Y3, Y4])
    
    # Get testing pieces
    # x 0
    # 0 0
    X1, Y1 = X[-1000:, 0:100, 0:100], Y[-1000:]
    # 0 x
    # 0 0
    X2, Y2 = X[-1000:, 100:200, 0:100], Y[-1000:]
    # 0 0
    # x 0
    X3, Y3 = X[-1000:, 0:100, 100:200], Y[-1000:]
    # 0 0
    # 0 x
    X4, Y4 = X[-1000:, 100:200, 100:200], Y[-1000:]
    
    # Stack all the training Xs and Ys
    Xtest, Ytest = np.vstack([X1, X2, X3, X4]), np.vstack([Y1, Y2, Y3, Y4])
    
    print('X shape is', Xtrain.shape)
    
    epochs = 300
    learning_rate = 1e-6
    N, K = Y.shape
    batch_sz = 128
    input_shape = Xtrain.shape[1:]
    
    
    model = DenNet(batch_sz)
    
    costs, errors = model.limited_fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=learning_rate, mu=0.9, decay=0.99, epochs=epochs)
    create_graph(errors, 'Cnntest_error_{}.png'.format(1 - errors[-1]), 'Error', 'Iterations')
    create_graph(costs, 'Cnntest_cost.png', 'Cost', 'Iterations')
    
    
if __name__ == '__main__':
    main()