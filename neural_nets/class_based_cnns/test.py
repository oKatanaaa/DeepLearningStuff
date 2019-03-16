from cnns.tensorflow_cnn import CNN
import numpy as np
from utils import get_dendrite_preprocessed_train_data_tf, error_rate, create_graph
from sklearn.utils import shuffle

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
    
    print('X shape is', X.shape)
    
    epochs = 200
    learning_rate = 1e-8
    N, K = Y.shape
    batch_sz = 150
    input_shape = Xtrain.shape[1:]
    conv_layer_sizes = [(3, 3, 32, False), (3, 3, 32, True),
                        (3, 3, 64, False), (3, 3, 64, True),
                        (3, 3, 128, False), (3, 3, 128, True),
                        (3, 3, 256, True)
                       ]
    dense_layer_sizes = [1000, 500, 200]
    
    
    model = CNN(input_shape, K, conv_layer_sizes, dense_layer_sizes, 150)
    
    costs, errors = model.limited_fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=learning_rate, mu=0.9, decay=0.99, epochs=epochs)
    create_graph(errors, 'Cnntest_error_{}.png'.format(1 - errors[-1]), 'Error', 'Iterations')
    create_graph(costs, 'Cnntest_cost.png', 'Cost', 'Iterations')
    
    
if __name__ == '__main__':
    main()