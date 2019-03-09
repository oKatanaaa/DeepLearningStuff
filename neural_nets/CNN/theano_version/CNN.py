import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor.signal import pool
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from utils import get_preprocessed_data, error_rate, create_graph

def convpool(X, W, b, poolsize=(2,2)):
    conv_out = T.nnet.conv2d(input=X, filters=W)
    
    pooled_out = pool.pool_2d(
        input=conv_out,
        ws=poolsize,
        ignore_border=True
    )
    
    # Add the bias term. Since the bias is a vector 1D array), we first
    # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
    # thus be broadcasted across mini_batches and feature map
    return T.nnet.relu(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))


def init_filter(shape):
    w = np.random.randn(*shape) / np.sqrt(2.0 / np.prod(shape[1:]))
    return w.astype(np.float32)

def rearrange(X):
    # input is (N, 784)
    # output is (N, 28, 28)
    new_X = np.zeros((X.shape[0], 1, 28, 28))
    for pixels_row, new_X_mat in zip(X, new_X):
        for j in range(28):
            new_X_mat[0, j] += pixels_row[j*28:(j+1)*28]

    return new_X.astype(np.float32)


def main():
    X_d, Y = get_preprocessed_data()
    X_d = rearrange(X_d)
    Y = Y.astype(np.float32)
    Xtrain, Ytrain = X_d[:-1000], Y[:-1000]
    Xtest, Ytest = X_d[-1000:], Y[-1000:]
    
    
    epochs = 150
    learning_rate = np.float32(0.00001)
    N = Xtrain.shape[0]
    batch_sz = 500
    n_batches = N // batch_sz
    
    poolsz = (2, 2)
    K = 10
    # Hidden layer size
    hl_sz = 300
    # Momentum factor
    mu = np.float32(0.9)
    
    
    # after conv will be of dimension 28 - 5 + 1 = 24
    # after downsample 24 / 2 = 12
    W1_shape = (30, 1, 5, 5) # (num_feature_maps, num_color_channels, filter_width, filter_height)
    W1_init = init_filter(W1_shape)
    b1_init = np.zeros(W1_shape[0], dtype=np.float32) # one bias per output feature map
    
    
    # after conv will be of dimension 14 - 5 + 1 = 8
    # after dimension 8 / 2 = 4
    W2_shape = (60, 30, 5, 5) # (num_feature_maps, num_color_channels, filter_width, filter_height)
    W2_init = init_filter(W2_shape)
    b2_init = np.zeros(W2_shape[0], dtype=np.float32) 
    
    # vanilla ANN weights
    W3_init = (np.random.randn(W2_shape[0]*4*4, hl_sz) / np.sqrt(W2_shape[0]*4*4 + hl_sz)).astype(np.float32)
    b3_init = np.zeros(hl_sz, dtype=np.float32)
    W4_init = (np.random.randn(hl_sz, K) / np.sqrt(hl_sz + K)).astype(np.float32)
    b4_init = np.zeros(K, dtype=np.float32)
    
    # Define theano variables
    
    X = T.tensor4('X', dtype='float32')
    Y = T.matrix('T')
    W1 = th.shared(W1_init, 'W1')
    b1 = th.shared(b1_init, 'b1')
    W2 = th.shared(W2_init, 'W2')
    b2 = th.shared(b2_init, 'b2')
    W3 = th.shared(W3_init, 'W3')
    b3 = th.shared(b3_init, 'b3')
    W4 = th.shared(W4_init, 'W4')
    b4 = th.shared(b4_init, 'b4')
    
    # forward pass
    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z3 = T.nnet.relu(Z2.flatten(ndim=2).dot(W3) + b3)
    Yp = T.nnet.softmax(Z3.dot(W4) + b4)
    
    # define the cost function
    cost = -(Y*T.log(Yp)).mean()
    prediction = T.argmax(Yp, axis=1)
    
    # training expressions and functions
    
    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    
    # momentum change
    
    dparams = [
        th.shared(
            np.zeros_like(
                p.get_value(),
                dtype=np.float32
            )
        ) for p in params
    ]
    updates = []
    grads = T.grad(cost, params)
    for p, g, dp in zip(params, grads, dparams):
        dp_update = learning_rate*g
        p_update = p - dp_update
        
        updates.append((dp, dp_update))
        updates.append((p, p_update))

    
    train = th.function(
        inputs=[X, Y],
        updates=updates,
    )
    
    get_prediction = th.function(
        inputs=[X, Y],
        outputs=[cost, prediction]
    )
    
    costs = []
    errors = []
    for i in range(epochs):
        Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
            Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
            
            train(Xbatch, Ybatch)
            cost_val, pred_val = get_prediction(Xtest, Ytest)
            costs.append(cost_val)
            error = error_rate(pred_val, np.argmax(Ytest, axis=1))
            errors.append(error)
            if j % 20 == 0:
                print('Accuracy: ', 1 - error)
                print('Cost: ', cost_val)
                
    
    create_graph(errors, 'Cnntest_error_{}.png'.format(1 - errors[-1]), 'Error', 'Iterations')
    create_graph(costs, 'Cnntest_cost.png', 'Cost', 'Iterations')
    
if __name__ == '__main__':
    main()
        