import tensorflow as tf
import numpy as np
from utils import get_preprocessed_data, error_rate, create_graph
from sklearn.utils import shuffle

def convpool(X, W, b):
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter(shape):
    w = np.random.randn(*shape) * np.sqrt(2.0 / np.prod(shape[:-1]))
    return w.astype(np.float32)

def rearrange(X):
    # input is (N, 784)
    # output is (N, 28, 28, 1)
    new_X = np.zeros((X.shape[0], 28, 28, 1))
    for j in range(28):
        new_X[:,j,:,0] += X[:,j*28:(j+1)*28]
        
    return new_X

def main():
    X, Y = get_preprocessed_data()
    X = rearrange(X)
    
    Xtrain, Ytrain = X[:-1000], Y[:-1000]
    Xtest, Ytest = X[-1000:], Y[-1000:]
    
    N, K = Y.shape
    batch_sz = 500
    n_batches = N // batch_sz
    
    epochs = 1000
    learning_rate = 0.00001
    decay = 0.999
    mu = 0.9
    # Hidden dense layer size
    hl_sz = 300
    
    # after conv will be dimension of 28 because we use SAME padding mode
    # after pooling 28 / 2 = 14
    W1_shape = (5, 5, 1, 20) # (filter_width, filter_height, num_color_channels, num_feature_maps)
    W1_init = init_filter(W1_shape)
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32)
    
    # after conv will be dimension of 14 because we use SAME padding mode
    # after pooling 14 / 2 = 7
    W2_shape = (5, 5, 20, 40) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
    W2_init = init_filter(W2_shape)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)
    
    W2_in_shape = (3, 3, 40, 50) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
    W2_in_init = init_filter(W2_in_shape)
    b2_in_init = np.zeros(W2_in_shape[-1], dtype=np.float32)
    
    # vanilla ANN weights
    W3_init = np.random.randn(W2_in_shape[-1]*7*7, hl_sz) * np.sqrt(2.0 / W2_in_shape[-1]*7*7)
    b3_init = np.zeros(hl_sz)
    W4_init = np.random.randn(hl_sz, K) * np.sqrt(2.0 / (hl_sz + K))
    b4_init = np.zeros(K)
    
    # define variables and expressions
    X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    T = tf.placeholder(tf.float32, shape=(None, K))
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W2_in = tf.Variable(W2_in_init.astype(np.float32))
    b2_in = tf.Variable(b2_in_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))
    
    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z2_in = tf.nn.conv2d(Z2, W2_in, strides=[1, 1, 1, 1], padding='SAME')
    Z2_out = tf.nn.relu(tf.nn.bias_add(Z2_in, b2_in))
    
    Z2_shape = Z2_out.get_shape().as_list()
    Z2r = tf.reshape(Z2_out, [-1 , np.prod(Z2_shape[1:]).astype(np.int32)])
    Z3 = tf.nn.relu( tf.matmul(Z2r, W3) + b3 )
    Yish = tf.matmul(Z3, W4) + b4
    
    cost = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=Yish,
            labels=T
        )
    )
    
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    # we'll use this ti calculate the error rate
    predict_op = tf.argmax(Yish, 1)
        
    costs = []
    errors = []
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        
        for i in range(epochs):
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
                Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
                
                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                cost_test = session.run(cost, feed_dict={X: Xtest, T: Ytest})
                predictions = session.run(predict_op, feed_dict={X: Xtest, T: Ytest})
                error = error_rate(predictions, np.argmax(Ytest, axis=1))
                
                costs.append(cost_test)
                errors.append(error)
                if j % 20 == 0:
                    print("Epoch/batch: {}/{}".format(i, j))
                    print("Accuracy: ", 1 - error)
                    print("Cost: ", cost_test)
    
    
    
    create_graph(errors, 'Cnntest_error_{}.png'.format(1 - errors[-1]), 'Error', 'Iterations')
    create_graph(costs, 'Cnntest_cost.png', 'Cost', 'Iterations')
    
    
    
if __name__ == '__main__':
    main()
    