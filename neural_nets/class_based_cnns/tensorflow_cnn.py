import tensorflow as tf
import numpy as np
from utils import get_preprocessed_image_data_tf, error_rate, create_graph
from sklearn.utils import shuffle

class ConvPoolLayer(object):
    def __init__(self, fw, fh, im, om, pool=False):
        """
        im - maps input
        om - maps output
        fw - filter width
        fh - filter height
        """
        # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
        self.shape = (fw, fh, im, om)
        self.pool = pool
        W = np.random.randn(*self.shape) * np.sqrt(2.0 / np.prod(self.shape[:-1]))
        W = W.astype(np.float32)
        b = np.zeros(om, dtype=np.float32)
        
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)
        
        
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        #print('Before conv', X.shape, 'After conv:', conv_out.get_shape())
        conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.pool:
            conv_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')
            #print('Before pool', conv_out.get_shape(), 'After pool:', conv_out.get_shape())
        return tf.nn.relu(conv_out)
    
    
class HiddenLayer(object):
    def __init__(self, M1, M2):
        W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
        W = W.astype(np.float32)
        b = np.zeros(M2)
        b = b.astype(np.float32)
        
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)

        
    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)
        

class CNN(object):
    def __init__(self, input_shape, K, conv_layer_sizes, hidden_layer_sizes, batch_sz):
        """
        input_shape - tuple with sizes of the input image. Data: (image_width, image_height, color_channels)
        K - number of classes. Data: int
        conv_layer_sizes - list of tuples with kernel sizes. Data: [(kernel_width=int, kernel_height=int, number_feature_maps=int, pool=bool)]
        hidden_layer_sizes - list of hidden layer sizes. Data: [int, int, ...]
        """
        self.conv_layer_sizes = conv_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_sz = batch_sz
        
        w, h, c = input_shape
        self.X = tf.placeholder(tf.float32, shape=(batch_sz, w, h, c), name='X')
        self.T = tf.placeholder(tf.float32, shape=(batch_sz, K), name='Y')
        
        # Init conv layers
        im = c # input maps
        self.conv_layers = []
        for k_w, k_h, om, pool in conv_layer_sizes:
            # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
            conv_layer = ConvPoolLayer(k_w, k_h, im, om, pool)
            self.conv_layers.append(conv_layer)
            
            im = om
            if pool:
                if w % 2 == 0:
                    w = w // 2
                    h = h // 2
                else:
                    w = w // 2 + 1
                    h = h // 2 + 1
        print('Final shape h:', h)
        # Init hidden layers
        self.hidden_layers = []
        M1 = w*h*im
        
        for M2 in hidden_layer_sizes:
            hidden_layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(hidden_layer)
            M1 = M2
        
        # Init classification layer
        W = np.random.randn(M1, K)  * np.sqrt(2.0 / np.prod(M1 + K))
        W = W.astype(np.float32)
        b = np.zeros(K).astype(np.float32)
        
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)
        
        
        
    def forward_train(self, X):
        Z = X
        for c in self.conv_layers:
            Z = c.forward(Z)
            #print(Z.get_shape().as_list())
            
        Z_shape = Z.get_shape().as_list()
        print('Final Z shape:', Z_shape)
        Z = tf.reshape(Z, [-1 , np.prod(Z_shape[1:])])
        
        for h in self.hidden_layers:
            Z = h.forward(Z)
            
        return tf.matmul(Z, self.W) + self.b
    
    
    def forward(self, X):
        Z = self.forward_train(X)
        return tf.nn.softmax(Z)
    
    
    def fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate=0.0001, mu=0.9, decay=0.999, epochs=9, batch_sz=100):
        Xtrain = Xtrain.astype(np.float32)
        Ytrain = Ytrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.astype(np.float32)
        
        
        Yish = self.forward_train(self.X)
        cost = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=self.T) )
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
        
        predict_op = tf.argmax(Yish, 1)
        
        n_batches = Xtrain.shape[0] // batch_sz
        
        init_op = tf.global_variables_initializer()
        costs = []
        errors = []
        
        with tf.Session() as session:
            session.run(init_op)
            
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for j in range(n_batches):
                    Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
                    
                    session.run(train_op, feed_dict={self.X: Xbatch, self.T: Ybatch})
                    test_cost = session.run(cost, feed_dict={self.X: Xbatch, self.T: Ybatch})
                    predictions = session.run(predict_op, feed_dict={self.X: Xtest})
                    error = error_rate(predictions, np.argmax(Ytest, axis=1))
                    costs.append(test_cost)
                    errors.append(error)
                    if j % 20 == 0:
                        print('Epoch:', i, 'Accuracy:', 1 - error, 'Cost:', test_cost)
                        
        return costs, errors
    
    
    def limited_fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate=0.0001, mu=0.9, decay=0.999, epochs=9):
        Xtrain = Xtrain.astype(np.float32)
        Ytrain = Ytrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.astype(np.float32)
        
        
        Yish = self.forward_train(self.X)
        cost = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=self.T) )
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
        
        predict_op = tf.argmax(Yish, 1)
        
        n_batches = Xtrain.shape[0] // self.batch_sz
        
        init_op = tf.global_variables_initializer()
        costs = []
        errors = []
        
        with tf.Session() as session:
            session.run(init_op)
            
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for j in range(n_batches):
                    Xbatch = Xtrain[j*self.batch_sz:(j+1)*self.batch_sz]
                    Ybatch = Ytrain[j*self.batch_sz:(j+1)*self.batch_sz]
                    
                    session.run(train_op, feed_dict={self.X: Xbatch, self.T: Ybatch})
                    
                    if j % 20 == 0:
                        test_cost = 0
                        predictions = np.zeros(len(Xtest))
                        for k in range(len(Xtest // batch_sz)):
                            Xtestbatch = Xtest[k*batch_sz:(k+1)*batch_sz]
                            Ytestbatch = Ytest[k*batch_sz:(k+1)*batch_sz]
                            test_cost += session.run(cost, feed_dict={self.X: Xtestbatch, self.T: Ytestbatch})
                            predictions[k*batch_sz:(k+1)*batch_sz] = session.run(
                                predict_op, feed_dict={self.X: Xtestbatch})
                        error = error_rate(predictions, np.argmax(Ytestbatch, axis=1))
                        costs.append(test_cost)
                        errors.append(error)
                        print('Epoch:', i, 'Accuracy:', 1 - error, 'Cost:', test_cost)
                        
        return costs, errors
        
        
        
        
def main():    
    X, Y = get_preprocessed_image_data_tf()
    Xtrain, Ytrain = X[:-1000], Y[:-1000]
    Xtest, Ytest = X[-1000:], Y[-1000:]
    print(X.shape)
    
    epochs = 200
    learning_rate = np.float32(0.00001)
    N = Xtrain.shape[0]
    batch_sz = 500
    
    model = CNN(input_shape=(28, 28, 1), K=10,
                conv_layer_sizes=[
                    (3, 3, 64, False), (3, 3, 64, True),
                     (3, 3, 128, False), (3, 3, 128, True),
                     (3, 3, 256, False), (3, 3, 256, True)],
                hidden_layer_sizes=[1000, 500, 200],
                batch_sz
               )
    costs, errors = model.limited_fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=1e-6, mu=0.4, decay=0.99, epochs=epochs)
    create_graph(errors, 'Cnntest_error_{}.png'.format(1 - errors[-1]), 'Error', 'Iterations')
    create_graph(costs, 'Cnntest_cost.png', 'Cost', 'Iterations')
    
if __name__ == '__main__':
    main()               
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        