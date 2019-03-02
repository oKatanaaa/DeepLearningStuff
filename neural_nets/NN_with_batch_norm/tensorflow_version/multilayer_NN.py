import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from utils import error_rate
from utils import create_graph
from utils import one_hot_encoding

class HiddenBatchnormLayer(object):
    def __init__(self, M1, M2, f):
        self.M1 = M1
        self.M2 = M2
        # Activation function
        self.f = f
        W = np.random.randn(M1, M2)
        # Used for the batch shift(changes mean)
        beta = np.zeros(M2)
        # Used for changing variance
        gamma = np.ones(M2)
        
        
        self.W = tf.Variable(W.astype(np.float32))
        self.beta = tf.Variable(beta.astype(np.float32))
        self.gamma = tf.Variable(gamma.astype(np.float32))
        
        # For the test time
        self.running_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
        self.running_var = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
    
    def forward(self, X, is_training=False, decay=0.9):
        Z = tf.matmul(X, self.W)
        if is_training:
            batch_mean, batch_var = tf.nn.moments(Z, [0])
            update_running_mean = tf.assign(
                self.running_mean,
                self.running_mean*decay + (1 - decay)*batch_mean
            )
            update_running_var = tf.assign(
                self.running_var,
                self.running_var*decay + (1 - decay)*batch_var
            )
            with tf.control_dependencies([update_running_mean, update_running_var]):
                out = tf.nn.batch_normalization(
                    Z,
                    batch_mean,
                    batch_var,
                    self.beta,
                    self.gamma,
                    1e-4
                )
        else:
            out = tf.nn.batch_normalization(
                Z,
                self.running_mean,
                self.running_var,
                self.beta,
                self.gamma,
                1e-4
            )
        return self.f(out)
            

class HiddenLayer(object):
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2
        W = np.random.randn(M1, M2)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        
    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)
    
    
class MultiLayerNN(object):
    def __init__(self, D, K, layer_sizes, ps_keep):
        self.ps_keep = ps_keep
        
        self.hidden_layers = []
        M1 = D
        for M2 in layer_sizes:
            h = HiddenBatchnormLayer(M1, M2, tf.nn.relu)
            self.hidden_layers.append(h)
            M1 = M2
        
        W = np.random.randn(M1, K)
        b = np.zeros(K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        
        self.X = tf.placeholder(tf.float32, shape=(None, D))
        self.T = tf.placeholder(tf.float32, shape=(None, K))
            
    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.nn.softmax(tf.matmul(Z, self.W) + self.b)
    
    def forward_test(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b
    
    def forward_train(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z, is_training=True)
        return tf.matmul(Z, self.W) + self.b
    
    
    def fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate=0.0001, mu=0.9, decay=0.999, epochs=9, batch_sz=100):
        Xtrain = Xtrain.astype(np.float32)
        Ytrain = Ytrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.astype(np.float32)
        
        Yish = self.forward_train(self.X)
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=self.T))
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
        test_logits = self.forward_test(self.X)
        test_cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=test_logits, labels=self.T))
        predict_op = tf.argmax(test_logits, axis=1)
        
        n_batches = len(Xtrain) // batch_sz
        
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
                    test_cost = session.run(test_cost_op, feed_dict={self.X: Xbatch, self.T: Ybatch})
                    Yp = session.run(predict_op, feed_dict={self.X: Xtest})
                    error = error_rate(Yp, np.argmax(Ytest, axis=1))
                    costs.append(test_cost)
                    errors.append(error)
                    if j % 20 == 0:
                        print("Accuracy: ", 1 - error)
                        print("Cost: ", test_cost)
                        
        return costs, errors
    
    
def main():
    Nclass = 500
    D = 2
    X1 = np.random.randn(Nclass, D) + np.array([0, 2])
    X2 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 0])
    X = np.vstack([X1, X2, X3])
    
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    Y = one_hot_encoding(Y)
    
    
    N = 1500
    H = 10
    K = 3
    
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]
    
    model = MultiLayerNN(D, K, [10, 10], [0.8, 0.5])
    costs, errors = model.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=1000)
    
    create_graph(costs, 'cost.png', y_label='cost', x_label='iteration')
    create_graph(errors, 'error.png', y_label='error', x_label='iteration')

if __name__ == '__main__':
    main()