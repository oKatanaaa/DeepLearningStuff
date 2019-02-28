import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from utils import error_rate
import matplotlib.pyplot as plt 
from utils import create_graph
from utils import one_hot_encoding

class TensorflowNN(object):
    def __init__(self, D, H, K):
        W1_init = np.random.randn(D, H) / np.sqrt(D)
        b1_init = np.zeros(H)
        W2_init = np.random.randn(H, K) / np.sqrt(H)
        b2_init = np.zeros(K) 
        
        self.W1 = tf.Variable(W1_init.astype(np.float32))
        self.b1 = tf.Variable(b1_init.astype(np.float32))
        self.W2 = tf.Variable(W2_init.astype(np.float32))
        self.b2 = tf.Variable(b2_init.astype(np.float32))
        
        self.X = tf.placeholder(tf.float32, shape=(None, D))
        self.T = tf.placeholder(tf.float32, shape=(None, K))
        
        self.Z = tf.nn.relu( tf.matmul(self.X, self.W1) + self.b1)
        self.Yish = tf.matmul(self.Z, self.W2) + self.b2
        
        
    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=1000, learning_rate=0.00001, batch_sz=100, reg=0.0, decay=0.999):
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Yish, labels=self.T))
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(cost)
        predict_op = tf.argmax(self.Yish, axis=1)
        
        n_batches = len(Xtrain) // batch_sz
        
        init_op = tf.global_variables_initializer()
        costs, errors = [], [] 
        with tf.Session() as session:
            session.run(init_op)
            
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for j in range(n_batches):
                    Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
                    
                    session.run(train_op, feed_dict={self.X: Xbatch, self.T: Ybatch})
                    test_cost = session.run(cost, feed_dict={self.X: Xtest, self.T: Ytest})
                    predictions = session.run(predict_op, feed_dict={self.X: Xtest})
                    error = error_rate(predictions, np.argmax(Ytest, axis=1))

                    costs.append(test_cost)
                    errors.append(error)
                    
                    if j % 20 == 0:
                        
                        
                        print("Test cost: ", test_cost)
                        print("Test accuracy: ", 1 - error)
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
    
    model = TensorflowNN(D, H, K)
    costs, errors = model.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=1000)
    
    create_graph(costs, 'cost.png', y_label='cost', x_label='iteration')
    create_graph(errors, 'error.png', y_label='error', x_label='iteration')

if __name__ == '__main__':
    main()
                        
                        
                        
                        