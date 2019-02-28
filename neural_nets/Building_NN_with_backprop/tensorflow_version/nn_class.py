import numpy as np
import tensorflow as tf
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
        
        self.D = D
        self.K = K

    
    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=1000, learning_rate=0.00001, batch_sz=100, reg=0.0):
        X = tf.placeholder(tf.float32, shape=(None, self.D))
        T = tf.placeholder(tf.float32, shape=(None, self.K))
        
        Z = tf.nn.relu( tf.matmul(X, self.W1) + self.b1 )
        Yish = tf.matmul(Z, self.W2) + self.b2
        
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))
        
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        
        predict_op = tf.argmax(Yish, axis=1)
        
        n_batches = len(Xtrain) // batch_sz
        
        costs = []
        errors = []
        init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(init)
            
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for j in range(n_batches):
                    Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
                    
                    session.run(train, feed_dict={X: Xbatch, T: Ybatch})
                    if j % 10 == 0:
                        test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest})
                        prediction = session.run(predict_op, feed_dict={X: Xtest, T: Ytest})
                        error = error_rate(prediction, np.argmax(Ytest, axis=1))
                        
                        costs.append(test_cost)
                        errors.append(error)
                        
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
    
    model = TensorflowNN(D, H, K)
    costs, errors = model.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=1000)
    
    create_graph(costs, 'cost.png', y_label='cost', x_label='iteration')
    create_graph(errors, 'error.png', y_label='error', x_label='iteration')

if __name__ == '__main__':
    main()

        
        
        
        
        