import theano.tensor as T
import numpy as np
import theano
from theano.tensor.nnet import relu
from theano.tensor.nnet import softmax
from sklearn.utils import shuffle
from utils import error_rate
import matplotlib.pyplot as plt 
from utils import create_graph
from utils import one_hot_encoding

class TheanoNN(object):
    
    def __init__(self, D, H, K):
        W1_init = np.random.randn(D, H) / np.sqrt(D)
        b1_init = np.zeros(H)
        W2_init = np.random.randn(H, K) / np.sqrt(H)
        b2_init = np.zeros(K)
        
        self.W1 = theano.shared(W1_init, 'W1')
        self.b1 = theano.shared(b1_init, 'b1')
        self.W2 = theano.shared(W2_init, 'W2')
        self.b2 = theano.shared(b2_init, 'b2')
        
        # Define holders for Input-X and Targets-T
        self.thX = T.matrix('X')
        self.thT = T.matrix('T')
        
        # Define layer outputs
        self.thZ = relu(self.thX.dot(self.W1) + self.b1)
        self.thY = softmax(self.thZ.dot(self.W2) + self.b2)
        
        # Define cost function
        self.cost = -(self.thT * T.log(self.thY)).sum()
        self.prediction = T.argmax(self.thY, axis=1)
        
        # Define function which is helpful for the training
        self.get_prediction = theano.function(
            inputs=[self.thX, self.thT],
            outputs=[self.cost, self.prediction]
        )

    
    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=1000, learning_rate=0.00001, batch_sz=100, reg=0.0, mu=0.9, decay=0.999):
        # Create 1st momentum variables
        m_W1 = theano.shared(np.zeros_like(self.W1.get_value()))
        m_b1 = theano.shared(np.zeros_like(self.b1.get_value()))
        m_W2 = theano.shared(np.zeros_like(self.W2.get_value()))
        m_b2 = theano.shared(np.zeros_like(self.b2.get_value()))
        
        updated_m_W1 = mu*m_W1 + (1 - mu)*T.grad(self.cost, self.W1)
        updated_m_b1 = mu*m_b1 + (1 - mu)*T.grad(self.cost, self.b1)
        updated_m_W2 = mu*m_W2 + (1 - mu)*T.grad(self.cost, self.W2)
        updated_m_b2 = mu*m_b2 + (1 - mu)*T.grad(self.cost, self.b2)
        
        # Create 2st momentum variables
        v_W1 = theano.shared(np.zeros_like(self.W1.get_value()))
        v_b1 = theano.shared(np.zeros_like(self.b1.get_value()))
        v_W2 = theano.shared(np.zeros_like(self.W2.get_value()))
        v_b2 = theano.shared(np.zeros_like(self.b2.get_value()))
        
        updated_v_W1 = decay*v_W1 + (1 - decay)*T.grad(self.cost, self.W1)**2
        updated_v_b1 = decay*v_b1 + (1 - decay)*T.grad(self.cost, self.b1)**2
        updated_v_W2 = decay*v_W2 + (1 - decay)*T.grad(self.cost, self.W2)**2
        updated_v_b2 = decay*v_b2 + (1 - decay)*T.grad(self.cost, self.b2)**2
        
        # Create epsilons to avoid division by zero
        eps_W1 = theano.shared(np.ones_like(self.W1.get_value())/10e8)
        eps_b1 = theano.shared(np.ones_like(self.b1.get_value())/10e8)
        eps_W2 = theano.shared(np.ones_like(self.W2.get_value())/10e8)
        eps_b2 = theano.shared(np.ones_like(self.b2.get_value())/10e8)
        
        # Create bias correction variables
        mbc = theano.shared(mu)
        vbc = theano.shared(decay)
        
        # Correct 1st momentum
        correction_one = 1 - mbc
        hat_m_W1 = updated_m_W1 / correction_one
        hat_m_b1 = updated_m_b1 / correction_one
        hat_m_W2 = updated_m_W2 / correction_one
        hat_m_b2 = updated_m_b2 / correction_one
        
        # Correct 2nd momentum
        correction_two = 1 - vbc
        hat_v_W1 = updated_v_W1 / correction_two
        hat_v_b1 = updated_v_b1 / correction_two
        hat_v_W2 = updated_v_W2 / correction_two
        hat_v_b2 = updated_v_b2 / correction_two
        
        # Update bias correction variables
        updated_mbc = mbc*mu
        updated_vbc = vbc*decay
        
        
        updated_W1 = self.W1 - learning_rate*hat_m_W1 / T.sqrt(eps_W1 + hat_v_W1)
        updated_b1 = self.b1 - learning_rate*hat_m_b1 / T.sqrt(eps_b1 + hat_v_b1)
        updated_W2 = self.W2 - learning_rate*hat_m_W2 / T.sqrt(eps_W2 + hat_v_W2)
        updated_b2 = self.b2 - learning_rate*hat_m_b2 / T.sqrt(eps_b2 + hat_v_b2)
        
        
        
        train = theano.function(
            inputs=[self.thX, self.thT],
            updates=[
                # Update 1st momentum
                (m_W1, updated_m_W1), (m_b1, updated_m_b1), (m_W2, updated_m_W2), (m_b2, updated_m_b2),
                # Update 2nd momentum
                (v_W1, updated_v_W1), (v_b1, updated_v_b1), (v_W2, updated_v_W2), (v_b2, updated_v_b2),
                # Update weights
                (self.W1, updated_W1), (self.b1, updated_b1), (self.W2, updated_W2), (self.b2, updated_b2),
                # Update bias correction variables
                (mbc, updated_mbc), (vbc, updated_vbc)
            ]
        )
        
        n_batches = len(Xtrain) // batch_sz
        # Create this variable in order to get cost per vector
        cost_divider = len(Xtest)
        
        costs = []
        errors = []
        for i in range(epochs):
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
                Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
                
                train(Xbatch, Ybatch)
                cost, Yp = self.get_prediction(Xtest, Ytest)
                error = error_rate(Yp, np.argmax(Ytest, axis=1))
                
                costs.append(cost/cost_divider)
                errors.append(error)
                if j % 10 == 0:
                    print('Accuracy: ', 1 - error)
                    print('Cost: ', cost/cost_divider)
                    
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
    
    model = TheanoNN(D, H, K)
    costs, errors = model.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=1000)
    
    create_graph(costs, 'cost.png', y_label='cost', x_label='iteration')
    create_graph(errors, 'error.png', y_label='error', x_label='iteration')

if __name__ == '__main__':
    main()

        
        
        
        
        