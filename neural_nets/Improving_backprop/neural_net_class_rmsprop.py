import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.utils import shuffle
from utils import error_rate

# This is only one-hidden-layer ANN
class NeuralNetwork(object):
    def __init__(self, D, H, C):
        """
        D - dimensinality of the input layer
        H - size of the hidden layer
        C - number of classes
        """
        
        self.D = D
        self.H = H
        self.C = C
        
        self.W2 = np.random.randn(H, C)
        self.b2 = np.zeros(C)
        # Hidden layer
        self.W1 = np.random.randn(D, H)
        self.b1 = np.zeros(H)
        
    def softmax(self, X):
        exp_A = np.exp(X)
        return exp_A / np.sum(exp_A, axis=1, keepdims=True)
    
    def cost(self, Y, T):
        return -(T*np.log(Y)).sum() / len(Y)
    
    def score(self, Y, T):
        return np.mean(T == Y)
    
    def __derivative_W2(self, Z, T, Y):
        return Z.T.dot(T - Y)
    
    def __derivative_b2(self, Y, T):
        return (T - Y).sum(axis=0)
    
    def __derivative_W1(self, Z, X, T, Y):
        beta = (T - Y).dot(self.W2.T) * (1 - Z**2)
        return X.T.dot(beta)
    
    def __derivative_b1(self, Z, T, Y):
        return ((T - Y).dot(self.W2.T) * (1 - Z**2)).sum(axis=0)
    
    def forward(self, X):
        Z = np.tanh(X.dot(self.W1))
        return self.softmax(Z.dot(self.W2)), Z
        
    def fit(self, Xtrain, Ytrain, Xtest, Ytest, 
            batch_sz = 100, epochs=1000, learning_rate=0.001, reg=0.0,
           decay=0.999):
        
        costs_test = []
        costs_train = []
        errors = []
        
        n_batches = Xtrain.shape[0] // batch_sz
        # Create caches
        cache_W2 = 0
        cache_b2 = 0
        cache_W1 = 0
        cache_b1 = 0
        
        # Create epsilon constant in order to avoid division by zero
        eps = 1e-8
        
        for i in range(epochs):
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for j in range(n_batches):
                X_bt = Xtrain[j*batch_sz:(j+1)*batch_sz,:]
                Y_bt = Ytrain[j*batch_sz:(j+1)*batch_sz,:]
                Yp_tr, Z = self.forward(X_bt)
                Yp_ts, _ = self.forward(Xtest)

                Yp_tr_c, _ = self.forward(Xtrain)
                Yp_ts_c, _ = self.forward(Xtest)
                ctr = self.cost(Yp_tr_c, Ytrain)
                cts = self.cost(Yp_ts_c, Ytest)
                error = error_rate(np.argmax(Yp_tr_c, axis=1), np.argmax(Ytrain, axis=1))

                costs_test.append(cts)
                costs_train.append(ctr)
                errors.append(error)

                if i % 10 == 0:
                    print("Train score: ", 1 - error)
                    print("Train cost: ", ctr)
                    print("Test cost: ", cts)
                
                # Gradients
                der_W2 = self.__derivative_W2(Z, Y_bt, Yp_tr) + reg*self.W2
                der_b2 = self.__derivative_b2(Y_bt, Yp_tr) + reg*self.b2
                der_W1 = self.__derivative_W1(Z, X_bt, Y_bt, Yp_tr) + reg*self.W1
                der_b1 = self.__derivative_b1(Z, Y_bt, Yp_tr) + reg*self.b1
                
                # Updating caches
                cache_W2 = decay*cache_W2 + (1 - decay)*der_W2**2
                cache_b2 = decay*cache_b2 + (1 - decay)*der_b2**2
                cache_W1 = decay*cache_W1 + (1 - decay)*der_W1**2
                cache_b1 = decay*cache_b1 + (1 - decay)*der_b1**2
                
                # Updating weights
                self.W2 += learning_rate*der_W2/np.sqrt(cache_W2 + eps)
                self.b2 += learning_rate*der_b2/np.sqrt(cache_b2 + eps)
                self.W1 += learning_rate*der_W1/np.sqrt(cache_W1 + eps)
                self.b1 += learning_rate*der_b1/np.sqrt(cache_b1 + eps)
        
        # Create empty figure for future cost graph
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        # Add set of axes to the figure
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        
        # Plot costs
        axes.plot(costs_test, 'b', label='Test cost')
        axes.plot(costs_train, 'r', label='Train cost')
        axes.legend()
        axes.set_xlabel('Iteration')
        axes.set_ylabel('Cost')
        fig.savefig('adagrad_costs.png')
        
        # Create empty figure for future error graph
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(errors, 'r')
        axes.set_ylabel('Error rate')
        axes.set_xlabel('Iteration')
        fig.savefig('adagrad_error.png')
        return errors
                      
                      
def main():
    Nclass = 500
    D = 2
    H = 5
    K = 3
    
    X1 = np.random.randn(Nclass, D) - np.array([2, 0])
    X2 = np.random.randn(Nclass, D) - np.array([0, 2])
    X3 = np.random.randn(Nclass, D) - np.array([0, -2])
    
    X = np.vstack([X1, X2, X3])
    
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)
    
    T = np.zeros((N, K))
    
    # One-hot encodding
    for i in range(N):
        T[i, Y[i]] = 1
                      
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.savefig("fig.png")
    
    model = NeuralNetwork(D, H, K)
    model.fit(X[:-100], T[:-100], X[-100:], T[-100:])
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    