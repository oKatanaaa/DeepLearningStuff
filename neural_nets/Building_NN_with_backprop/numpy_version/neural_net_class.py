import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.utils import shuffle
from stochastic_batch_full_GD.utils import error_rate

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
        return (Y - T).sum(axis=0)
    
    def __derivative_W1(self, Z, X, T, Y):
        N = len(X)
        D = self.D
        H = self.H
        C = self.C
        
        #der_W1 = np.zeros((D, H))
        #for d in range(D):
        #    for h in range(H):
        #        for n in range(N):
        #            for g in range(C):
        #                der_W1[d, h] += (1 - Z[n,h]**2)*X[n,d]*self.W2[h,g]*(T[n,g]-Y[n,g])
        beta = (T - Y).dot(self.W2.T) * (1 - Z**2)
        return X.T.dot(beta)
    
    def __derivative_b1(self, Z, T, Y):
        return ((Y - T).dot(self.W2.T) * (1 - Z**2)).sum(axis=0)
    
    def forward(self, X):
        Z = np.tanh(X.dot(self.W1))
        return self.softmax(Z.dot(self.W2)), Z
        
    def fit(self, X, Y, batch_sz = 100, epochs=1000, learning_rate=0.00001, reg=0.0):
        X, Y = shuffle(X, Y)
        Xtrain = X[:-100]
        Ytrain = Y[:-100]
        
        Xtest = X[-100:]
        Ytest = Y[-100:]
        
        costs_test = []
        costs_train = []
        errors = []
        
        n_batches = Xtrain.shape[0] // batch_sz
        
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

                self.W2 += learning_rate*(self.__derivative_W2(Z, Y_bt, Yp_tr) + reg*self.W2)
                self.b2 += learning_rate*(self.__derivative_b2(Y_bt, Yp_tr) + reg*self.b2)
                self.W1 += learning_rate*(self.__derivative_W1(Z, X_bt, Y_bt, Yp_tr) + reg*self.W1)
                self.b1 += learning_rate*(self.__derivative_b1(Z, Y_bt, Yp_tr) + reg*self.b1)
        
        # Create empty figure for future cost graph
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        # Add set of axes to the figure
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        
        # Plot costs
        axes.plot(costs_test, 'b', label='Train cost')
        axes.plot(costs_train, 'r', label='Test cost')
        axes.legend()
        axes.set_xlabel('Iteration')
        axes.set_ylabel('Cost')
        fig.savefig('costs.png')
        
        # Create empty figure for future error graph
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(errors, 'r')
        axes.set_ylabel('Error rate')
        axes.set_xlabel('Iteration')
        fig.savefig('error.png')
                      
                      
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
    model.fit(X, T)
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    