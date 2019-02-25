import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import get_preprocessed_data, error_rate
from datetime import datetime
from sklearn.utils import shuffle


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
    
    
    
    
    
    def forward(self, X):
        Z = np.tanh(X.dot(self.W1))
        return self.softmax(Z.dot(self.W2)), Z
        
    def fit_full(self, X, Y, epochs=100, learning_rate=0.0001, reg=0.0):
        X, Y = shuffle(X, Y)
        Xtrain = X[:-1000]
        Ytrain = Y[:-1000]
        
        Xtest = X[-1000:]
        Ytest = Y[-1000:]
        
        costs_test = []
        costs_train = []
        errors = []
        
        t0 = datetime.now()
        for i in range(epochs):
            Yp_tr, Z = self.forward(Xtrain)
            Yp_ts, _ = self.forward(Xtest)
            
            ctr = self.cost(Yp_tr, Ytrain)
            cts = self.cost(Yp_ts, Ytest)
            error = error_rate(np.argmax(Yp_tr, axis=1), np.argmax(Ytrain, axis=1))
            
            costs_test.append(cts)
            costs_train.append(ctr)
            errors.append(error)
            
            if i % 10 == 0:
                print("Train score: ", 1 - error)
                print("Train cost: ", ctr)
                print("Test cost: ", cts)
        
            self.W2 += learning_rate*(self.__derivative_W2(Z, Ytrain, Yp_tr) + reg*self.W2)
            self.b2 += learning_rate*(self.__derivative_b2(
            self.W1 += learning_rate*(self.__derivative_W1(Z, Xtrain, Ytrain, Yp_tr) + reg*self.W1)
        
        # Create empty figure for future cost graph
        fig = plt.figure()
        # Add set of axes to the figure
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        
        # Plot costs
        axes.plot(costs_test, 'b', label='Train cost')
        axes.plot(costs_train, 'r', label='Test cost')
        axes.legend()
        axes.set_xlabel('Iterations')
        axes.set_ylabel('Cost')
        fig.savefig('fig_full.png')
        return errors, datetime.now() - t0
        
        
    def fit_batch(self, X, Y, epochs=10, learning_rate=0.0001, reg=0.0, batch_sz=100):
        X, Y = shuffle(X, Y)
        Xtrain = X[:-1000]
        Ytrain = Y[:-1000]

        Xtest = X[-1000:]
        Ytest = Y[-1000:]

        n_batches = Xtrain.shape[0] // batch_sz

        costs_test = []
        costs_train = []
        errors = []

        t0 = datetime.now()
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
                self.W1 += learning_rate*(self.__derivative_W1(Z, X_bt, Y_bt, Yp_tr) + reg*self.W1)

        # Create empty figure for future cost graph
        fig = plt.figure()
        # Add set of axes to the figure
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # Plot costs
        axes.plot(costs_test, 'b', label='Train cost')
        axes.plot(costs_train, 'r', label='Test cost')
        axes.legend()
        axes.set_xlabel('Iterations')
        axes.set_ylabel('Cost')
        fig.savefig('fig_batch.png')
        return errors, datetime.now() - t0
        
    def fit_stoch(self, X, Y, epochs=1, learning_rate=0.0001, reg=0.0):
        X, Y = shuffle(X, Y)
        Xtrain = X[:-1000]
        Ytrain = Y[:-1000]

        Xtest = X[-1000:]
        Ytest = Y[-1000:]

        costs_test = []
        costs_train = []
        errors = []

        t0 = datetime.now()
        for _ in range(epochs):
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for i in range(min(Xtrain.shape[0], 500)):
                Yp_tr, Z = self.forward(Xtrain[i, :].reshape(1, self.D))
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

                self.W2 += learning_rate*(self.__derivative_W2(Z, Ytrain[i,:].reshape(1, self.C), Yp_tr) + reg*self.W2)
                self.W1 += learning_rate*(self.__derivative_W1(Z, Xtrain[i,:].reshape(1, self.D), Ytrain[i,:].reshape(1, self.C), Yp_tr) + reg*self.W1)

        # Create empty figure for future cost graph
        fig = plt.figure()
        # Add set of axes to the figure
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # Plot costs
        axes.plot(costs_test, 'b', label='Train cost')
        axes.plot(costs_train, 'r', label='Test cost')
        axes.legend()
        axes.set_xlabel('Iterations')
        axes.set_ylabel('Cost')
        fig.savefig('fig_stoch.png')
        return errors, datetime.now() - t0
                      
                      
def main():
    X, Y = get_preprocessed_data()
    N, D = X.shape
    K = Y.shape[1]
    H = 100
    
    model1 = NeuralNetwork(D, H, K)
    error_full, time_full = model1.fit_full(X, Y)
    
    
    model2 = NeuralNetwork(D, H, K)
    error_batch, time_batch = model2.fit_batch(X, Y)
    
    
    model3 = NeuralNetwork(D, H, K)
    error_stoch, time_stoch = model3.fit_stoch(X, Y)
    
    
    print('Time full: ', time_full)
    print('Time batch: ', time_batch)
    print('Time stochastic: ', time_stoch)
    
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Error')
    
    x1 = np.linspace(0, 1, len(error_full))
    axes.plot(x1, error_full, 'r', label='full')
    
    x2 = np.linspace(0, 1, len(error_batch))
    axes.plot(x2, error_batch, 'b', label='batch')
    
    x3 = np.linspace(0, 1, len(error_stoch))
    axes.plot(x3, error_stoch, 'g', label='stochastic')
    fig.legend()
    fig.savefig('error_rate.png')
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    