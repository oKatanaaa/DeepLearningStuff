import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs

class SimpleRNN:
    def __init__(self, D, M, K, activation=T.tanh):
        self.f = activation

        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)

        initial_state = np.zeros(M)

        # Martices for classification layer
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        self.Wx = theano.shared(Wx.astype(np.float32))
        self.Wh = theano.shared(Wh.astype(np.float32))
        self.bh = theano.shared(bh.astype(np.float32))
        # Initial state
        self.h0 = theano.shared(initial_state.astype(np.float32))

        self.Wo = theano.shared(Wo.astype(np.float32))
        self.bo = theano.shared(bo.astype(np.float32))
        self.params = [self.Wx, self.Wh, self.bh, self.Wo, self.bo, self.h0]

        

    def fit(self, X, Y, lr=0.1, mu=0.99, epochs=1):
        thX = T.fmatrix('X')
        thY = T.ivector('Y')

        def recurrence(X, previous_out):
            current_out = self.f( X.dot(self.Wx) + previous_out.dot(self.Wh) + self.bh )
            current_network_answer = T.nnet.softmax( current_out.dot(self.Wo) + self.bo )
            return current_network_answer, current_out

        [y, h], _ = theano.scan(
            fn=recurrence,
            outputs_info=[None, self.h0],
            sequences=thX,
            n_steps=thX.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + dp - lr*g) for p, dp, g in zip(self.params, dparams,grads)
        ] + [
            (dp, dp*mu - lr*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(
            inputs=[thX], 
            outputs=prediction
        )
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(X.shape[0]):
                c, p = self.train_op(X[j], Y[j])
                cost += c
                if p[-1] == Y[j, -1]:
                    n_correct += 1
            print('i', i, 'cost:', cost, 'classification rate:', n_correct/X.shape[0])
            costs.append(cost)
        return costs

def parity(B=12, lr=10e-5, epochs=200):
    X, Y = all_parity_pairs(B)
    N, t = X.shape
    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in range(N):
        ones_count = 0
        for i in range(t):
            if X[n, i] == 1:
                ones_count += 1
            if ones_count % 2 == 1:
                Y_t[n,i] = 1
    rnn = SimpleRNN(1, 4, 2, activation=T.nnet.sigmoid)
    X = X.reshape(N, t, 1).astype(np.float32)
    rnn.fit(X, Y_t, epochs=epochs)

if __name__ == '__main__':
    parity()





        
