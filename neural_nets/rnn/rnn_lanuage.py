import numpy as np
import theano
import theano.tensor as T
from util import remove_punctuation, get_robert_frost, init_weight

from sklearn.utils import shuffle
from future.utils import iteritems


class SimpleRNN:
    def __init__(self, D, M, V):
        self.D = D
        self.M = M
        self.V = V

    def fit(self, X, lr=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=100):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V

        # Init weights

        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)

        Wo = init_weight(M, V)
        bo = np.zeros(V)

        self.set(We, Wx, Wh, bh, h0, Wo, bo, activation)
        
        thX = T.ivector('X')
        Ei = self.We[thX]
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t
        
        [h, y], _ = theano.scan(
            fn=recurrence,
            sequences=Ei,
            n_steps=Ei.shape[0],
            outputs_info=[self.h0, None]
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        updates = [
            (p, p - lr*g + dp) for p, g, dp in zip(self.params, grads, dparams)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True
        )
        

        train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        n_total = sum((len(sentence) + 1 for sentence in X))
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            cost = 0
            for j in range(len(X)):
                x_input = [0] + X[j]
                y_input = X[j] + [1]
                c, pred = train_op(x_input, y_input)
                cost += c
                for pr, y_in in zip(pred, y_input):
                    n_correct += (pr == y_in)
                
            print('i', i, 'cost', cost, 'correct rate', n_correct/n_total)
            costs.append(cost)
        return costs

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(filename, activation):
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wo = npz['arr_5']
        bo = npz['arr_6']

        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wo, bo, activation)
        return rnn
    
    def set(self, We, Wx, Wh, bh, h0, Wo, bo, activation):
        self.f = activation
        self.We = theano.shared(We.astype(np.float32))
        self.Wx = theano.shared(Wx.astype(np.float32))
        self.Wh = theano.shared(Wh.astype(np.float32))
        self.bh = theano.shared(bh.astype(np.float32))
        self.h0 = theano.shared(h0.astype(np.float32))
        self.Wo = theano.shared(Wo.astype(np.float32))
        self.bo = theano.shared(bo.astype(np.float32))
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X')
        Ei = self.We[thX]
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t
        
        [h, y], _ = theano.scan(
            fn=recurrence,
            sequences=Ei,
            n_steps=Ei.shape[0],
            outputs_info=[self.h0, None]
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True
        )
    
    def generate(self, pi, word2idx):
        idx2word = {v:k for k, v in iteritems(word2idx)}
        V = len(pi)

        n_lines = 0

        X = [np.random.choice(V, p=pi) ]
        print(idx2word[X[0]])
        while n_lines < 4:
            P = self.predict_op(X)[-1]
            X += [P]
            if P > 1:
                word = idx2word[P]
                print(word, end=' ')
            elif P == 1:
                n_lines += 1
                print('')
                if n_lines < 4:
                    X = [np.random.choice(V, p=pi) ]
                    print(idx2word[X[0]], end=' ')


def train_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN(30, 30, len(word2idx))
    rnn.fit(sentences, lr=10e-5, activation=T.nnet.relu, epochs=2000)
    rnn.save('RNN_D30_M30_epochs2000_relu.npz')

def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('RNN_D30_M30_epochs2000_relu.npz', T.nnet.relu)

    V = len(word2idx)
    pi = np.zeros(V)
    for sentences in sentences:
        pi[sentences[0]] += 1
    pi /= pi.sum()
    rnn.generate(pi, word2idx)

if __name__ == '__main__':
    #train_poetry()
    generate_poetry()

