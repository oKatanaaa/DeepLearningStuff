# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import sys
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import json

from datetime import datetime
from sklearn.utils import shuffle
from gru import GRU
from gru_batch import GRU as GRUb
from lstm import LSTM
from util import init_weight, get_wikipedia_data
from brown import get_sentences_with_word2idx_limit_vocab


class RNN:
    def __init__(self, D, hidden_layer_sizes, V):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.D = D
        self.V = V
    
    def fit(self, X, batch_sz=256, lr=10e-5, mu=0.99, epochs=100, activation=T.nnet.relu, recurrent_unit=GRUb, normalize=True):
        D = self.D
        V = self.V
        N = len(X)

        We = init_weight(V, D)
        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = recurrent_unit(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo
        Wo = init_weight(Mi, V)
        bo = np.zeros(V)
        self.We = theano.shared(We.astype(np.float32))
        self.Wo = theano.shared(Wo.astype(np.float32))
        self.bo = theano.shared(bo.astype(np.float32))
        self.params = [self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params
        
        thX = T.ivector('X')
        thY = T.ivector('Y')
        start_p = T.ivector('St')

        Z = self.We[thX]
        for ru in self.hidden_layers:
            Z = ru.output(Z, start_p)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)

        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs=[thX, start_p],
            outputs=[py_x, prediction],
            allow_input_downcast=True
        )

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        dWe = theano.shared(self.We.get_value()*0)

        gWe = T.grad(cost, self.We)
        dWe_update = mu*dWe - lr*gWe
        We_update = self.We + dWe_update
        if normalize:
            We_update /= We_update.norm(2)

        updates = [
            (p, p + dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ] + [
            (self.We, We_update), (dWe, dWe_update)
        ]

        self.train_op = theano.function(
            inputs=[thX, thY, start_p],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        rates = []
        n_batches = N // batch_sz
        for i in range(epochs):
            t0 = datetime.now()
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                Xinput, Yinput, start_points = self.adjust_batch(Xbatch)
                n_total += len(Xinput)
                c, p = self.train_op(Xinput, Yinput, start_points)
                cost += c
                for pj, xj in zip(p, Yinput):
                    if pj == xj:
                        n_correct += 1
                
                if j % 1 == 0:
                    sys.stdout.write("j/n_batches:%d/%d correct rate: %f\r" % (j, n_batches, n_correct/n_total))
                    sys.stdout.flush()
                
            print("i", i, "cost", cost, 'rate', n_correct/n_total, 'time for epoch', (datetime.now() - t0))
            costs.append(cost)
            rates.append(n_correct/n_total)
        return costs, rates
    
    def adjust_batch(self, Xbatch):
        start_points = []
        Xinput = []
        Yinput = []
        for i in range(len(Xbatch)):
            start_points += [1]
            if np.random.random() < 0.01 or len(Xbatch[i]) <= 1:
                input_sequence = [0] + Xbatch[i]
                output_sequence = Xbatch[i] + [1]
                start_points += len(Xbatch[i])*[0]
            else:
                input_sequence = [0] + Xbatch[i][:-1]
                output_sequence = Xbatch[i]
                start_points += len(Xbatch[i][:-1])*[0]
            Xinput += input_sequence
            Yinput += output_sequence
        return Xinput, Yinput, np.array(start_points, dtype=np.int32)

                    
def train_wikipedia(we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json', recurrent_unit=GRU):
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab()
    print('finished retrieving data')
    print('vocab size', len(word2idx), 'number of sentences', len(sentences))
    rnn = RNN(50, [50], len(word2idx))
    rnn.fit(sentences, lr=1e-2, epochs=50, normalize=False, activation=T.tanh)
    np.save(we_file, rnn.We.get_value())
    with open(w2i_file, 'w') as f:
        json.dump(word2idx, f)

def find_analogies(w1, w2, w3, we_file='word_embeddings.npy', w2i_file='wikipedia_word2idx.json'):
    We = np.load(we_file)
    with open(w2i_file) as f:
        word2idx = json.load(f)
    
    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    def dist1(a, b):
        return np.linalg.norm(a - b)
    
    def dist2(a, b):
        return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    for dist, name in [(dist1, 'euclidian'), (dist2, 'cosine')]:
        min_dist = float('inf')
        best_word = ''
        for word, idx in iteritems(word2idx):
            if word not in (w1, w2, w3):
                v1 = We[idx]
                d = dist(v0, v1)
                if d < min_dist:
                    min_dist = d
                    best_word = word
        print('closest match by', name, 'distance', best_word)
        print(w1, '-', w2, '=', best_word, '-', w3)


if __name__ == '__main__':
    we = 'lstm_word_embeddings2.npy'
    w2i = 'lstm_wikipedia_word2idx2.json'
    train_wikipedia(we, w2i, recurrent_unit=GRU)
    find_analogies('king', 'man', 'woman', we, w2i)
    find_analogies('france', 'paris', 'london', we, w2i)
    find_analogies('france', 'paris', 'rome', we, w2i)
    find_analogies('paris', 'france', 'italy', we, w2i)



