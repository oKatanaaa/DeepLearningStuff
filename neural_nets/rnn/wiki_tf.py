import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from util import init_weight
import numpy as np

def x2sequence(x, tfSeqsLengthes, D):
    total_length = tf.reduce_sum(tfSeqsLengthes)
    x = tf.reshape(x, (total_length, D))
    x = tf.split(x, tfSeqsLengthes)
    return x

class RNN:
    def __init__(self, V, D, hidden_layers_sizes, activation=tf.nn.tanh):
        # Vocab size
        self.V = V
        # Vocab dimensionality
        self.D = D
        self.hidden_layers_sizes = hidden_layers_sizes
        self.f = activation

    def fit(self, X, batch_sz=128, lr=10e-2, epochs=1000, cell_type=GRUCell):
        N = len(X)
        V = self.V
        D = self.D

        self.layers = []
        for size in self.hidden_layers_sizes:
            self.layers += [cell_type(size, activation=self.f)]
        

        tfX = tf.placeholder(dtype=tf.float32, shape=(batch_sz, None))
        tfY = tf.placeholder(dtype=tf.float32, shape=(batch_sz, None))
        tfSeqsLengthes = tf.placeholder(dtype=tf.int32, shape=(batch_sz))
        # Create vocab
        We = init_weight(V, D)
        Wo = init_weight(self.hidden_layers_sizes[-1], V)
        bo = np.zeros(V)
        self.We = tf.Variable(We, dtype=tf.float32, name='vocab')
        self.Wo = tf.Variable(Wo, dtype=tf.float32, name='output mat')
        self.bo = tf.Variable(bo, dtype=tf.float32, name='output bias')


        total_batch_length = tf.reduce_sum(tfSeqsLengthes)
        X = tf.reshape(tfX, (total_batch_length))
        WX = tf.nn.embedding_lookup(self.We, X)
        # (total_batch_length, D)
        h = tf.split(WX, tfSeqsLengthes)
        for rnn_layer in self.layers:
            h, states = get_rnn_output(rnn_layer, WX, dtype=tf.float32)
        
        X = tf.reshape(h, (total_batch_length, D))
        logits = tf.matmul(h, self.Wo) + self.bo

        Y = tf.reshape(tfY, (total_batch_length))
        

        
        


