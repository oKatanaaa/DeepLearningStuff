import tensorflow as tf
from util import init_weight
from tensorflow.contrib.rnn import GRUCell, static_rnn, MultiRNNCell
from tensorflow.nn import dynamic_rnn
import numpy as np


class Dense:
    def __init__(self, Mi, Mo):
        self.Mi = Mi
        self.Mo = Mo

        W = init_weight(Mi, Mo)
        b = np.zeros(Mo)

        self.W = tf.Variable(W, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)

    def forward(self, X):
        return tf.matmul(X, self.W) + self.b


class RNNBlock:
    def __init__(self, layers, input_shape):
        """
        input_shape is a list of shapes: [batch_sz, seq_legth, num_features]
        """
        self.layers = layers
        self.T = input_shape[1]
        self.D = input_shape[2]
        self.batch_size = input_shape[0]
        self.cells = [layer.get_cells() for layer in layers]
        self.rnn_combined = MultiRNNCell(cells=self.cells)
    
    def x2sequence(self, x):
        T = self.T
        D = self.D
        batch_sz = self.batch_size
        # Permuting batch_size and n_steps
        x = tf.transpose(x, (1, 0, 2))
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, (T*batch_sz, D))
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # x = tf.split(0, T, x) # v0.1
        x = tf.split(x, T) # v1.0
        # print "type(x):", type(x)
        return x

    def forward(self, X):
        #X = self.x2sequence(X)
        #X = tf.unstack(X, self.T, 1)
        #return static_rnn(self.rnn_combined, X, dtype=tf.float32)
        #return self.x2sequence(X)
        return tf.unstack(X, self.T, 1)




class GRU:
    def __init__(self, num_units, activation=tf.nn.relu):
        self.num_units = num_units
        self.cells = GRUCell(num_units=num_units, activation=activation)
    
    def get_cells(self):
        return self.cells
    
    def forward(self, X):
        # X is a sequence of shape (T, batch_sz, D)
        return static_rnn(self.cells, X, dtype=tf.float32)
    
class SimpleRNN:
    def __init__(self, layers, input_shape):
        """
        input_shape is a list of shapes: [batch_sz, seq_legth, num_features]
        """
        self.layers = layers
        self.tfX = tf.placeholder(tf.float32, shape=[*input_shape])
        self.forward_x = self.forward(self.tfX)

    def forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer.forward(X)
        return Z

    def predict(self, X, session):
        return session.run(
            self.forward_x,
            feed_dict={self.tfX: X}
        )


if __name__ == "__main__":
    layers = [
        RNNBlock(layers=[GRU(4), GRU(4)], input_shape=[2, 3, 2])
    ]
    tf.set_random_seed(1)
    np.random.seed(1)
    rnn = SimpleRNN(layers=layers, input_shape=[2, 3, 2])
    session = tf.Session()
    init_op = tf.global_variables_initializer()
    session.run(init_op)
    X = np.random.randn(2, 3, 2)
    x = rnn.predict(X, session)
    print(X)
    print(x)
    #outs, final_out = rnn.predict(X, session)
    """
    print('outs')
    for o in outs:
        print(o)
    print('states')
    for state in final_out:
        print(state)
    print('final_out')
    print(final_out)
    """
    
    
