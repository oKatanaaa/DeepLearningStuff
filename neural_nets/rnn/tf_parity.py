import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels

class SimpleRNN:
    def __init__(self, T, D, H, K, activation):
        self.T = T
        self.D = D
        self.H = H
        self.K = K
        self.f = activation

        Wx = init_weight(D, H)
        Wh = init_weight(H, H)
        bh = np.zeros(H)
        h0 = np.zeros(H)

        Wo = init_weight(H, K)
        bo = np.zeros(K)

        self.Wx = tf.Variable(Wx, dtype=tf.float32)
        self.Wh = tf.Variable(Wh, dtype=tf.float32)
        self.bh = tf.Variable(bh, dtype=tf.float32)
        self.Wo = tf.Variable(Wo, dtype=tf.float32)
        self.bo = tf.Variable(bo, dtype=tf.float32)
        self.h0 = tf.Variable(h0, dtype=tf.float32)
        self.params = [self.Wx, self.Wh, self.bh, self.Wo, self.bo, self.h0]

        
    
    def fit(self, X, Y, lr=10e-2, epochs=10):
        N = len(Y)
        T = self.T
        D = self.D
        M = self.H
        tfX = tf.placeholder(tf.float32, shape=(T, D), name='X')
        tfY = tf.placeholder(tf.int32, shape=(T), name='Y')
        
        XWx = tf.matmul(tfX, self.Wx) + self.bh
        def reccurence(xw_t, h_t1):
            h_t = self.f(xw_t + tf.matmul(tf.reshape(h_t1,(1, M)), self.Wh))
            return tf.reshape(h_t, (M,))

        h = tf.scan(
            fn=reccurence,
            elems=XWx,
            initializer=self.h0
        )
        
        logits = tf.matmul(h, self.Wo) + self.bo

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tfY,
                logits=logits
            )
        )

        predict_op = tf.argmax(logits, axis=1)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        init_op = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init_op)

            costs = []
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                n_correct = 0
                cost = 0
                for j in range(N):
                    _, c, p = session.run(
                        [train_op, loss, predict_op],
                        feed_dict={tfX: X[j].reshape(T, D), tfY: Y[j]}
                    )
                    cost += c
                    if p[-1] == Y[j, -1]:
                        n_correct += 1
                print('i', i, 'cost', cost, 'accuracy', n_correct/N)

def parity(B=12, learning_rate=1e-4, epochs=200):
    X, Y = all_parity_pairs_with_sequence_labels(B)
    X = X.astype(np.float32)

    rnn = SimpleRNN(12, 1, 20, 2, activation=tf.nn.relu)
    rnn.fit(X, Y, lr=learning_rate, epochs=epochs)


if __name__ == '__main__':
    parity()
