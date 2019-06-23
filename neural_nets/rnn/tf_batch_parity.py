import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
import numpy as np
from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs

def x2sequence(x, T, D, batch_sz):
    x = tf.transpose(x, (1, 0, 2))
    x = tf.reshape(x, (T*batch_sz, D))
    x = tf.split(x, T)
    return x

class SimpleRNN:
    def __init__(self, M):
        self.M = M
    

    def fit(self, X, Y, batch_sz=20, lr=0.1, activation=tf.nn.relu, epochs=100):
        N, T, D = X.shape
        K = len(set(Y.flatten()))
        M = self.M
        self.f = activation

        Wo = init_weight(M, K)
        bo = np.zeros(K)

        self.Wo = tf.Variable(Wo, dtype=tf.float32)
        self.bo = tf.Variable(bo, dtype=tf.float32)

        tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
        tfY = tf.placeholder(tf.int32, shape=(batch_sz, T), name='targets')

        seqX = x2sequence(tfX, T, D, batch_sz)

        rnn_unit = GRUCell(num_units=M, activation=self.f)
        outputs, states = get_rnn_output(rnn_unit, seqX, dtype=tf.float32)

        # Ouputs are now of shape (T, batch_sz, M)
        # Make it (batch_sz, T, M)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (T*batch_sz, M))

        logits = tf.matmul(outputs, self.Wo) + self.bo
        predict_op = tf.argmax(logits, axis=1)
        targets = tf.reshape(tfY, (T*batch_sz,))

        cost_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=targets
            )
        )

        train_op = tf.train.AdamOptimizer(lr).minimize(cost_op)
        n_batches = N // batch_sz

        init_op = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init_op)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                n_correct = 0
                cost = 0
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Y[j*batch_sz:(j+1)*batch_sz]
                    _, c, p, s = session.run([train_op, cost_op, predict_op, seqX], feed_dict={tfX: Xbatch, tfY: Ybatch})
                    cost += c
                    for b in range(batch_sz):
                        idx = (b+1)*T - 1
                        n_correct += (p[idx] == Ybatch[b][-1])
                if i % 10 == 0:
                    print('s len', len(s), 's shape', s[0].shape)
                    print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                if n_correct == N:
                    print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                    break

def parity(B=12, learning_rate=10e-4, epochs=1000):
  X, Y = all_parity_pairs_with_sequence_labels(B)

  rnn = SimpleRNN(4)
  rnn.fit(
    X, Y,
    batch_sz=len(Y),
    lr=learning_rate,
    epochs=epochs,
    activation=tf.nn.relu
  )


if __name__ == '__main__':
  parity()