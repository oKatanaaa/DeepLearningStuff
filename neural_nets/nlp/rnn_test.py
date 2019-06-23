import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from tensorflow.nn import dynamic_rnn
import numpy as np
tf.set_random_seed(1)
np.random.seed(1)

def get_weights_test():
    cell = GRUCell(num_units=5, dtype=tf.float32)
    input_shape = [None, tf.Dimension(2)]
    cell.build(inputs_shape=input_shape)

    weights = cell.variables
    print(weights)

def stack_built_cells():
    cell1 = GRUCell(3, dtype=tf.float32)
    cell2 = GRUCell(2, dtype=tf.float32)
    cell3 = GRUCell(2, dtype=tf.float32)
    cell1.build(inputs_shape=[None, tf.Dimension(3)])
    cell2.build(inputs_shape=[None, tf.Dimension(3)])
    cell3.build(inputs_shape=[None, tf.Dimension(2)])
    multicell = MultiRNNCell([cell1, cell2, cell3])
    session = tf.Session()
    
    
    inputs = tf.placeholder(tf.float32, shape=[2, 3, 3], name='input')
    state, out = dynamic_rnn(multicell, inputs, dtype=tf.float32)
    init_op = tf.global_variables_initializer()
    session.run(init_op)
    out1, out2 = session.run(
        [state, out],
        feed_dict={inputs: np.random.randn(2, 3, 3).astype(np.float32)}
    )
    print('states')
    print(out1)
    print('outs')
    print(out2)


if __name__ == "__main__":
    get_weights_test()
    stack_built_cells()