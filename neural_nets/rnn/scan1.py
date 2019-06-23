import numpy as np
import theano
import theano.tensor as T

x = T.ivector('x')

def square(x):
    return x*x

outputs, updates = theano.scan(
    fn=square,
    sequences=x,
    n_steps=x.shape[0]
)

square_op = theano.function(
    inputs=[x],
    outputs=[outputs]
)

o_val = square_op(np.array([1, 2, 3, 4], dtype=np.int32))
print(o_val)