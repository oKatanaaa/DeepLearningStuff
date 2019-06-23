import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

X = 2*np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300))
plt.plot(X)
plt.savefig('before.png')
plt.close()

decay = T.scalar('decay')
sequence = T.vector('sequence')

def recurrence(x, last, decay):
    return (1-decay)*x + decay*last

outputs, _ = theano.scan(
    fn=recurrence,
    sequences=sequence,
    n_steps=sequence.shape[0],
    outputs_info=[0.],
    non_sequences=[decay]
)

lpf = theano.function(
    inputs=[sequence, decay],
    outputs=outputs
)

Y = lpf(X.astype(np.float32), np.float32(0.99))
plt.plot(Y)
plt.savefig('after.png')