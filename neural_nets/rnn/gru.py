import numpy as np
import theano 
import theano.tensor as T

from util import init_weight

class GRU:
    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f = activation

        Wxr = init_weight(Mi, Mo)
        Whr = init_weight(Mo, Mo)
        br = np.zeros(Mo)
        Wxz = init_weight(Mi, Mo)
        Whz = init_weight(Mo, Mo)
        bz = np.zeros(Mo)
        Wxh = init_weight(Mi, Mo)
        Whh = init_weight(Mo, Mo)
        bh = np.zeros(Mo)
        h0 = np.zeros(Mo) 

        self.Wxr = theano.shared(Wxr.astype(np.float32))
        self.Whr = theano.shared(Whr.astype(np.float32))
        self.br = theano.shared(br.astype(np.float32))
        self.Wxz = theano.shared(Wxz.astype(np.float32))
        self.Whz = theano.shared(Whz.astype(np.float32))
        self.bz = theano.shared(bz.astype(np.float32))
        self.Wxh = theano.shared(Wxh.astype(np.float32))
        self.Whh = theano.shared(Whh.astype(np.float32))
        self.bh = theano.shared(bh.astype(np.float32))
        self.h0 = theano.shared(h0.astype(np.float32))
        self.params = [
            self.Wxr, self.Whr, self.br,
            self.Wxz, self.Whz, self.bz,
            self.Wxh, self.Whh, self.bh,
            self.h0
        ]

    def recurrence(self, X_t, h_t1):
        # Reset gate
        r = T.nnet.sigmoid(X_t.dot(self.Wxr) + h_t1.dot(self.Whr) + self.br)
        # Update gate
        z = T.nnet.sigmoid(X_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
        # H hat
        hhat = self.f(X_t.dot(self.Wxh) + (r* h_t1).dot(self.Whh) + self.bh)
        h_t = (1 - z)*h_t1 + z*hhat
        return h_t
    
    def output(self, x):
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0],
            n_steps=x.shape[0]
        )
        return h


        