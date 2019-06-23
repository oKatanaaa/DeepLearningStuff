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

    def recurrence(self, WXr_t, WXz_t, WXh_t, is_start, h_t1):
        h_t1 = T.switch(
            T.eq(is_start, 1),
            self.h0,
            h_t1
        )
        r = T.nnet.sigmoid(WXr_t + h_t1.dot(self.Whr))
        # Update gate
        z = T.nnet.sigmoid(WXz_t + h_t1.dot(self.Whz))
        # H hat
        hhat = self.f(WXh_t + (r* h_t1).dot(self.Whh))
        h_t = (1 - z)*h_t1 + z*hhat
        return h_t
    
    def output(self, x, start_points):
        WXr = x.dot(self.Wxr) + self.br
        WXz = x.dot(self.Wxz) + self.bz
        WXh = x.dot(self.Wxh) + self.bh
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=[WXr, WXz, WXh, start_points],
            outputs_info=[self.h0],
            n_steps=x.shape[0]
        )
        return h


        