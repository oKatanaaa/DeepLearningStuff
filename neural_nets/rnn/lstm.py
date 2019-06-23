import numpy as np
import theano 
import theano.tensor as T

from util import init_weight


class LSTM:
    def __init__(self, Mi, Mo, activation):
        self.f = activation
        self.Mi = Mi
        self.Mo = Mo

        # Input gate weights
        Wxi = init_weight(Mi, Mo)
        Whi = init_weight(Mo, Mo)
        Wci = init_weight(Mo, Mo)
        bi = np.zeros(Mo)

        # Forget gate weights
        Wxf = init_weight(Mi, Mo)
        Whf = init_weight(Mo, Mo)
        Wcf = init_weight(Mo, Mo)
        bf = np.zeros(Mo)

        # Candidate weights
        Wxc = init_weight(Mi, Mo)
        Whc = init_weight(Mo, Mo)
        bc = np.zeros(Mo)
        c0 = np.zeros(Mo)

        # Output gate weights
        Wxo = init_weight(Mi, Mo)
        Who = init_weight(Mo, Mo)
        Wco = init_weight(Mo, Mo)
        bo = np.zeros(Mo)
        h0 = np.zeros(Mo)

        self.Wxi = theano.shared(Wxi.astype(np.float32))
        self.Whi = theano.shared(Whi.astype(np.float32))
        self.Wci = theano.shared(Wci.astype(np.float32))
        self.bi = theano.shared(bi.astype(np.float32))

        self.Wxf = theano.shared(Wxf.astype(np.float32))
        self.Whf = theano.shared(Whf.astype(np.float32))
        self.Wcf = theano.shared(Wcf.astype(np.float32))
        self.bf = theano.shared(bf.astype(np.float32))

        self.Wxc = theano.shared(Wxc.astype(np.float32))
        self.Whc = theano.shared(Whc.astype(np.float32))
        self.bc = theano.shared(bc.astype(np.float32))
        self.c0 = theano.shared(c0.astype(np.float32))

        self.Wxo = theano.shared(Wxo.astype(np.float32))
        self.Who = theano.shared(Who.astype(np.float32))
        self.Wco = theano.shared(Wco.astype(np.float32))
        self.bo = theano.shared(bo.astype(np.float32))
        self.h0 = theano.shared(h0.astype(np.float32))
        self.params = [
            self.Wxi, self.Whi, self.Wci, self.bi,
            self.Wxf, self.Whf, self.Wcf, self.bf,
            self.Wxc, self.Whc, self.bc, self.c0,
            self.Wxo, self.Who, self.Wco, self.bo,
            self.h0
        ]
    
    def recurrence(self, x_t, h_t1, c_t1):
        # Input gate: affects how much of new value the cell will consider
        i_t = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        # Forget gate: affects how much of old valuew the cell will consider
        f_t = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        # Candidate value
        c_t = f_t*c_t1 + i_t*self.f(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        # Output gate
        o_t = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o_t*self.f(c_t)
        return h_t, c_t
    
    def output(self, x):
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            n_steps=x.shape[0],
            outputs_info=[self.h0, self.c0]
        )
        return h

