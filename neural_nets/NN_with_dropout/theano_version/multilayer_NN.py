import theano as th
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.utils import shuffle
from utils import error_rate, create_graph, one_hot_encoding

class HiddenLayer(object):
    def __init__(self, M1, M2, h_id):
        self.id = h_id
        self.M1 = M1
        self.M2 = M2
        W_init = np.random.randn(M1, M2)
        b_init = np.zeros(M2)
        self.W = th.shared(W_init, 'W_{}'.format(h_id))
        self.b = th.shared(b_init, 'b_{}'.format(h_id))
        self.params = [self.W, self.b]
        
    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)
    

class MultiLayerNN(object):
    def __init__(self, D, K, h_layer_sizes, ps_keep):
        self.dropout_rates = ps_keep
        self.hidden_layers = []
        self.mask_generator = RandomStreams()
        M1 = D
        M2 = None
        count = 0
        for M2 in h_layer_sizes:
            W = np.random.randn(M1, M2)
            b = np.zeros(M2)
            self.hidden_layers.append(HiddenLayer(M1, M2, count))
            M1 = M2
        
        W = np.random.randn(M1, K) 
        b = np.zeros(K)
        self.W = th.shared(W, 'W_last')
        self.b = th.shared(b, 'b_last')
        
        self.thX = T.matrix('X')
        self.thT = T.matrix('T')
        # Collect params for future i=use of gradient descent
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
            
        
    def forward(self, X):
        Z = X
        for hidden_layer, p in zip(self.hidden_layers, self.dropout_rates):
            Z = hidden_layer.forward(Z*p)
        
        return T.nnet.softmax(Z.dot(self.W) + self.b)
    
    def forward_train(self, X):
        Z = X
        for hidden_layer, p_keep in zip(self.hidden_layers, self.dropout_rates):
            mask = self.mask_generator.binomial(n=1, p=p_keep, size=Z.shape)
            Z = Z * mask
            Z = hidden_layer.forward(Z)
        
        return T.nnet.softmax(Z.dot(self.W) + self.b)
        
    def fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate=0.0001, mu=0.9, decay=0.999, epochs=9, batch_sz=100):
        Xtrain = Xtrain.astype(np.float32)
        Ytrain = Ytrain.astype(np.int32)
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.astype(np.int32)
        
        Yp_train = self.forward_train(self.thX)
        cost = -T.mean(self.thT*T.log(Yp_train))
        
        # Find all gradients at once
        grads = T.grad(cost, self.params)
        
        # For momentum
        m_params = [th.shared(np.zeros_like(p.get_value())) for p in self.params]
        
        # For rmsprop
        caches = [th.shared(np.ones_like(p.get_value())) for p in self.params]
        eps = 10e-10
        
        new_caches = [decay*cache + (1-decay)*grad*grad for cache, grad in zip(caches, grads)]
        new_m_params = [mu*m_param + learning_rate*g/T.sqrt(new_cache + eps) for m_param, g, new_cache in zip(m_params, grads, new_caches)]
        new_params = [param - new_m_param for param, new_m_param in zip(self.params, new_m_params)] 
        
        updates = [
            (cache, new_cache) for cache, new_cache in zip(caches, new_caches)
        ] + [
            (m_param, new_m_param) for m_param, new_m_param in zip(m_params, new_m_params)
        ] + [
            (param, new_param) for param, new_param in zip(self.params, new_params)
        ]
        
        train_op = th.function(
            inputs=[self.thX, self.thT],
            updates=updates
        )
        
        Yp_predict = self.forward(self.thX)
        cost_predict = -(self.thT*T.log(Yp_predict)).mean()
        cost_predict_op = th.function(inputs=[self.thX, self.thT], outputs=[cost_predict, Yp_predict])
        
        n_batches = len(Xtrain) // batch_sz
        costs, errors = [], []
        
        for i in range(epochs):
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz]
                Ybatch = Ytrain[j*batch_sz:(j+1)*batch_sz]
        
                train_op(Xbatch, Ybatch)
                c, Yp = cost_predict_op(Xtest, Ytest)
                err = error_rate(np.argmax(Yp, axis=1), np.argmax(Ytest, axis=1))
                costs.append(c)
                errors.append(err)
                if j % 20 == 0:
                    print('Accuracy: ', 1 - err)
                    print('Cost: ', c)
        
        return costs, errors
                
        
        
def main():
    Nclass = 500
    D = 2
    X1 = np.random.randn(Nclass, D) + np.array([0, 2])
    X2 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 0])
    X = np.vstack([X1, X2, X3])
    
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    Y = one_hot_encoding(Y)
    
    
    N = 1500
    H = 10
    K = 3
    
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]
    
    model = MultiLayerNN(D, K, [10, 10], [0.5, 0.5])
    costs, errors = model.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=1000)
    
    create_graph(costs, 'cost.png', y_label='cost', x_label='iteration')
    create_graph(errors, 'error.png', y_label='error', x_label='iteration')

if __name__ == '__main__':
    main()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        