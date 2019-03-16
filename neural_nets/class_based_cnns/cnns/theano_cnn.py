import theano as th
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
import numpy as np
from utils import get_preprocessed_image_data, error_rate, create_graph
from sklearn.utils import shuffle

class ConvPoolLayer(object):
    def __init__(self, shape=(1, 1, 1, 1), poolsize=(2, 2), conv_id=0):
        self.shape = shape
        self.poolsize = poolsize
        
        W = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]))
        b = np.zeros(shape[0])
        
        # Convert to float32
        W = W.astype(np.float32)
        b = b.astype(np.float32)
        
        self.W = th.shared(W, 'C_{}'.format(conv_id))
        self.b = th.shared(b, 'b_{}'.format(conv_id))
        
        self.params = [self.W, self.b]
    
    def forward(self, X):
        conv_out = T.nnet.conv2d(input=X, filters=self.W)
        pool_out = pool_2d(input=conv_out, ws=(2,2), ignore_border=True)
        
        # Add the bias term. Since the bias is a vector 1D array, we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini_batches and feature maps
        return T.nnet.relu(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    
    
class HiddenLayer(object):
    def __init__(self, M1, M2, h_id):
        self.id = h_id
        self.M1 = M1
        self.M2 = M2
        W_init = np.random.randn(M1, M2).astype(np.float32)
        b_init = np.zeros(M2).astype(np.float32)
        self.W = th.shared(W_init, 'W_{}'.format(h_id))
        self.b = th.shared(b_init, 'b_{}'.format(h_id))
        self.params = [self.W, self.b]
        
    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)
    
    
class CNN(object):
    def __init__(self, input_shape, K, conv_layer_sizes, hidden_layer_sizes):
        """
        input_shape - tuple with sizes of the input image. Data: (color_channels, image_width, image_height)
        K - number of classes. Data: int
        conv_layer_sizes - list of tuples with kernel sizes. Data: [(num_feature_maps, kernel_width, kernel_height)]
        hidden_layer_sizes - list of hidden layer sizes. Data: [int, int, ...]
        """
        self.conv_layer_sizes = conv_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        
        # Init conv_layers
        self.conv_layers = []
        c, w, h = input_shape
        count = 0
        output_shape = w
        for f_maps, k_w, k_h in conv_layer_sizes:
            # k_w - kernel width
            # k_h - kernel height
            # f_maps - feature maps
            layer_shape = (f_maps, c, k_w, k_h)
            layer = ConvPoolLayer(layer_shape, count)
            self.conv_layers.append(layer)
            
            c = f_maps
            w = k_w
            h = k_h
            output_shape = (output_shape - k_w + 1) // 2
            count += 1
        
        # Init dense layers
        self.hidden_layers = []
        M1 = c*output_shape**2
        count = 0
        for M2 in hidden_layer_sizes:
            h_layer = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h_layer)
            M1 = M2
            count += 1
        
        # Create last classification layer
        W = np.random.randn(M1, K).astype(np.float32) 
        b = np.zeros(K).astype(np.float32)
        self.W = th.shared(W, 'W_last')
        self.b = th.shared(b, 'b_last')
        
        
        # Collect all the trainable params
        self.params = []

        for conv_layer in self.conv_layers:
            self.params += conv_layer.params
            
        for hidden_layer in self.hidden_layers:
            self.params += hidden_layer.params
            
        self.params += [self.W, self.b]
        
        self.thX = T.tensor4('X', dtype='float32')
        self.thT = T.matrix('T')
    
    
    def forward(self, X):
        Z = X
        
        for c in self.conv_layers:
            Z = c.forward(Z)
            
        Z = Z.flatten(ndim=2)
        for h in self.hidden_layers:
            Z = h.forward(Z)
    
        return T.nnet.softmax( Z.dot(self.W) + self.b )
    
    def fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate=0.0001, mu=0.9, decay=0.99, epochs=5, batch_sz=100):
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        decay = np.float32(decay)
        eps = np.float32(10e-10)
        
        Xtrain = Xtrain.astype(np.float32)
        Ytrain = Ytrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.astype(np.float32)
        
        
        Yp = self.forward(self.thX)
        cost = -(self.thT*T.log(Yp)).mean()
        # Find all gradients at once
        grads = T.grad(cost, self.params)

        updates = [ (p, p - learning_rate*g) for p, g in zip(self.params, grads) ]
        
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
                    print('Epoch:', i, 'Accuracy:', 1 - err, 'Cost:', c)
        
        return costs, errors
        
     
def main():    
    X, Y = get_preprocessed_image_data()
    Y = Y.astype(np.float32)
    Xtrain, Ytrain = X[:-1000], Y[:-1000]
    Xtest, Ytest = X[-1000:], Y[-1000:]
    
    epochs = 150
    learning_rate = np.float32(0.00001)
    N = Xtrain.shape[0]
    batch_sz = 500
    n_batches = N // batch_sz
    
    poolsz = (2, 2)
    K = 10
    # Hidden layer size
    hl_sz = 300
    # Momentum factor
    mu = np.float32(0.9)
    
    model = CNN(input_shape=(1, 28, 28), K=10,
                conv_layer_sizes=[(30, 5, 5), (40, 5, 5)],
                hidden_layer_sizes=[300, 100]
               )
    costs, errors = model.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate, epochs=epochs)
    create_graph(errors, 'Cnntest_error_{}.png'.format(1 - errors[-1]), 'Error', 'Iterations')
    create_graph(costs, 'Cnntest_cost.png', 'Cost', 'Iterations')
    
if __name__ == '__main__':
    main()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        