import theano as th
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
import numpy as np
from utils import get_preprocessed_image_data, error_rate, create_graph
from sklearn.utils import shuffle

class ConvPoolLayer(object):
    def __init__(self, mi, mo, fw=5, fh=5, pool=False):
        """
        mi - maps input
        mo - maps output
        fw - filter width
        fh - filter height
        """
        # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
        self.shape = (fw, fh, mi, mo)
        self.pool = pool
        W = np.random.randn(*self.shape)  * np.sqrt(2.0 / np.prod(shape[:-1]))
        W = W.astype(np.float32)
        b = np.zeros(mo, dtype=np.float32)
        
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)
        
        
    
    
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(cov_out, self.b)
        if pool:
            conv_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')
        return tf.relu(conv_out)
    
    
class HiddenLayer(object):
    def __init__(self, M1, M2):
        W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
        W = W.astype(np.float32)
        b = np.zeros(M2)
        b = b.astype(np.float32)
        
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)

        
    def forward(self, X):
        return tf.relu(tf.matmul(X, self.W) + self.b)
        

class CNN(object):
    def __init__(self, input_shape, K, conv_layer_sizes, hidden_layer_sizes):
        """
        input_shape - tuple with sizes of the input image. Data: (image_width, image_height, color_channels)
        K - number of classes. Data: int
        conv_layer_sizes - list of tuples with kernel sizes. Data: [(kernel_width=int, kernel_height=int, number_feature_maps=int, pool=bool)]
        hidden_layer_sizes - list of hidden layer sizes. Data: [int, int, ...]
        """
        self.conv_layer_sizes = conv_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        
        # Init conv layers
        w, h, c = input_shape
        
        im = c # input maps
        self.conv_layers = []
        for k_w, k_h, n_f, pool in conv_layer_sizes:
            # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
            conv_layer = ConvPoolLayer(im, n_f, k_w, k_h, pool)
            self.conv_layer.append(conv_layer)
            
            im = n_f
            if pool:
                w = w // 2
                h = h // 2
        
        # Init hidden layers
        self.hidden_layers = []
        M1 = w*h*im
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        