import numpy as np
import tensorflow as tf

class ConvLayer(object):
    def __init__(self, nf, kw, kh, nof, stride=1, padding='SAME', activation=tf.nn.relu):
        """
        nf - number of input feature maps (treat it as number of color channels if this layer is the first one).
        kw - kernel width.
        kh - kernel height.
        nof - number of output feature maps
        """
        
        self.shape = (kw, kh, nf, nof)
        self.stride = stride
        self.padding = padding
        self.f = activation
        W = np.random.randn(*self.shape) * np.sqrt(2.0 / np.prod(self.shape[:-1]))
        b = np.zeros(nof)
        
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]
        
        
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.f == None:
            return conv_out
        
        return self.f(conv_out)
    
    
    # For Keras interface
    def get_params(self):
        return self.params
    
    
    def copyFromKerasLayers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        
        self.session.run((op1, op2))
    

class DenseLayer(object):
    def __init__(self, input_shape, neuron_number, activation=tf.nn.relu):
        """
        input_shape - the shape of the input.
        neuron_number - number of the neurons in the dense layer.
        activation - activation function is performed on the output of the layer. 
        """
        self.f = activation
        W = np.random.randn(input_shape, neuron_number)
        # Perform Xavier initialization
        if activation == tf.nn.relu:
            W /= (input_shape + neuron_number) / 2
        # Perform Lasange initialization
        else:
            W *=  np.sqrt(12 / (input_shape + neuron_number))
        
        b = np.zeros(neuron_number)
        
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

        
    def forward(self, X):
        out = tf.matmul(X, self.W) + self.b
        if self.f == None:
            return out
        
        return self.f(out)
    
    
    def copyFromKerasLayers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1, op2))
    
    
    # For Keras interface
    def get_params(self):
        return self.params
                 
                 
class BatchNormLayer(object):
    def __init__(self, D):
        """
        decay - this argument is responsible for how fast batchnorm layer is trained. Values between 0.9 and 0.999 are 
        commonly used.
        D - number of tensors to be normalized.
        """
        # Mean which is used during normal work of the neural net
        self.D = D
        mean_init = np.zeros(D)
        var_init = np.ones(D)
        # These variables are needed to change the mean and variance of the batch after
        # the batchnormalization
        beta = np.zeros(D)
        gamma = np.zeros(D)
        
        self.running_mean = tf.Variable(mean_init.astype(np.float32), trainable=False)
        self.running_variance = tf.Variable(var_init.astype(np.float32), trainable=False)
        self.beta = tf.Variable(beta.astype(np.float32))
        self.gamma = tf.Variable(gamma.astype(np.float32))
        self.params = [self.running_mean, self.running_variance, self.gamma, self.beta]
        
        
    def forward(self, X, is_training=False, decay=0.9):
        if is_training:
            batch_mean, batch_var = tf.nn.moments(X, [0])
            update_running_mean = tf.assign(
                self.running_mean,
                self.running_mean*decay + batch_mean*(1 - decay)
            )
            update_running_variance = tf.assign(
                self.running_variance,
                self.running_variance*decay + batch_var*(1 - decay)
            )
            with tf.control_dependencies([update_running_mean, update_running_variance]):
                out = tf.nn.batch_normalization(
                    X,
                    batch_mean,
                    batch_variance,
                    self.beta,
                    self.gamma,
                    1e-4
                )
        else:
            out = tf.nn.batch_normalization(
                    X,
                    self.running_mean,
                    self.running_variance,
                    self.beta,
                    self.gamma,
                    1e-4
                )
        return out
    
    
    # For Keras interface
    def get_params(self):
        return self.params
    
    
    def copyFromKerasLayers(self, layer):
        # only 1 layer to copy from
        # order:
        # gamma, beta, moving mean, moving variance
        gamma, beta, running_mean, running_variance = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_variance.assign(running_variance)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)
        self.session.run((op1, op2, op3, op4))
        
        
class MaxPoolLayer(object):
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
    
    
    def forward(self, X):
        return tf.nn.max_pool(
            X, 
            ksize=self.ksize, 
            strides=self.strides, 
            padding=self.padding
        )
    
    
    # For Keras interface
    def get_params(self):
        return []
    
    
class AvgPoolLayer(object):
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
    
    
    def forward(self, X):
        return tf.nn.avg_pool(
            X, 
            ksize=self.ksize, 
            strides=self.strides, 
            padding=self.padding
        )
    
    
    # For Keras interface
    def get_params(self):
        return []

    
class ActivationLayer(object):
    def __init__(self, activation=tf.nn.relu):
        self.f = activation
    
    
    def forward(self, X):
        return self.f(X)
                 
        
    # For Keras interface
    def get_params(self):
        return []
                 

# This class is copied from Udemy discussion.
# It solves the problem with testing partial ResNet built by me.
# I need to add this layer so that output shape of my ResNet matches
# Keras' esNet
class ZeroPaddingLayer(object):
    def __init__(self, padding):
        self.padding=padding

        
    def forward(self, X):
        return tf.keras.layers.ZeroPadding2D(padding=self.padding)(X)

    
    def get_params(self):
        return []
                 
                 
class FlattenLayer(object):
    def __inti__(self):
        pass
    
    
    def forward(self, X):
        return tf.contrib.layers.flatten(X)
    
    
    def get_params(self):
        return []
                 
                 