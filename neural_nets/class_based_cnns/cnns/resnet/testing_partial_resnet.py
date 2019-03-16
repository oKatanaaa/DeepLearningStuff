import tensorflow as tf
import numpy as np
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


from resnet_conv_block import ConvBlock
from layers import BatchNormLayer, ConvLayer, ActivationLayer, MaxPoolLayer, ZeroPaddingLayer

class PartialResNet:
    def __init__(self):
        
        self.layers = [
            # Before conv block
            # note: need tp add ZeroPaddingLayer in order to compare results with
            # keras ResNet
            ZeroPaddingLayer(padding=3),
            ConvLayer(nf=3, kw=7, kh=7, nof=64, stride=2, padding='VALID'),
            BatchNormLayer(64),
            # ReLU activation
            ActivationLayer(),
            ZeroPaddingLayer(padding=1),
            MaxPoolLayer(ksize=[1, 3, 3, 1], padding='VALID'),
            # Conv block
            ConvBlock(input_depth=64, fm_sizes=[64, 64, 256])
        ]
        
        self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.output = self.forward(self.input_)
    
    
    def copyFromKerasLayers(self, layers):
        self.layers[1].copyFromKerasLayers(layers[2])
        self.layers[2].copyFromKerasLayers(layers[3])
        self.layers[6].copyFromKerasLayers(layers[7:])
        

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    
    def predict(self, X):
        assert(self.session is not None)
        return self.session.run(
            self.output,
            feed_dict={self.input_: X}
        )
    
    
    def set_session(self, session):
        self.session = session
        self.layers[1].session = session
        self.layers[2].session = session
        self.layers[6].set_session(session)
        

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
            
            
if __name__ == '__main__':
    resnet = ResNet50(weights='imagenet')
    
    partial_model = Model(
        inputs=resnet.input,
        output=resnet.layers[18].output
    )
    print(partial_model.summary())
    
    my_partial_resnet = PartialResNet()
    
    X = np.random.random((1, 224, 224, 3))
    
    keras_output = partial_model.predict(X)
    
    init = tf.variables_initializer(my_partial_resnet.get_params())
    
    # note: starting a new session messes up the Keras model
    session = keras.backend.get_session()
    my_partial_resnet.set_session(session)
    session.run(init)
    
    # first, just make sure we can get any output
    first_output = my_partial_resnet.predict(X)
    print('first_output.shape', first_output.shape)
    
    # Copy params from Keras model
    my_partial_resnet.copyFromKerasLayers(partial_model.layers)
    
    # compare the 2 models
    output = my_partial_resnet.predict(X)
    diff = np.abs(output - keras_output).sum()
    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = {}".format(diff))