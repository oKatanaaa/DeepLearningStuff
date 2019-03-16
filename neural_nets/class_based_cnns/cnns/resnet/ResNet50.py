import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input, decode_predictions
from resnet_conv_block import ConvBlock
from resnet_identity_block import IdentityBlock
from layers import BatchNormLayer, ConvLayer, ActivationLayer, MaxPoolLayer, ZeroPaddingLayer, AvgPoolLayer, FlattenLayer, DenseLayer
import numpy as np

# NOTE: dependent on your Keras version
#       this script used 2.1.1
# [<keras.engine.topology.InputLayer at 0x112fe4358>,
#  <keras.layers.convolutional.Conv2D at 0x112fe46a0>,
#  <keras.layers.normalization.BatchNormalization at 0x112fe4630>,
#  <keras.layers.core.Activation at 0x112fe4eb8>,
#  <keras.layers.pooling.MaxPooling2D at 0x10ed4be48>,
#
#  ConvBlock
#  IdentityBlock x 2
#
#  ConvBlock
#  IdentityBlock x 3
#
#  ConvBlock
#  IdentityBlock x 5
#
#  ConvBlock
#  IdentityBlock x 2
#
#  AveragePooling2D
#  Flatten
#  Dense (Softmax)
# ]


class TFResNet:
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
            ConvBlock(input_depth=64, fm_sizes=[64, 64, 256]),
            # Identity block x2
            IdentityBlock(input_depth=256, fm_sizes=[64, 64, 256]),
            IdentityBlock(input_depth=256, fm_sizes=[64, 64, 256]),
            # Conv block
            ConvBlock(input_depth=256, fm_sizes=[128, 128, 512], stride=2),
            # Identity block x3
            IdentityBlock(input_depth=512, fm_sizes=[128, 128, 512]),
            IdentityBlock(input_depth=512, fm_sizes=[128, 128, 512]),
            IdentityBlock(input_depth=512, fm_sizes=[128, 128, 512]),
            # Conv block
            ConvBlock(input_depth=512, fm_sizes=[256, 256, 1024], stride=2),
            # Identity block x5
            IdentityBlock(input_depth=1024, fm_sizes=[256, 256, 1024]),
            IdentityBlock(input_depth=1024, fm_sizes=[256, 256, 1024]),
            IdentityBlock(input_depth=1024, fm_sizes=[256, 256, 1024]),
            IdentityBlock(input_depth=1024, fm_sizes=[256, 256, 1024]),
            IdentityBlock(input_depth=1024, fm_sizes=[256, 256, 1024]),
            # Conv block
            ConvBlock(input_depth=1024, fm_sizes=[512, 512, 2048], stride=2),
            # Identity block x2
            IdentityBlock(input_depth=2048, fm_sizes=[512, 512, 2048]),
            IdentityBlock(input_depth=2048, fm_sizes=[512, 512, 2048]),
            # pool / flatten / dense
            AvgPoolLayer(ksize=[1, 7, 7, 1], padding='VALID'),
            FlattenLayer(),
            DenseLayer(input_shape=2048, neuron_number=1000)
        ]
        self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.output = self.forward(self.input_)
        
        
    def copyFromKerasLayers(self, layers):
        # conv
        self.layers[1].copyFromKerasLayers(layers[2])
        # bn
        self.layers[2].copyFromKerasLayers(layers[3])
        # cb
        self.layers[6].copyFromKerasLayers(layers[7:19]) # size=12
        # id x2
        self.layers[7].copyFromKerasLayers(layers[19:29]) # size=10
        self.layers[8].copyFromKerasLayers(layers[29:39])
        # cb
        self.layers[9].copyFromKerasLayers(layers[39:51])
        # id x3
        self.layers[10].copyFromKerasLayers(layers[51:61])
        self.layers[11].copyFromKerasLayers(layers[61:71])
        self.layers[12].copyFromKerasLayers(layers[71:81])
        # cb
        self.layers[13].copyFromKerasLayers(layers[81:93])
        # id x5
        self.layers[14].copyFromKerasLayers(layers[93:103])
        self.layers[15].copyFromKerasLayers(layers[103:113])
        self.layers[16].copyFromKerasLayers(layers[113:123])
        self.layers[17].copyFromKerasLayers(layers[123:133])
        self.layers[18].copyFromKerasLayers(layers[133:143])
        # cb
        self.layers[19].copyFromKerasLayers(layers[143:155])
        # id X2
        self.layers[20].copyFromKerasLayers(layers[155:165])
        self.layers[21].copyFromKerasLayers(layers[165:175])
        # dense
        self.layers[24].copyFromKerasLayers(layers[176])
    
    
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
        for layer in self.layers:
            if isinstance(layer, ConvBlock) or isinstance(layer, IdentityBlock):
                layer.set_session(session)
            else:
                layer.session = session
    
    
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()

            
            
if __name__ == '__main__':
    # you can also set weights to None, it doesn't matter
    resnet_ = ResNet50(weights='imagenet')

    # make a new resnet without the softmax
    x = resnet_.layers[-2].output
    W, b = resnet_.layers[-1].get_weights()
    y = Dense(1000)(x)
    resnet = Model(resnet_.input, y)
    resnet.layers[-1].set_weights([W, b])

    # you can determine the correct layer
    # by looking at resnet.layers in the console
    partial_model = Model(
        inputs=resnet.input,
        outputs=resnet.layers[176].output
    )

    # maybe useful when building your model
    # to look at the layers you're trying to copy
    print(partial_model.summary())

    # create an instance of our own model
    my_partial_resnet = TFResNet()

    # make a fake image
    X = np.random.random((1, 224, 224, 3))

    # get keras output
    keras_output = partial_model.predict(X)

    ### get my model output ###

    # init only the variables in our net
    init = tf.variables_initializer(my_partial_resnet.get_params())

    # note: starting a new session messes up the Keras model
    session = keras.backend.get_session()
    my_partial_resnet.set_session(session)
    session.run(init)

    # first, just make sure we can get any output
    first_output = my_partial_resnet.predict(X)
    print("first_output.shape:", first_output.shape)
    print("keras_output.shape:", keras_output.shape)
    # copy params from Keras model
    my_partial_resnet.copyFromKerasLayers(partial_model.layers)

    # compare the 2 models
    output = my_partial_resnet.predict(X)
    diff = np.abs(output - keras_output).sum()
    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = %s" % diff)
            
            
            
            
            
            