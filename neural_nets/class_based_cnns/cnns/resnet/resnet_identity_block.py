from layers import ConvLayer, DenseLayer, BatchNormLayer, MaxPoolLayer, ActivationLayer
import tensorflow as tf
import numpy as np

class IdentityBlock:
    def __init__(self, input_depth, fm_sizes, activation=tf.nn.relu):
        """
        input_depth - number of the feature maps(trear it as color channels if the block is
        the first component in the network)
        fm_sizes - list of sizes of output feature maps of conv layers. There must be 3 number
        in the list since they are parameters for the 3 conv layers in the main branch. 
        Example: [32, 32, 32]
        """
        assert(len(fm_sizes) == 3)
        self.f = activation
        
        # Init main branch
        # Conv -> BN -> relu -> Conv -> BN -> relu -> Conv -> BN
        self.conv1 = ConvLayer(input_depth, 1, 1, fm_sizes[0], padding='VALID', activation=None)
        self.bn1 = BatchNormLayer(fm_sizes[0])
        self.conv2 = ConvLayer(fm_sizes[0], 3, 3, fm_sizes[1], activation=None)
        self.bn2 = BatchNormLayer(fm_sizes[1])
        self.conv3 = ConvLayer(fm_sizes[1], 1, 1, fm_sizes[2], padding='VALID', activation=None)
        self.bn3 = BatchNormLayer(fm_sizes[2])
        
        self.layers = [
            self.conv1, self.bn1,
            self.conv2, self.bn2,
            self.conv3, self.bn3
        ]
        
        # this will not be used when input passed in from
        # a previous layer
        self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, input_depth))
        self.output = self.forward(self.input_)
        
        
    def forward(self, X):
        # main branch
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.f(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.f(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)
        
        return self.f(FX + X)
    
    
    def predict(self, X):
        assert(self.session is not None)
        return self.session.run(
          self.output,
          feed_dict={self.input_: X}
        )

    def set_session(self, session):
        # need to make this a session
        # so assignment happens on sublayers too
        self.session = session
        self.conv1.session = session
        self.bn1.session = session
        self.conv2.session = session
        self.bn2.session = session
        self.conv3.session = session
        self.bn3.session = session

    def copyFromKerasLayers(self, layers):
        assert(len(layers) == 10)
        # <keras.layers.convolutional.Conv2D at 0x7fa44255ff28>,
        # <keras.layers.normalization.BatchNormalization at 0x7fa44250e7b8>,
        # <keras.layers.core.Activation at 0x7fa44252d9e8>,
        # <keras.layers.convolutional.Conv2D at 0x7fa44253af60>,
        # <keras.layers.normalization.BatchNormalization at 0x7fa4424e4f60>,
        # <keras.layers.core.Activation at 0x7fa442494828>,
        # <keras.layers.convolutional.Conv2D at 0x7fa4424a2da0>,
        # <keras.layers.normalization.BatchNormalization at 0x7fa44244eda0>,
        # <keras.layers.merge.Add at 0x7fa44245d5c0>,
        # <keras.layers.core.Activation at 0x7fa44240aba8>
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[7])

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
    
if __name__ == '__main__':
    identity_block = IdentityBlock(input_depth=256, fm_sizes=[64, 64, 256])

    # make a fake image
    X = np.random.random((1, 224, 224, 256))

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        identity_block.set_session(session)
        session.run(init)

        output = identity_block.predict(X)
        print("output.shape:", output.shape)