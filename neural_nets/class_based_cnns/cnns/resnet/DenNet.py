from .resnet_identity_block import IdentityBlock
import tensorflow as tf
import numpy as np
from .layers import ConvLayer, BatchNormLayer, ActivationLayer, AvgPoolLayer, FlattenLayer, DenseLayer
from sklearn.utils import shuffle
from .utils import error_rate, create_graph


class DenIdentityBlock:
    def __init__(self, input_depth, fm_sizes, activation=tf.nn.relu):
        """
        input_depth - number of the feature maps(trear it as color channels if the block is
        the first component in the network)
        fm_sizes - list of sizes of output feature maps of conv layers. There must be 3 number
        in the list since they are parameters for the 3 conv layers in the main branch. 
        Example: [32, 32]
        """
        assert(len(fm_sizes) == 2)
        self.f = activation
        
        # Init main branch
        # Conv -> BN -> relu -> Conv -> BN -> relu -> Conv -> BN
        self.conv1 = ConvLayer(input_depth, 5, 5, fm_sizes[0], padding='SAME', activation=None)
        self.bn1 = BatchNormLayer(fm_sizes[0])
        self.conv2 = ConvLayer(fm_sizes[0], 4, 4, fm_sizes[1], activation=None)
        self.bn2 = BatchNormLayer(fm_sizes[1])
        
        self.layers = [
            self.conv1, self.bn1,
            self.conv2, self.bn2,
        ]
        
        # this will not be used when input passed in from
        # a previous layer
        self.input_ = tf.placeholder(tf.float32, shape=(1, 224, 224, input_depth))
        self.output = self.forward(self.input_)
        
        
    def forward(self, X):
        # main branch
        X = tf.cast(X, tf.float16)
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.f(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = tf.cast(FX, tf.float16)
        
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


class DenConvBlock(object):
    def __init__(self, input_depth, fm_sizes, stride=1, activation=tf.nn.relu):
        """
        input_depth - number of the feature maps(trear it as color channels if the block is
        the first component in the network)
        fm_sizes - list of sizes of output feature maps of conv layers. There must be 3 number
        in the list since they are parameters for the 3 conv layers in the main branch. 
        Example: [32, 32]
        """
        assert(len(fm_sizes) == 2)
        self.f = activation
        
        # Init main branch
        # Conv -> BN -> F() ----> Conv -> BN -> F() ----> Conv -> BN
        self.conv1 = ConvLayer(input_depth, 5, 5, fm_sizes[0], padding='SAME', activation=None)
        self.bn1 = BatchNormLayer(fm_sizes[0])
        self.conv2 = ConvLayer(fm_sizes[0], 3, 3, fm_sizes[1], stride, activation=None)
        self.bn2 = BatchNormLayer(fm_sizes[1])
        
        # Init shortcut branch
        # Conv -> BN
        self.convs = ConvLayer(input_depth, 3, 3, fm_sizes[1], stride, padding='SAME', activation=None)
        self.bns = BatchNormLayer(fm_sizes[1])
        
        self.layers = [
            self.conv1, self.bn1,
            self.conv2, self.bn2,
            self.convs, self.bns
        ]
        
    
    def forward(self, X):
        # Main branch
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.f(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = tf.cast(FX, tf.float16)
        
        # Shortcur branch
        SX = self.convs.forward(X)
        SX = self.bns.forward(SX)
        SX = tf.cast(SX, tf.float16)
        
        return self.f(FX + SX)
    
    
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
        self.convs.session = session
        self.bns.session = session


    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        
        return params

    
#  EXPERIMENTAL RESNET ARCHITECTURE FOR DENDRITE CLASSIFICATION
#  ConvLayer
#  BatchNorm
#  ReLU
#
#  ConvLayer
#  BatchNorm
#  ReLU
#
#  ConvBlock
#  IdentityBlock 
#
#  ConvBlock
#  IdentityBlock
#
#  ConvBlock
#  IdentityBlock
#
#  ConvBlock
#  IdentityBlock x2
#
#  AveragePooling2D
#  Flatten
#  Dense (Softmax)


class DenNet(object):
    def __init__(self, batch_sz):
        self.batch_sz = batch_sz
        self.layers = [
            # (M, 100, 100, 1)
            # Before residual blocks
            ConvLayer(nf=1, kw=5, kh=5, nof=32),
            BatchNormLayer(32),
            ActivationLayer(),
            # (M, 100, 100, 1)
            ConvLayer(nf=32, kw=3, kh=3, nof=64, stride=2),
            BatchNormLayer(64),
            ActivationLayer(),
            # (N, 50, 50, 64)
            DenIdentityBlock(input_depth=64, fm_sizes=[48, 64]),
            # (N, 50, 50, 64)
            DenConvBlock(input_depth=64, fm_sizes=[64, 128], stride=2),
            # (N, 25, 25, 128)
            DenIdentityBlock(input_depth=128, fm_sizes=[96, 128]),
            # (N, 25, 25, 128)
            DenConvBlock(input_depth=128, fm_sizes=[128, 256], stride=2),
            # (N, 13, 13, 256)
            DenIdentityBlock(input_depth=256, fm_sizes=[128, 256]),
            # (N, 13, 13, 256)
            DenConvBlock(input_depth=256, fm_sizes=[256, 512], stride=2),
            # (N, 7, 7, 512)
            DenIdentityBlock(input_depth=512, fm_sizes=[256, 512]),
            # avgpool / flatten / dense
            AvgPoolLayer(ksize=[1, 2, 2, 1], padding='VALID'),
            # (N, 1, 1, 512)
            FlattenLayer(),
            DenseLayer(input_shape=4608, neuron_number=1000),
            BatchNormLayer(1000),
            DenseLayer(input_shape=1000, neuron_number=10)
        ]
        
        self.input_ = tf.placeholder(tf.float16, shape=(None, 100, 100, 1))
        self.output = tf.nn.softmax(self.forward(self.input_))
        
        # Input-output variables for training
        self.X = tf.placeholder(tf.float16, shape=(batch_sz, 100, 100, 1), name='X')
        self.T = tf.placeholder(tf.float16, shape=(batch_sz, 10), name='Y')
        
        
    def set_session(self, session):
        self.session = session
        for layer in self.layers:
            if isinstance(layer, DenConvBlock) or isinstance(layer, IdentityBlock):
                layer.set_session(session)
            else:
                layer.session = session
        
        
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    
    def forward_train(self, X):
        for layer in self.layers:
            if isinstance(layer, BatchNormLayer):
                X = layer.forward(X, is_training=True)
            else:
                X = layer.forward(X)
        return X
    
    
    def predict(self, X):
        assert(self.session is not None)
        return self.session.run(
            self.output,
            feed_dict={self.input_: X}
        )
    
    
    def limited_fit(self, Xtrain, Ytrain, Xtest, Ytest, learning_rate=0.0001, mu=0.9, decay=0.999, epochs=9):
        Xtrain = Xtrain.astype(np.float16)
        Ytrain = Ytrain.astype(np.float16)
        Xtest = Xtest.astype(np.float16)
        Ytest = Ytest.astype(np.float16)
        
        
        Yish = self.forward_train(self.X)
        cost = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=self.T) )
        train_op = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5).minimize(cost)
        Yish_test = self.forward(self.X)
        cost_test = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish_test, labels=self.T) )
        predict_op = tf.argmax(Yish_test, 1)
        
        n_batches = Xtrain.shape[0] // self.batch_sz
        
        init_op = tf.global_variables_initializer()
        costs = []
        errors = []
        with tf.Session() as session:
            #self.set_session(session)
            session.run(init_op)
            
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                for j in range(n_batches):
                    Xbatch = Xtrain[j*self.batch_sz:(j+1)*self.batch_sz]
                    Ybatch = Ytrain[j*self.batch_sz:(j+1)*self.batch_sz]
                    
                    session.run(train_op, feed_dict={self.X: Xbatch, self.T: Ybatch})
                    
                    if j % 100 == 0:
                        test_cost = 0
                        predictions = np.zeros(len(Xtest))
                        for k in range(len(Xtest) // self.batch_sz):
                            Xtestbatch = Xtest[k*self.batch_sz:(k+1)*self.batch_sz]
                            Ytestbatch = Ytest[k*self.batch_sz:(k+1)*self.batch_sz]
                            test_cost += session.run(cost_test, feed_dict={self.X: Xtestbatch, self.T: Ytestbatch})
                            predictions[k*self.batch_sz:(k+1)*self.batch_sz] = session.run(predict_op, feed_dict={self.X: Xtestbatch})
                            
                        error = error_rate(predictions, np.argmax(Ytest, axis=1))
                        costs.append(test_cost)
                        errors.append(error)
                        print('Epoch:', i, 'Accuracy:', 1 - error, 'Cost:', test_cost)
                        
        return costs, errors
            
        
        
            
if __name__ == '__main__':
    # create an instance of our own model
    dennet = DenNet(1)

    # make a fake image
    X = np.random.random((1, 100, 100, 1))

    ### get my model output ###

    # init only the variables in our net
    init = tf.global_variables_initializer()

    # note: starting a new session messes up the Keras model
    session = tf.Session()
    dennet.set_session(session)
    session.run(init)

    # first, just make sure we can get any output
    first_output = dennet.predict(X)
    print("first_output.shape:", first_output.shape)