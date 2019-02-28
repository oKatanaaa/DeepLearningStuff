from theano_version.utils import get_preprocessed_data
import matplotlib.pyplot as plt
from theano_version.multilayer_NN import MultiLayerNN as TheanoNN
#from NN_vanilla import TheanoNN as VanillaNN
#from NN_momentum import TheanoNN as MomentumNN
#from NN_nesterov import TheanoNN as NMomentumNN
#from NN_adagrad import TheanoNN as AdagradNN
#from NN_rmsprop import TheanoNN as RMSpropNN
#from NN_adam import TheanoNN as AdamNN
#from NN_rmsmomentum import TheanoNN as RMSMomentumMM

def main():
    X, Y = get_preprocessed_data()
    N, D = X.shape
    # Number of hidden neurons
    H = 100
    K = Y.shape[1]
    
    # Split data
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    
    Xtest = X[-1000:]
    Ytest = Y[-1000:]
    
    # Theano NN with RMSprop with momentum
    model1 = TheanoNN(D, K, [500, 300], [1, 1])
    _, th_error1 = model1.fit(Xtrain, Ytrain, Xtest, Ytest,
                         epochs=500, learning_rate=0.0001, batch_sz=500)
    
    model2 = TheanoNN(D, K, [500, 300], [0.8, 0.6])
    _, th_error2 = model2.fit(Xtrain, Ytrain, Xtest, Ytest,
                         epochs=500, learning_rate=0.0001, batch_sz=500)
    
    model3 = TheanoNN(D, K, [500, 300], [0.5, 0.5])
    _, th_error3 = model3.fit(Xtrain, Ytrain, Xtest, Ytest,
                         epochs=500, learning_rate=0.0001, batch_sz=500)

    
    # Create error graphs
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(th_error1, label='Theano NN1')
    axes.plot(th_error2, label='Theano NN2')
    axes.plot(th_error3, label='Theano NN3')
    axes.legend()
    
    axes.set_ylabel('Error')
    axes.set_xlabel('Iteration')
    
    fig.savefig('Test5_theano.png')
    
if __name__ == '__main__':
    main()