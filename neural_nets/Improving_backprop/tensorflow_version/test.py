from utils import get_preprocessed_data
import matplotlib.pyplot as plt
from nn_vanilla import TensorflowNN as VanillaNN
#from NN_momentum import TheanoNN as MomentumNN
#from NN_nesterov import TheanoNN as NMomentumNN
#from NN_adagrad import TheanoNN as AdagradNN
from nn_rmsprop import TensorflowNN as RMSpropNN
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
    
    # NN with vanilla GD
    model1 = VanillaNN(D, H, K)
    _, v_error = model1.fit(Xtrain, Ytrain, Xtest, Ytest,
                         epochs=100, learning_rate=0.00001, batch_sz=500)
    
    # NN with momentum GD
    #model2 = MomentumNN(D, H, K)
    #_, m_error = model2.fit(Xtrain, Ytrain, Xtest, Ytest,
    #                     epochs=100, learning_rate=0.00001, batch_sz=500)
    
    # NN with Nesterov momentum GD
    #model3 = NMomentumNN(D, H, K)
    #_, nm_error = model3.fit(Xtrain, Ytrain, Xtest, Ytest,
    #                     epochs=100, learning_rate=0.00001, batch_sz=500)
    
    # NN with Adagard
    #model4 = AdagradNN(D, H, K)
    #ag_error = model4.fit(Xtrain, Ytrain, Xtest, Ytest,
    #                     epochs=100, learning_rate=0.001, batch_sz=500)
    
    # NN with RMSprop
    model5 = RMSpropNN(D, H, K)
    _, rms_error = model5.fit(Xtrain, Ytrain, Xtest, Ytest,
                         epochs=100, learning_rate=0.001, batch_sz=500, decay=0.999)
    
    # NN with Adam
    #model6 = AdamNN(D, H, K)
    #_, am_error = model6.fit(Xtrain, Ytrain, Xtest, Ytest,
    #                     epochs=100, learning_rate=0.001, batch_sz=500)
    
    # NN with custom algorithm
    #model7 = RMSMomentumMM(D, H, K)
    #cs_error = model7.fit(Xtrain, Ytrain, Xtest, Ytest,
    #                    epochs=80, learning_rate=0.001, batch_sz=500)
    
    # Create error graphs
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(v_error, label='Vanilla')
    #axes.plot(m_error, label='Momentum')
    #axes.plot(nm_error, label='Nesterov momentum')
    #axes.plot(ag_error, label='Adagrad')
    axes.plot(rms_error, label='RMSprop')
    #axes.plot(am_error, label='Adam')
    #axes.plot(cs_error, label='Custom algorithm')
    axes.legend()
    
    axes.set_ylabel('Error')
    axes.set_xlabel('Iteration')
    
    fig.savefig('Test2_vanilla_rmsprop.png')
    
if __name__ == '__main__':
    main()