import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D

def generate_figs(ax, fig, name='def'):
    angles = [(10, 150), (15, 180), (15, 240)]
    for angle in angles:
        ax.view_init(angle[0], angle[1])
        fig.savefig(name+str(angle)+'.png')
    

N = 1000
D = 2

X = np.random.randn(N, D) - 2
Y = X[:, 0] * X[:, 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
ax.view_init(10, 150)
generate_figs(ax, fig)

# hidden layer size
M = 10

W1 = np.random.randn(D, M)
b1 = np.zeros(M).reshape(M,1)
W2 = np.random.randn(M, 1)
b2 = 0

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def square_error(Y, T):
    return np.power(Y - T, 2).sum()

def forward(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1)
    return Z.dot(W2) + b2, Z


                
learning_rate = 0.0001
reg = 0.
epochs = 1000

Xtrain = X[:-100]
Ytrain = Y[:-100].reshape(900, 1)

Xtest = X[-100:]
Ytest = Y[-100:].reshape(100, 1)

ctrain = []
ctest = []

ones = np.ones(900)

for i in range(epochs):
    Yp_test, _= forward(Xtest, W1, b1, W2, b2)
    Yp_train, Z = forward(Xtrain, W1, b1, W2, b2)
    print(str(Ytrain.shape))
    print(str(Yp_train.shape))
    print(str(Z.shape))
    print((Yp_train - Ytrain).shape)
    print((Yp_train - Ytrain).T.shape)
    print((Yp_train - Ytrain).T.dot(Z).shape)
    print(W2.shape)
    print(b1.shape)
    print((Yp_train - Ytrain).T.dot(2*Z*(1 - Z)*np.outer(ones, W2)).T.shape)
    cts = square_error(Yp_test, Ytest)
    ctr = square_error(Yp_train, Ytrain)
    
    ctest.append(cts)
    ctrain.append(ctr)
    
    if i % 10 == 0:
        print("Train error: ", ctr)
        print("Test error: ", cts)
       
    W2 -= learning_rate * (((Yp_train - Ytrain).T.dot(Z) * 2).T + W2*reg)
    b2 -= learning_rate * ((Yp_train - Ytrain).sum() * 2 + b2*reg)
    
    dZ = np.outer(Yp_train - Ytrain, W2) * Z*(1 - Z)
    W1 -= learning_rate * (Xtrain.T.dot(dZ) + W1*reg)
    b1 -= learning_rate * (Yp_train - Ytrain).T.dot(2*Z*(1 - Z)*np.outer(ones, W2))
        

newY = forward(X, W1, b1, W2, b2)
fig2 = Axes3D.scatter(xs=X, ys=newY)
fig2.savefig('fig2.png')

plot1 = plt.plot(ctest, 'ctest')
plot1.savefig('plot1.png')

plot2 = plt.plot(ctrain, 'ctrain')
plot1.savefig('plot2.png')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        