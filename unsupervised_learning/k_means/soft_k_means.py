import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot_colored_points(X, colors):
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.scatter(X[:,0], X[:,1], c=colors)
    fig.savefig('k_mean.png')


def plot_cost(cost):
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(cost)
    fig.savefig('cost.png')

# Function for calculation square Euclidean distance between two points
def d(x, y):
    diff = x - y
    return diff.dot(diff)


def cost(X, M, R):
    cost = 0
    for n in range(len(X)):
        for m in range(len(M)):
            cost += R[n,m]*d(X[n], M[m])
    
    return cost


def plot_k_means(X, K, max_iter=20, beta=1):
    N, D = X.shape
    
    # Resposibility matrix
    R = np.zeros((N, K))
    # Centers of the clusters
    M = np.zeros((K, D))
    # Cost array
    costs = []
    
    # Randomly initialize the centers
    for k in range(K):
        M[k] = X[np.random.choice(N)]
        
    # Start soft k-means
    for i in range(max_iter):
        for n in range(N):
            for k in range(K):
                R[n, k] = np.exp(-beta*d(X[n], M[k])) / np.sum( np.exp(-beta*d(X[n], M[j])) for j in range(K))
        for k in range(K):
            M[k] = R[:,k].dot(X) / R[:,k].sum()
        
        costs.append(cost(X, M, R))
    plot_cost(costs)
    print(R.shape)
    colors = np.random.random((K,3))
    p_colors = R.dot(colors)
    print(p_colors.shape)
    plot_colored_points(X, p_colors)
                
                                                               


if __name__ == '__main__':
    # There will be 3 mean points e.g. 3 clusters
    
    
    
    mean1 = np.array([0, 0])
    mean2 = np.array([4, 0])
    mean3 = np.array([4, 4])
    
    N = 900
    
    X = np.random.randn(N, 2)
    
    X[:300] += mean1
    X[300:600] += mean2
    X[600:] += mean3
    
    plt.scatter(X[:,0], X[:,1])
    plt.savefig('scatter.png')
    
    plot_k_means(X, K=3, beta=1)