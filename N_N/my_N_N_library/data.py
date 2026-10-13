import numpy as np
import matplotlib.pyplot as plt
def Data_creation(N=100,K=3,D=2):
    # N = number of points per class
    # D = dimensionality
    # K =  number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return(X,y)

if __name__ == '__main__':
    print("Testing data creation script...")
    X, y = Data_creation()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
