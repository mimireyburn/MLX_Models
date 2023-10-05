# Labratory 2
import numpy as np


#What does this code do?
# It creates a neural network with 2 inputs, 3 neurons in the hidden layer and 1 output.

W1 = np.random.randn(3,2) # 3 neurons x 2 dimensional data (2 outputs, 0 or 1) array, values from standard normal distribution
B1 = np.random.randn(3) # 3 neurons
W2 = np.random.randn(1,3) # 1 neuron x 3 dimensional data 
B2 = np.random.randn(1)

def sigm(X, W, B): #sigmoid function

    M = 1/(1+np.exp(-(X.dot(W.T)+B)))

    return M

def Forward(X, W1, B1, W2, B2):

    #first layer
    H = sigm(X, W1, B1) #same sigmoid activation function for first and second layer. Three input vectors for 3 neurons. W and B also have three dimensions. B has one dimension, 3 values. 
    # print(H)
    #second layer
    Y = sigm(H, W2, B2)

    return Y, H #We return both the final output and the output from the hidden layer 

def diff_B2(Z,Y): 
    dB = (Z-Y)*Y*(1-Y) 
    return dB.sum(axis=0)

def diff_W2(H, Z, Y): 
    dW = (Z-Y)*Y*(1-Y)
    return H.T.dot(dW)

def diff_W1(X, H, Z, Y, W2): 
    dZ = (Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)
    return X.T.dot(dZ) 

def diff_B1(Z, Y, W2, H):
    return ((Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)).sum(axis=0)

# Create training set
X = np.random.randint(2, size=[15,2]) # 2 means 0 or 1, 2 x 15 matrix
Z = np.array([X[:,0] | X[:,1]]).T 

# Create test set
X_Test = np.random.randint(2, size=[15,2])
Y_Test = np.array([X_Test[:,0] | X_Test[:, 1]]).T

learning_rate = 1e-2

for epoch in range(10000):

    Y, H = Forward(X, W1, B1, W2, B2)

    W2 += learning_rate * diff_W2(H, Z, Y).T
    B2 += learning_rate * diff_B2(Z,Y)
    W1 += learning_rate * diff_W1(X, H, Z, Y, W2).T
    B1 += learning_rate * diff_B1(Z, Y, W2, H)
    if not epoch % 50: 
        Accuracy = 1 - np.mean((Z-Y)**2)
        print('Epoch: ', epoch, 'Accuracy: ', Accuracy)

