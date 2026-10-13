if __name__ == '__main__':

import numpy as np
from data import Data_creation

# Data creation
X,Y = Data_creation()
class Dense_layer:
    # Dense Layer
    def __init__(self, n_inputs, n_neurons):
        self.weigths = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    # forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs , self.weigths) + self.biases

    def backwards(self, d_values):
        self.dweigths = np.dot(self.inputs.T, d_values))
        self.biases = np.sum(d_values,axis=0,keepdims= True))
        self.dinputs = np.dot(d_values,self.weigths.T))

def ReLU_derivate(inputs):
    return (inputs > 0).astype(float)

class Activation_ReLU:
    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    # Backpropagation
    def backwards(self, d_values):
        self.dinputs = d_values.copy()
        self.dinputs[self.inputs <= 0 ] == 0

class Activation_softmax:
     # forwarding
    def forward(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis= 1, keepdims= True))
        probabilities = exp_values/np.sum(exp_values, axis= 1, keepdims= True)
        self.output= probabilities


class Loss_categorical_cross_entropy():
    #loss function
    def calculate(self, output,Y):
        sample_losses = self.forward(output,Y)
        data_loss= np.mean(sample_losses)
        return data_loss

  # for categorical labels
    def forward(self,y_pred,Y):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7)
        if len(Y.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),Y]
    # mask values for one-hot encoded labels
        elif len(Y.shape) ==2:
            correct_confidences = np.sum(y_pred_clipped*Y, axis=1)
        neg_log_likelihood = -np.log(correct_confidences)
        return neg_log_likelihood

    def backwards(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if (y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        #derivate
        self.dinputs = (-1 * y_true)/dvalues
        #normalization
        self.dinputs = self.dinputs / samples


layer1 = Dense_layer(2,3)
activation1 = Activation_ReLU()
layer2 = Dense_layer(3,3)
activation2 = Activation_softmax()
L_c_c_e = Loss_categorical_cross_entropy()
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
neg_log_likelihood = L_c_c_e.calculate(activation2.output,Y)
print(neg_log_likelihood)
print(activation2.output[:5])


