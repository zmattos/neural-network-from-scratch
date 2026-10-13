import numpy as np

# Layer class intialization:
class Layer_Dense:

    #init function
    def __init__(self,n_inputs,n_neurons):
        #weigth intialization with gaussian distribution
        self.weigths = 0.01 * np.random.randn(n_inputs,n_neurons)
        # bias initialization on matrix of zeros
        self.biases = np.zeros((1,n_neurons))

    #forwad-pass
    def forward(self,inputs):
        #memory of inputs for backprop:
        self.inputs = inputs
        # calculates ouput values for forward pass
        self.output = np.dot(inputs,self.weigths)+self.biases

    def backwards(self, dvalues):
        #weigths gradient(dL_dw)
        self.dweigths = np.dot(self.inputs.T, dvalues)
        #bias gradients(dL_db)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #input gradients(dL_dx)
        self.dinputs = np.dot(dvalues,self.weigths.T)

# Relu activation function
class Activation_ReLU:

    def forward(self,inputs):
        #inputs for backpropagation
        self.inputs = inputs
        # Relu function plied to the values
        self.output = np.maximum(0,inputs)

    def backwards(self, dvalues):
        #since we need to modify tha values we make a copy to not hinder the original matrix
        self.dinputs = dvalues.copy()
        # we aplied the derivate of ReLU
        self.dinputs[self.inputs <= 0 ] = 0

#softmaax activation (only forward)
class activation_softmax:

    def forward(self, inputs):
        #get unormalized values
        not_normal_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
        #normalize values
        probabilities = not_normal_values/np.sum(not_normal_values, axis=1, keepdims = True)
        # output
        self.output = probabilities


#categorical cross entropy()
class cat_cross_entropy:
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

class Activation_softmax_Loss_categoricalcrossentropy:

    def __init__(self):
        self.activation = activation_softmax()
        self.loss = cat_cross_entropy()


    def forward(self, inputs, y_true):
        #outputs layer activation function:
        self.activation.forward(inputs)
        # set the output
        self.output = self.activation.output
        # calculate and return Loss value
        return self.loss.calculate(self.output,y_true)

    def backwards(self,dvalues,y_true):
        # number of samples
        samples = len(dvalues)
        # turn one -hot encoded values into descrete values
        if len(y_true.shape)==2:
            y_true = np.argmax(y_true,axis = 1)
        # copy inputs to safely modify the matrix
        self.dinputs = dvalues.copy()
        #calculate gradient
        self.dinputs[range(samples),y_true] -=1
        #normalize gradient
        self.dinputs = self.dinputs/samples

# gradient descent optimizer 
class optimizer_GD:
    def __init__(self,learning_rate=1,decay=0.):
        self.learning_rate = learning_rate
        self.iterations = 0
        self.current_learning_rate = learning_rate
        self.decay = decay

  # learing decay implementation
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ((1.) /(1. + self.decay * self.iterations))

    def update_parameters(self,layer):
        layer.weigths += -self.learning_rate * layer.dweigths
        layer.biases += -self.learning_rate * layer.dbiases

    def post_update_parameters(self):
        self.iterations += 1

if __name__ == '__main__':
    pass

