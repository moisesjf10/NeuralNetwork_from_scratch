import numpy as np
from activations import ReLU, Sigmoid, Tanh

class Layer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.activation=activation
        self.n_inputs=n_inputs
        self.n_neurons=n_neurons
        
        if isinstance(self.activation, ReLU):
            #He initialization
            sd=np.sqrt(2/n_inputs)
        else:
            #Xavier initialization
            sd=np.sqrt(2/(n_inputs+n_neurons))    
                    
        self.W=np.random.normal(loc=0, scale=sd, size=(n_inputs, n_neurons))


        self.b=np.zeros((1,n_neurons))
    
    def forward(self, X):
        self.X=X #(n_ejemplos, n_inputs) lo guardo para backward
        self.Z= X@self.W+self.b

        return self.activation(self.Z)

    def backward(self,dA):
        # Backpropagate the gradient through the activation function (element wise)
        dZ = dA * self.activation.derivative(self.Z) 
        
        # Compute gradients for the learnable parameters (Weights and Biases)
        # (n_inputs, n_neurons)
        self.dW = self.X.T @ dZ 
        
        # Sum of dZ along the batch dimension to reverse the forward pass broadcasting
        # (1, n_neurons)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        
        # Compute the gradient with respect to the inputs
        # (batch_size, n_inputs)
        dX = dZ @ self.W.T

        return dX


class NeuralNetwork:
    def __init__(self, layers, loss):
        self.layers=layers
        self.loss=loss
    
    def forward(self, X):
        y_pred=X
        for l in self.layers:
            y_pred=l.forward(y_pred)
        
        return y_pred
    
    def backward(self, y_pred, y_true):
        loss_value = self.loss(y_pred, y_true)
        grad = self.loss.derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss_value

    def update(self, lr):
        for l in self.layers:
            l.W=l.W-lr*l.dW
            l.b=l.b-lr*l.db
    
    def train(self, X, y, epochs, lr):
        losses=[]
        for i in range(epochs):
            y_pred=self.forward(X)
            loss_value=self.backward(y_pred, y)
            self.update(lr)
            losses.append(loss_value)
            if(i%100==0):
                print(f"Loss (epoch {i}): {losses[i]}")
        
        return losses
        
