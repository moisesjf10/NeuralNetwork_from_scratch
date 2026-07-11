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
        self.original_shape = X.shape
        if len(X.shape) == 3:
            batch_size, time_steps, features = X.shape
            X = X.reshape(batch_size * time_steps, features)

        self.X=X #(n_ejemplos, n_inputs) lo guardo para backward
        self.Z= X@self.W+self.b

        A = self.activation(self.Z)

        if len(self.original_shape) == 3:
            A = A.reshape(batch_size, time_steps, self.n_neurons)

        return A

    def backward(self,dA):
        batch_size, time_steps = None, None
        if len(dA.shape) == 3:
            batch_size, time_steps, n_neurons = dA.shape
            dA = dA.reshape(batch_size * time_steps, n_neurons)
        
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

        if len(self.original_shape) == 3:
            dX = dX.reshape(batch_size, time_steps, self.n_inputs)

        return dX
