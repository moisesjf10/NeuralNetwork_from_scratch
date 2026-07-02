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
    
    def train(self, X, y, epochs, lr, batch_size=32):
        losses_epoch=[]
        for epoch in range(epochs):
            loss_epoch=0
            #shuffle and create batches
            X_batches, y_batches = self.create_batches(X, y, batch_size=batch_size)

            for X_batch, y_batch in zip(X_batches, y_batches):
                y_pred=self.forward(X_batch)
                loss_epoch+=self.backward(y_pred, y_batch)
                self.update(lr)
            
            losses_epoch.append(loss_epoch/len(X_batches))
            if(epoch%100==0):
                print(f"Loss (epoch {epoch}): {losses_epoch[epoch]}")
        return losses_epoch
    
    def create_batches(self, X, y, batch_size):
        # Shuffle the data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Create batches
        X_batches = [X_shuffled[i:i + batch_size] for i in range(0, X.shape[0], batch_size)]
        y_batches = [y_shuffled[i:i + batch_size] for i in range(0, y.shape[0], batch_size)]

        return X_batches, y_batches
        
