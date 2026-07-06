import numpy as np
from activations import ReLU, Sigmoid, Tanh
from layers import Layer
import pickle
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
            if hasattr(l, 'dW'): #flattenLayer and poolingLayer haven't dW and db attribute
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
            #if(epoch%100==0):
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
    
    def save_weights(self, filepath):
        model_state = []
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                model_state.append({'W': layer.W, 'b': layer.b})
            else:
                model_state.append(None)
                
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Weights saved to: {filepath}")

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
            
        for i, layer in enumerate(self.layers):
            if model_state[i] is not None:
                layer.W = model_state[i]['W']
                layer.b = model_state[i]['b']
        print(f"Weights loaded successfully from: {filepath}")
        
