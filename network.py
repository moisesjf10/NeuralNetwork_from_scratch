import numpy as np
from activations import ReLU, Sigmoid, Tanh
from layers import Layer
import pickle


class NeuralNetwork:
    def __init__(self, layers, loss):
        """
        Initializes the network with a sequence of layers and a loss function.
        """
        self.layers = layers
        self.loss = loss
    
    def forward(self, X):
        """
        Propagates the input forward through all layers sequentially.
        """
        y_pred = X
        for l in self.layers:
            y_pred = l.forward(y_pred)
        return y_pred
    
    def backward(self, y_pred, y_true):
        """
        Computes the loss derivative and propagates the gradients backward
        through all layers in reverse order. Returns the scalar loss value.
        """
        loss_value = self.loss(y_pred, y_true)
        grad = self.loss.derivative(y_pred, y_true)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        return loss_value

    def update(self, lr):
        """
        Applies Vanilla Stochastic Gradient Descent (SGD) to all learnable
        parameters across different layer architectures safely.
        """
        for l in self.layers:
            # Standard weights (Dense, Conv, Embedding Layer)
            if hasattr(l, 'W') and hasattr(l, 'dW') and l.W is not None:
                l.W -= lr * l.dW
                
            # Standard biases (Dense, Conv)
            if hasattr(l, 'b') and hasattr(l, 'db') and l.b is not None:
                l.b -= lr * l.db
                
            # RNN specific hidden-to-input weights
            if hasattr(l, 'W_hx') and hasattr(l, 'dW_hx'):
                l.W_hx -= lr * l.dW_hx
                
            # RNN specific hidden-to-hidden weights
            if hasattr(l, 'W_hh') and hasattr(l, 'dW_hh'):
                l.W_hh -= lr * l.dW_hh
        
    def train(self, X, y, epochs, lr, batch_size=32, optimizer=None):
        """
        Main training optimization loop using mini-batching.
        """
        losses_epoch = []
        for epoch in range(epochs):
            loss_epoch = 0
            # Shuffle and partition data into batches
            X_batches, y_batches = self.create_batches(X, y, batch_size=batch_size)

            for X_batch, y_batch in zip(X_batches, y_batches):
                self.zero_grad()
                y_pred = self.forward(X_batch)
                loss_epoch += self.backward(y_pred, y_batch)
                
                if optimizer is not None:
                    optimizer.step(self.layers)
                else: # Fallback to default SGD
                    self.update(lr)
            
            epoch_avg_loss = loss_epoch / len(X_batches)
            losses_epoch.append(epoch_avg_loss)
            print(f"Loss (epoch {epoch}): {epoch_avg_loss:.4f}")
            
        return losses_epoch
    
    def create_batches(self, X, y, batch_size):
        """
        Shuffles the dataset and shards it into discrete mini-batches.
        """
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        X_batches = [X_shuffled[i:i + batch_size] for i in range(0, X.shape[0], batch_size)]
        y_batches = [y_shuffled[i:i + batch_size] for i in range(0, y.shape[0], batch_size)]

        return X_batches, y_batches
    
    def zero_grad(self):
        """
        Resets all parameter gradient accumulations to zero before every batch pass.
        Using .fill(0) updates the array elements in place, maximizing memory efficiency.
        """
        for layer in self.layers:
            if hasattr(layer, 'dW') and layer.dW is not None:
                layer.dW.fill(0)
            if hasattr(layer, 'db') and layer.db is not None:
                layer.db.fill(0)
            if hasattr(layer, 'dW_hx') and layer.dW_hx is not None:
                layer.dW_hx.fill(0)
            if hasattr(layer, 'dW_hh') and layer.dW_hh is not None:
                layer.dW_hh.fill(0)
    
    def save_weights(self, filepath):
        """
        Dynamically extracts and serializes any active learnable weights present 
        in each layer to a binary file via pickle.
        """
        model_state = []
        for layer in self.layers:
            layer_state = {}
            
            # Extract standard parameters if they exist
            if hasattr(layer, 'W') and layer.W is not None:
                layer_state['W'] = layer.W
            if hasattr(layer, 'b') and layer.b is not None:
                layer_state['b'] = layer.b
                
            # Extract recurrent parameters if they exist
            if hasattr(layer, 'W_hx') and layer.W_hx is not None:
                layer_state['W_hx'] = layer.W_hx
            if hasattr(layer, 'W_hh') and layer.W_hh is not None:
                layer_state['W_hh'] = layer.W_hh
            
            model_state.append(layer_state if layer_state else None)
                
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Weights successfully saved to: {filepath}")

    def load_weights(self, filepath):
        """
        Loads weight states and re-assigns parameter matrices safely based on 
        saved dictionary keys.
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
            
        for i, layer in enumerate(self.layers):
            state = model_state[i]
            if state is not None:
                if 'W' in state: layer.W = state['W']
                if 'b' in state: layer.b = state['b']
                if 'W_hx' in state: layer.W_hx = state['W_hx']
                if 'W_hh' in state: layer.W_hh = state['W_hh']
                
        print(f"Weights successfully loaded from: {filepath}")
        
