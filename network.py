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
        
    def _get_all_parameters_layers(self):
        """
        Recursively extracts all layers and sub-layers from composite blocks 
        (like TransformerBlock or PositionwiseFeedForward) to ensure the 
        optimizer finds every learnable parameter.
        """
        all_layers = []
        def extract_layers(layer_list):
            for layer in layer_list:
                if hasattr(layer, 'sub_layers'):
                    extract_layers(layer.sub_layers)
                else:
                    all_layers.append(layer)
                    
        extract_layers(self.layers)
        return all_layers
    
    def forward(self, X):
        """
        Propagates the input forward through all base layers sequentially.
        """
        y_pred = X
        for l in self.layers:
            y_pred = l.forward(y_pred)
        return y_pred
    
    def backward(self, y_pred, y_true):
        """
        Computes the loss derivative and propagates the gradients backward
        through all base layers in reverse order.
        """
        loss_value = self.loss(y_pred, y_true)
        grad = self.loss.derivative(y_pred, y_true)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        return loss_value

    def update(self, lr, clip_value=1.0):
        """
        Applies SGD with Gradient Clipping to prevent exploding gradients.
        """
        # Función auxiliar para aplicar el límite
        def clip(grad):
            if grad is None: return None
            return np.clip(grad, -clip_value, clip_value)

        for l in self._get_all_parameters_layers():
            
            # Standard weights & biases
            if hasattr(l, 'W') and hasattr(l, 'dW') and l.W is not None:
                l.W -= lr * clip(l.dW)
            if hasattr(l, 'b') and hasattr(l, 'db') and l.b is not None:
                l.b -= lr * clip(l.db)
                
            # Attention weights
            for suffix in ['q', 'k', 'v', 'o']:
                W_attr = f'W_{suffix}'
                dW_attr = f'dW_{suffix}'
                if hasattr(l, W_attr) and hasattr(l, dW_attr) and getattr(l, W_attr) is not None:
                    setattr(l, W_attr, getattr(l, W_attr) - lr * clip(getattr(l, dW_attr)))
                
            # RNN specific weights
            if hasattr(l, 'W_hx') and hasattr(l, 'dW_hx'):
                l.W_hx -= lr * clip(l.dW_hx)
            if hasattr(l, 'W_hh') and hasattr(l, 'dW_hh'):
                l.W_hh -= lr * clip(l.dW_hh)
            
            # LayerNorm parameters
            if hasattr(l, 'gamma') and hasattr(l, 'dgamma') and l.gamma is not None:
                l.gamma -= lr * clip(l.dgamma)
            if hasattr(l, 'beta') and hasattr(l, 'dbeta') and l.beta is not None:
                l.beta -= lr * clip(l.dbeta)
        
    def train(self, X, y, epochs, lr, batch_size=32, optimizer=None):
        """Main training optimization loop using mini-batching."""
        losses_epoch = []
        for epoch in range(epochs):
            loss_epoch = 0
            X_batches, y_batches = self.create_batches(X, y, batch_size=batch_size)

            for X_batch, y_batch in zip(X_batches, y_batches):
                self.zero_grad()
                y_pred = self.forward(X_batch)
                loss_epoch += self.backward(y_pred, y_batch)
                
                if optimizer is not None:
                    optimizer.step(self._get_all_parameters_layers())
                else: 
                    self.update(lr)
            
            epoch_avg_loss = loss_epoch / len(X_batches)
            losses_epoch.append(epoch_avg_loss)
            print(f"Loss (epoch {epoch}): {epoch_avg_loss:.4f}")
            
        return losses_epoch
    
    def create_batches(self, X, y, batch_size):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        X_batches = [X_shuffled[i:i + batch_size] for i in range(0, X.shape[0], batch_size)]
        y_batches = [y_shuffled[i:i + batch_size] for i in range(0, y.shape[0], batch_size)]

        return X_batches, y_batches
    
    def zero_grad(self):
        """Resets all parameter gradient accumulations to zero."""
        for layer in self._get_all_parameters_layers():
            attributes_to_zero = ['dW', 'db', 'dW_hx', 'dW_hh', 'dgamma', 'dbeta', 
                                  'dW_q', 'dW_k', 'dW_v', 'dW_o']
            for attr in attributes_to_zero:
                if hasattr(layer, attr) and getattr(layer, attr) is not None:
                    getattr(layer, attr).fill(0)
    
    def save_weights(self, filepath):
        """Serializes all active learnable weights to a binary file via pickle."""
        model_state = []
        for layer in self._get_all_parameters_layers():
            layer_state = {}
            
            attributes_to_save = ['W', 'b', 'W_hx', 'W_hh', 'gamma', 'beta', 
                                  'W_q', 'W_k', 'W_v', 'W_o']
            
            for attr in attributes_to_save:
                if hasattr(layer, attr) and getattr(layer, attr) is not None:
                    layer_state[attr] = getattr(layer, attr)
            
            model_state.append(layer_state if layer_state else None)
                
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Weights successfully saved to: {filepath}")

    def load_weights(self, filepath):
        """Loads weight states and re-assigns parameter matrices."""
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
            
        flat_layers = self._get_all_parameters_layers()
        
        for i, layer in enumerate(flat_layers):
            state = model_state[i]
            if state is not None:
                for key, val in state.items():
                    if hasattr(layer, key):
                        setattr(layer, key, val)
                        
        print(f"Weights successfully loaded from: {filepath}")

    
