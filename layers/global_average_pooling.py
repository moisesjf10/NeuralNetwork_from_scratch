import numpy as np

class GlobalAveragePooling1D:
    def __init__(self):
        self.cache = {}

    def forward(self, X):
        """ X shape: (B, T, D) """
        self.cache['T'] = X.shape[1]
        # Average across the time/patch dimension (axis=1) -> Returns (B, D)
        return np.mean(X, axis=1)

    def backward(self, dZ):
        """ dZ shape: (B, D) """
        T = self.cache['T']
        # Distribute the gradient equally to all patches
        # From (B, D) -> (B, 1, D) -> broadcast to (B, T, D)
        dX = np.expand_dims(dZ, axis=1) / T
        return np.repeat(dX, T, axis=1)