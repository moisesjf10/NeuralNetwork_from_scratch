import numpy as np

class FlattenLayer:
    def __init__(self):
        self.input_shape=None
    
    def forward(self, X):
        self.input_shape=X.shape
        return np.reshape(X, (X.shape[0], -1))

    def backward(self, dA):
        return np.reshape(dA, self.input_shape)
