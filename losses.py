import numpy as np

class MSE:
    def __call__(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)
    
    def derivative(self, y_pred, y_true):
        return 2/y_pred.shape[0] * (y_pred - y_true)

class BCE:
    def __call__(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-8, 1-1e-8)
        return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    
    def derivative(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-8, 1-1e-8)
        return -(y_true/y_pred - (1-y_true)/(1-y_pred))