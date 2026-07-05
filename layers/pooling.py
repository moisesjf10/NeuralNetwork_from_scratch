import numpy as np

class PoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size=pool_size
        self.stride=stride
        self.mask=None
    
    def forward(self, X):
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got shape {X.shape}")
        self.input_shape=X.shape
        N, C, H, W=X.shape
        if H % self.pool_size != 0 or W % self.pool_size != 0:
            raise ValueError(
                f"Spatial dimensions ({H}, {W}) must be divisible by pool_size ({self.pool_size})"
            )

        H_out=H//self.pool_size
        W_out=W//self.pool_size

        X_reshaped=np.reshape(X, (N,C,H_out, self.pool_size, W_out, self.pool_size))

        #the windows are in axis 3 and 5
        out=X_reshaped.max(axis=(3,5))

        #expand out
        out_expanded=np.repeat(np.repeat(out, self.pool_size, axis=2), self.pool_size, axis=3)
        #create the mask
        self.mask=(X==out_expanded).astype(float)

        return out
    
    def backward(self, dA):
        dA_expanded=np.repeat(np.repeat(dA, self.pool_size, axis=2), self.pool_size, axis=3)
        dX=dA_expanded*self.mask

        return dX
