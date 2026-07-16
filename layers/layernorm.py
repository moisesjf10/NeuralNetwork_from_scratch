import numpy as np

class LayerNorm:
    def __init__(self, embedding_dim, epsilon=1e-5):
        """
        embedding_dim (D): The dimension of the feature vectors.
        epsilon: A small constant for numerical stability to prevent division by zero.
        """
        self.D = embedding_dim
        self.eps = epsilon
        
        # Trainable parameters (Gamma for scale, Beta for shift)
        # Gamma starts at 1 (no scaling), Beta starts at 0 (no shift)
        self.gamma = np.ones(self.D)
        self.beta = np.zeros(self.D)
        
        # Gradients for optimization
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        
        self.cache = {}

    def forward(self, X):
        """
        X shape: (Batch_Size, Time_Steps, Embedding_Dim)
        """
        # Compute mean and variance along the last dimension (Embedding_Dim)
        # keepdims=True ensures the shape becomes (B, T, 1) to allow broadcasting
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        
        # Standardize the inputs (Zero mean, Unit variance)
        std = np.sqrt(var + self.eps)
        X_norm = (X - mean) / std
        
        # Scale and Shift using trainable parameters
        # NumPy broadcasts (D,) to (B, T, D) automatically
        out = self.gamma * X_norm + self.beta
        
        # Save variables for the backward pass
        self.cache.update({
            'X_norm': X_norm,
            'std': std
        })
        
        return out

    def backward(self, d_out):
        """
        Calculates gradients for Gamma, Beta, and passes the error down.
        d_out shape: (Batch_Size, Time_Steps, Embedding_Dim)
        """
        X_norm = self.cache['X_norm']
        std = self.cache['std']
        
        # Gradients with respect to trainable parameters (Gamma and Beta)
        # We sum across Batch (axis=0) and Time (axis=1) because Gamma and Beta 
        # are applied identically across all sequences.
        self.dgamma = np.sum(d_out * X_norm, axis=(0, 1))
        self.dbeta = np.sum(d_out, axis=(0, 1))
        
        # Gradient with respect to the input X
        # This is the analytical simplification of the Jacobians for the mean and variance.
        # It ensures that backpropagation doesn't break the zero-mean/unit-variance constraint.
        N = self.D  # Number of features we normalized across
        
        # Derivative of the normalized output scaled by Gamma
        dX_norm = d_out * self.gamma
        
        # Analytical backward pass for Layer Normalization
        dX = (1.0 / (N * std)) * (
            N * dX_norm 
            - np.sum(dX_norm, axis=-1, keepdims=True) 
            - X_norm * np.sum(dX_norm * X_norm, axis=-1, keepdims=True)
        )
        
        return dX