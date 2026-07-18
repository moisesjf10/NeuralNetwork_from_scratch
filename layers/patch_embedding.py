import numpy as np

class PatchEmbedding:
    def __init__(self, in_channels, patch_size, embedding_dim):
        """
        Splits images into patches and linearly projects them to embedding_dim.
        """
        self.P = patch_size
        self.C = in_channels
        self.D = embedding_dim
        
        # Flattened patch dimension
        self.patch_dim = self.C * self.P * self.P
        
        # Linear projection weights (Xavier/He initialization)
        sd = np.sqrt(2.0 / self.patch_dim)
        self.W = np.random.normal(0, sd, (self.patch_dim, self.D))
        self.b = np.zeros(self.D)
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        self.cache = {}

    def forward(self, X):
        """
        X shape: (Batch, Channels, Height, Width)
        Returns: (Batch, Num_Patches, embedding_dim) -> (B, T, D)
        """
        B, C, H, W = X.shape
        self.cache['X_shape'] = X.shape
        
        # Calculate number of patches
        num_patches_h = H // self.P
        num_patches_w = W // self.P
        T = num_patches_h * num_patches_w
        
        # Slice image into patches 
        # From (B, C, H, W) -> (B, C, H/P, P, W/P, P)
        patches = X.reshape(B, C, num_patches_h, self.P, num_patches_w, self.P)
        
        # Transpose to group the spatial patches together
        # -> (B, H/P, W/P, C, P, P)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        
        # Flatten each patch -> (B, T, C*P*P)
        X_flat = patches.reshape(B, T, self.patch_dim)
        self.cache['X_flat'] = X_flat
        
        # Linear Projection to Embedding Dimension -> (B, T, D)
        Z = X_flat @ self.W + self.b
        
        return Z

    def backward(self, dZ):
        """dZ shape: (B, T, D)"""
        X_flat = self.cache['X_flat']
        B, C, H, W = self.cache['X_shape']
        
        num_patches_h = H // self.P
        num_patches_w = W // self.P
        T = num_patches_h * num_patches_w
        
        # Reshape for matrix multiplication
        dZ_flat = dZ.reshape(B * T, self.D)
        X_flat_2d = X_flat.reshape(B * T, self.patch_dim)
        
        # Parameter gradients
        self.dW = X_flat_2d.T @ dZ_flat
        self.db = np.sum(dZ_flat, axis=0)
        
        # Gradient routing back to the input patches
        dX_flat = dZ @ self.W.T  # (B, T, C*P*P)
        
        # Reverse the reshape magic to reconstruct the image gradient
        dX_patches = dX_flat.reshape(B, num_patches_h, num_patches_w, C, self.P, self.P)
        dX = dX_patches.transpose(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        return dX