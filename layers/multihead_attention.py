import numpy as np

class MultiHeadAttention:
    def __init__(self, embedding_dim, num_heads):
        """
        embedding_dim (D): Total dimension of the model.
        num_heads (H): Number of parallel attention heads. 
                        (embedding_dim must be strictly divisible by num_heads).
        """
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
            
        self.D = embedding_dim
        self.H = num_heads
        self.d_k = embedding_dim // num_heads
        self.scale = np.sqrt(self.d_k)
        
        # Initialization scalar (Glorot/Xavier)
        sd = np.sqrt(2.0 / (self.D + self.D))
        
        # Q, K, V projections (Single large matrices for efficiency)
        self.W_q = np.random.normal(0, sd, (self.D, self.D))
        self.W_k = np.random.normal(0, sd, (self.D, self.D))
        self.W_v = np.random.normal(0, sd, (self.D, self.D))
        
        # Output projection (Blends the concatenated heads back together)
        self.W_o = np.random.normal(0, sd, (self.D, self.D))
        
        # Gradient accumulators
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)
        
        self.cache = {}

    def _softmax(self, x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def forward(self, X):
        """
        X shape: (B, T, D)
        """
        B, T, D = X.shape
        self.cache['X'] = X
        
        # Linear Projections (B, T, D)
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        
        # Reshape and Transpose for Multi-Head
        # From (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        Q = Q.reshape(B, T, self.H, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.H, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.H, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled Dot-Product Attention
        # Q @ K^T: (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / self.scale

        # Create a boolean upper triangular matrix: True where j > i
        mask = np.triu(np.ones((T, T)), k=1)  # Shape: (T, T)
        # Subtract a huge value from masked positions so exp(-1e9) becomes 0
        # NumPy broadcasting automatically extends (T, T) to (B, H, T, T)
        scores = scores - (mask * 1e9)

        #Softmax weights (Masked values will receive exactly 0% attention)
        A = self._softmax(scores, axis=-1)
        
        # Context Vector calculation
        # A @ V: (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        context = A @ V
        
        # Concatenate heads back together
        # Transpose back: (B, H, T, d_k) -> (B, T, H, d_k)
        # Reshape flattens the last two dims back to D: (B, T, D)
        context_concat = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        
        # Final Output Projection
        Z = context_concat @ self.W_o
        
        # Save state for backward pass
        self.cache.update({'Q': Q, 'K': K, 'V': V, 'A': A, 'scores': scores, 'context_concat': context_concat})
        
        return Z

    def backward(self, dZ):
        """
        Rigorous 4D Backpropagation through the Multi-Head Attention mechanism.
        dZ shape: (B, T, D)
        """
        X = self.cache['X']
        Q, K, V = self.cache['Q'], self.cache['K'], self.cache['V']
        A, context_concat = self.cache['A'], self.cache['context_concat']
        
        B, T, D = X.shape
        
        # Backprop through output projection (Z = context_concat @ W_o)
        # Flatten time and batch for weight gradient accumulation
        dZ_flat = dZ.reshape(B * T, D)
        context_flat = context_concat.reshape(B * T, D)
        
        self.dW_o = context_flat.T @ dZ_flat
        d_context_concat = dZ @ self.W_o.T  # Shape: (B, T, D)
        
        # Reshape gradient to match Multi-Head format (B, H, T, d_k)
        d_context = d_context_concat.reshape(B, T, self.H, self.d_k).transpose(0, 2, 1, 3)
    
        # Backprop through Attention Context (context = A @ V)
        # dV = A^T @ d_context
        dV = A.transpose(0, 1, 3, 2) @ d_context  # (B, H, T, d_k)
        
        # dA = d_context @ V^T
        dA = d_context @ V.transpose(0, 1, 3, 2)  # (B, H, T, T)
        
        # Backprop through Softmax
        d_scores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
        d_scores = d_scores / self.scale
        
        # Backprop through Dot Product (scores = Q @ K^T)
        dQ = d_scores @ K  # (B, H, T, d_k)
        dK = d_scores.transpose(0, 1, 3, 2) @ Q  # (B, H, T, d_k)
        
        # Revert Multi-Head Reshape for Q, K, V gradients -> (B, T, D)
        dQ_concat = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK_concat = dK.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV_concat = dV.transpose(0, 2, 1, 3).reshape(B, T, D)
        
        # Backprop through initial Linear Projections
        X_flat = X.reshape(B * T, D)
        
        self.dW_q = X_flat.T @ dQ_concat.reshape(B * T, D)
        self.dW_k = X_flat.T @ dK_concat.reshape(B * T, D)
        self.dW_v = X_flat.T @ dV_concat.reshape(B * T, D)
        
        # Gradient to pass down to lower layers
        dX = (dQ_concat @ self.W_q.T) + (dK_concat @ self.W_k.T) + (dV_concat @ self.W_v.T)
        
        return dX