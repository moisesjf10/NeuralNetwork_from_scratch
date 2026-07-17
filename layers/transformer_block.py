from activations import ReLU, Linear
from layers import Layer, LayerNorm, MultiHeadAttention

class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        """
        Position-wise Feed-Forward Network built using the standard Layer class.
        """
        # First layer expands dimensionality and applies ReLU
        self.layer1 = Layer(n_inputs=d_model, n_neurons=d_ff, activation=ReLU())
        
        # Second layer compresses back to d_model with NO activation (Linear)
        self.layer2 = Layer(n_inputs=d_ff, n_neurons=d_model, activation=Linear())
        
        # Store for the network optimizer
        self.sub_layers = [self.layer1, self.layer2]

    def forward(self, X):
        """
        X shape: (B, T, d_model)
        """
        out1 = self.layer1.forward(X)
        out2 = self.layer2.forward(out1)
        return out2

    def backward(self, dZ):
        """
        dZ shape: (B, T, d_model)
        """
        d_out1 = self.layer2.backward(dZ)
        dX = self.layer1.backward(d_out1)
        return dX

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        """
        A complete Pre-LayerNorm Transformer Block.
        """
        self.norm1 = LayerNorm(embedding_dim=d_model)
        self.attention = MultiHeadAttention(embedding_dim=d_model, num_heads=num_heads)
        
        self.norm2 = LayerNorm(embedding_dim=d_model)
        self.ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        
        # We store references to sub-layers in a list so the NeuralNetwork 
        # optimizer can dynamically find and update their weights via hasattr()
        self.sub_layers = [self.norm1, self.attention, self.norm2, self.ff]

    def forward(self, X):
        """ X shape: (B, T, D) """
        
        # Pre-LN Attention Sublayer + Residual Connection
        # The input bypasses the attention via the residual connection (+)
        norm_X1 = self.norm1.forward(X)
        attn_out = self.attention.forward(norm_X1)
        Z1 = X + attn_out
        
        # Pre-LN Feed-Forward Sublayer + Residual Connection
        norm_X2 = self.norm2.forward(Z1)
        ff_out = self.ff.forward(norm_X2)
        Z2 = Z1 + ff_out
        
        return Z2

    def backward(self, dZ2):
        """
        Backward pass through the residual connections.
        When a forward pass splits via a residual addition (out = X + Sublayer(X)), 
        the gradient copies and flows backwards identically to both branches.
        """
        
        # --- Backprop through Feed-Forward Sublayer ---
        # Branch 1 (Residual): dZ2 flows directly back to the Z1 stream
        # Branch 2 (Sublayer): dZ2 flows through FFN and Norm2
        d_ff = self.ff.backward(dZ2)
        d_norm2 = self.norm2.backward(d_ff)
        
        # The gradients from both branches sum together
        dZ1 = dZ2 + d_norm2
        
        # --- Backprop through Attention Sublayer ---
        # Branch 1 (Residual): dZ1 flows directly back to the X stream
        # Branch 2 (Sublayer): dZ1 flows through Attention and Norm1
        d_attn = self.attention.backward(dZ1)
        d_norm1 = self.norm1.backward(d_attn)
        
        # Final gradient passed down to the previous TransformerBlock or PositionalEncoding
        dX = dZ1 + d_norm1
        
        return dX