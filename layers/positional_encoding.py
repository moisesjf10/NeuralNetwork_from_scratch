import numpy as np

class PositionalEncoding:
    def __init__(self, max_seq_length, embedding_dim):
        """
        max_seq_length: The maximum length of a sequence the model will ever see.
        embedding_dim: Must match the dimension of the upstream EmbeddingLayer.
        """

        self.pe=np.zeros((max_seq_length, embedding_dim))

        # Create a column vector of positions: [[0], [1], [2], ..., [max_seq_length - 1]]
        position=np.arange(0,max_seq_length)[:,np.newaxis]

        div_term=np.exp(np.arange(0, embedding_dim,2)*-(np.log(10000.0)/embedding_dim))
        
        #apply sine to even indices
        self.pe[:,0::2]=np.sin(position*div_term)
        #apply cosine to odd indices
        self.pe[:,1::2]=np.cos(position*div_term)
    
    def forward(self, X):
        """
        Injects the positional signal into the semantic embeddings.
        
        X shape: (Batch_Size, Time_Steps, Embedding_Dim)
        Returns: The same shape, geometrically shifted.
        """
        time_steps=X.shape[1]

        # We slice self.pe up to the current sequence length to match X's shape.
        # NumPy broadcasting automatically adds this 2D matrix to every item in the 3D Batch.
        return X+self.pe[:time_steps,:]

    def backward(self, dA):
        """
        The operation during forward is simply Z = X + Constant.
        The derivative of X + C with respect to X is 1. 
        Therefore, the gradient flows backwards completely unchanged.
        """
        return dA
