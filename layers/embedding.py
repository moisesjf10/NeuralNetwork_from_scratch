import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the Embedding Layer.
        
        vocab_size: Total number of unique tokens/characters in the vocabulary.
        embedding_dim: The size of the dense continuous vector space.
        """
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim

        # W is the look-up table 
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01

        self.dW=np.zeros_like(self.W)

        self.inputs=None

    def forward(self, X):
        """
        Forward pass: Discrete index to Continuous vector projection.
        
        X: Matrix of integer token IDs. Shape: (Batch_Size, Time_Steps)
        Returns: 3D Dense Tensor. Shape: (Batch_Size, Time_Steps, Embedding_Dim)
        
        Theoretically, this projection is e = X_one_hot @ W.
        However, multiplying a sparse one-hot vector by a dense matrix 
        is extremely inefficient. Since X_one_hot only has a '1' at index k, the dot 
        product is strictly equal to extracting the k-th row of W. 
        NumPy's Advanced Indexing (self.W[X]) does exactly this in O(1) time without 
        any floating-point multiplications.
        """
        self.inputs=X
        outputs=self.W[X]

        return outputs

    def backward(self, dA):
        """
        Backward pass: Sparse Gradient Accumulation.
        
        dA: Gradient coming from the upstream layer (e.g., RNN).
                   Shape: (Batch_Size, Time_Steps, Embedding_Dim)
        Returns: None
        
        SPARSE GRADIENTS (np.add.at)
        We only need to update the rows of W that were actually used in the forward pass.
        If a word (e.g., "the") appears 3 times in the same sequence, its row must 
        receive the sum of the 3 corresponding upstream gradients.
        Using standard assignment (self.dW[self.inputs] += d_outputs) fails due to 
        memory buffering (it overwrites instead of accumulating). 
        `np.add.at` guarantees an unbuffered, atomic sequential sum.
        
        THE END OF BACKPROPAGATION
        Unlike dense layers that return dX = dZ @ W.T to pass the gradient downwards,
        the Embedding layer is the absolute bottom of the network. Its inputs (X) 
        are discrete integer IDs (categorical data). Since discrete space is 
        non-differentiable (you cannot calculate the slope between the word "cat" 
        and "dog"), the gradient flow terminates here. Thus, we return None.
        """

        self.dW.fill(0)

        np.add.at(self.dW, self.inputs, dA)

        return None
        
