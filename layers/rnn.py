import numpy as np
from activations import Tanh

class RNNLayer:
    def __init__(self, input_size, hidden_size):
        """
        input_size: vocabulary size
        hidden_size: number of neurons
        """
        self.input_size=input_size
        self.hidden_size=hidden_size

        self.activation=Tanh()

        #weights and biases
        # W_hx (Input to Hidden): Xavier/Glorot Normal Initialization
        xavier_std = np.sqrt(2.0 / (input_size + hidden_size))
        self.W_hx = np.random.randn(input_size, hidden_size) * xavier_std

        # W_hh (Hidden to Hidden): Orthogonal Initialization
        random_matrix = np.random.randn(hidden_size, hidden_size)
        Q, R = np.linalg.qr(random_matrix)
        self.W_hh = Q * np.sign(np.diag(R))

        # Biases: Initialized to purely zeros
        self.b = np.zeros((1, hidden_size))

        
        # Matrices to store gradients
        self.dW_hx = np.zeros_like(self.W_hx)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db = np.zeros_like(self.b)

        # Cache for Backpropagation Through Time (BPTT)
        self.inputs = None
        self.hidden_states = None
        self.Z_states = None
    
    def forward(self, X):
        """
        X shape: (Batch, Time, Input_Size)
        Devuelve: (Batch, Time, Hidden_Size)
        """
        self.inputs=X
        batch_size, time_steps, _=X.shape

        #initialize hidden states
        self.hidden_states = np.zeros((batch_size, time_steps + 1, self.hidden_size))

        #pre-activations cache
        self.Z_states=np.zeros((batch_size, time_steps, self.hidden_size))

        outputs=np.zeros((batch_size, time_steps, self.hidden_size))

        #unrolling the network
        for t in range(time_steps):
            x_t=inputs[:,t,:]
            h_prev=self.hidden_states[:,t,:]

            #calculate Z
            Z_t=np.dit(x_t, self.W_hx)+np.dot(h_prev, self.W_hh)+self.b
            
            #save Z in cache
            self.Z_states[:,t,:]=Z_t

            #apply activation
            ht=self.activation(Z_t)ç

            #save the state for the next iteration
            self.hidden_states[:,t+1,:]=h_t
            outputs[:,t,:]=h_t

        return outputs

        