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
        return: (Batch, Time, Hidden_Size)
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
            x_t=self.inputs[:,t,:]
            h_prev=self.hidden_states[:,t,:]

            #calculate Z
            Z_t=np.dot(x_t, self.W_hx)+np.dot(h_prev, self.W_hh)+self.b
            
            #save Z in cache
            self.Z_states[:,t,:]=Z_t

            #apply activation
            h_t=self.activation(Z_t)

            #save the state for the next iteration
            self.hidden_states[:,t+1,:]=h_t
            outputs[:,t,:]=h_t

        return outputs

    def backward(self, dA):
        """
        dA shape: (Batch, Time, Hidden_Size)
        Gradient of the loss with respect to the RNN outputs. 
        Returns: Shape (Batch_Size, Time_Steps, Input_Size)
        """
        batch_size, time_steps, _ = dA.shape

        dX=np.zeros_like(self.inputs)

        #dh_next represents the gradient flowing backward from the future
        dh_next = np.zeros((batch_size, self.hidden_size))

        #reset weight and bias gradients
        self.dW_hx.fill(0)
        self.dW_hh.fill(0)
        self.db.fill(0)

        #Backpropagation Through Time
        for t in reversed(range(time_steps)):
            # The total gradient at time t is the sum of the upstream gradient (from the layer above)
            # AND the gradient passed back from the future (t+1)
            dh_t = dA[:, t, :] + dh_next
            
            # Fetch the pre-activation values (Z) for this time step and compute the local derivative
            Z_t = self.Z_states[:, t, :]
            dtanh = dh_t * self.activation.derivative(Z_t)

          
            # Jacobians: Accumulate gradients for weights and biases
            # Gradient w.r.t input weights (W_hx)
            x_t = self.inputs[:, t, :]
            self.dW_hx += np.dot(x_t.T, dtanh)
            
            # Gradient w.r.t hidden memory weights (W_hh)
            h_prev = self.hidden_states[:, t, :]
            self.dW_hh += np.dot(h_prev.T, dtanh)
            
            # Gradient w.r.t biases (summed across the batch dimension)
            self.db += np.sum(dtanh, axis=0, keepdims=True)

            # Gradients to propagate further backwards
            # Gradient to pass down to the lower layer
            dX[:, t, :] = np.dot(dtanh, self.W_hx.T)
            
            # Gradient to pass back in time to the previous step (t-1)
            dh_next = np.dot(dtanh, self.W_hh.T)
        
        # Gradient Clipping
        for dparam in [self.dW_hx, self.dW_hh, self.db]:
            np.clip(dparam, -5.0, 5.0, out=dparam)

        return dX