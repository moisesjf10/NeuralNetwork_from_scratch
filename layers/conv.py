import numpy as np

class ConvLayer:
    def __init__(self, n_inputs_channels, n_filters, filter_size, stride=1, padding=0):
        self.C=n_inputs_channels
        self.F=n_filters
        self.f=filter_size
        self.stride=stride
        self.padding=padding

        #Learnable parameters
        #W with He initialization
        n_fan_in=self.C*self.f*self.f
        self.W = np.random.normal(0, np.sqrt(2.0 / n_fan_in), (self.F, self.C, self.f, self.f))

        #b: one bias per filter
        self.b=np.zeros((self.F,1))

        self.dW=np.zeros_like(self.W)
        self.db=np.zeros_like(self.b)

        self.X_cache=None #for backward pass

    
    def forward(self, X):
        self.X_cache=X
        N,C,H,W=X.shape

        #output dimensions
        H_out = (H - self.f + 2 * self.padding) // self.stride + 1
        W_out = (W - self.f + 2 * self.padding) // self.stride + 1

        #apply padding
        if self.padding>0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_pad=X
        
        #im2col
        shape_windows=(N,C,H_out, W_out, self.f, self.f)
        strides_windows = (
            X_pad.strides[0],                     # Move to next image in batch
            X_pad.strides[1],                     # Move to next channel
            X_pad.strides[2] * self.stride,       # Move window down (stride)
            X_pad.strides[3] * self.stride,       # Move window right (stride)
            X_pad.strides[2],                     # Move inside window down
            X_pad.strides[3]                      # Move inside window right
        )

        X_windows=np.lib.stride_tricks.as_strided(X_pad, shape=shape_windows, strides=strides_windows)

        #Generate X_col (flatten the filter spatial dimension intro rows)
        self.X_col=X_windows.transpose(1,4,5,0,2,3).reshape(C*self.f*self.f, -1)

        W_row=self.W.reshape(self.F, -1)

        # W_row (F, C*f*f) @ X_col (C*f*f, N*H_out*W_out) = Z_col (F, N*H_out*W_out)
        Z_col=W_row@self.X_col+self.b

        #reshape Z_col back to 4D 
        # First to (F, N, H_out, W_out), then transpose to standard (N, F, H_out, W_out)
        Z=Z_col.reshape(self.F, N, H_out, W_out).transpose(1,0,2,3)

        return Z

    def backward(self, dZ):
        X=self.X_cache
        N,C,H,W=X.shape

        #output dimensions
        H_out = (H - self.f + 2 * self.padding) // self.stride + 1
        W_out = (W - self.f + 2 * self.padding) // self.stride + 1

        #flatten dZ
        dZ_col=dZ.transpose(1,0,2,3).reshape(self.F, -1)

        #bias gradient. Sum the gradient acrros all axes except the filters
        self.db=np.sum(dZ, axis=(0,2,3)).reshape(self.F, -1)

        #W gradient
        dW_row=dZ_col@self.X_col.T
        self.dW=dW_row.reshape(self.F, self.C, self.f, self.f)

        #flattened input gradient
        W_row=self.W.reshape(self.F,-1)
        dX_col=W_row.T@dZ_col

        #col2im
        # First, initialize a gradient tensor with zeros, including padding
        dX_pad = np.zeros((N, C, H + 2 * self.padding, W + 2 * self.padding))
        
        # Reshape dX_col to separate the filter dimensions from the spatial dimensions
        # From (C*f*f, N*H_out*W_out) to (C, f, f, N, H_out, W_out)
        dX_col_reshaped = dX_col.reshape(self.C, self.f, self.f, N, H_out, W_out)
        
        # Transpose to bring the Batch dimension to the front: (N, C, f, f, H_out, W_out)
        dX_col_reshaped = dX_col_reshaped.transpose(3, 0, 1, 2, 4, 5)
        
        # Iterate only over the filter size (e.g., 3x3 = 9 loops)
        # Use in-place addition (+=) to aggregate overlapping gradients
        for i in range(self.f):
            for j in range(self.f):
                dX_pad[:, :, i:i+self.stride*H_out:self.stride, j:j+self.stride*W_out:self.stride] += dX_col_reshaped[:, :, i, j, :, :]
                
        # Remove padding
        if self.padding > 0:
            dX = dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_pad
            
        return dX

