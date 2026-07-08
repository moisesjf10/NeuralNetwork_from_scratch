import numpy as np

class Sigmoid:
    def __call__(self, x):
        """Compute sigmoid of x"""
        return np.where(x>=0, 1/(1+np.exp(-x)),
                        np.exp(x)/(1+np.exp(x)))

    def derivative(self, x):
        """Compute the derivative of the sigmoid in x"""
        sig=self(x)
        return sig*(1-sig)

class ReLU:
    def __call__(self, x):
        """Computes ReLU of x"""
        return x*(x>0)

    def derivative(self, x):
        """Compute the derivative of ReLU in x"""
        return 1*(x>0)

class Tanh:
    def __call__(self, x):
        """Compute hiperbolic tangent of x"""
        return np.tanh(x)

    def derivative(self, x):
        """Computes the derivative of tanh in x"""
        return 1-self(x)**2

class Softmax:
    def __call__(self, Z):
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)  # Shift for numerical stability
        exp_Z = np.exp(Z_shifted)
        S = np.sum(exp_Z, axis=1, keepdims=True)
        return exp_Z / S
    
    def derivative(self, Z):
        """
        'Dummy' derivative by architectural design.
        
        Mathematically, the Softmax derivative generates a Jacobian matrix. However, 
        when Softmax is combined with Categorical Cross-Entropy (CCE), the derivatives 
        cancel each other out and the gradient with respect to Z simplifies to: 
        (Predictions - True Labels).
        
        In this architecture, the CCE.derivative() function already takes care of 
        calculating and returning this simplified gradient (dZ). Since the preceding 
        layer computes the backward pass by propagating `dZ = dA * activation_derivative`, 
        we return a matrix of ones (1) to act as a multiplicative identity. This allows 
        the exact gradient computed by CCE to pass through unaltered to the hidden layers.
        
        WARNING: This simplification tightly couples Softmax with CCE. 
        If you use this activation combined with MSE or any other loss function, 
        the propagated gradients will be mathematically incorrect.
        """
        return np.ones_like(Z)

class Tanh:
    def __call__(self, x):
        """Compute hyperbolic tangent of X"""
        return np.tanh(x)
    
    def derivative(self, x):
        """Compute the derivative of tanh in X"""
        return 1 - self(x)**2


