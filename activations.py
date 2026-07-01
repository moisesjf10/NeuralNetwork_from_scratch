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
