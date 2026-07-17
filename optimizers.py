import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.t = 0
        self.moments = {}        

    def clip(self, grad):
        """prevent exploding gradients by clipping the gradient values to a specified range."""
        if grad is None: return None
        return np.clip(grad, -self.clip_value, self.clip_value)

    def step(self, layers):
        self.t += 1

        params_to_update = [
            ('W', 'dW'), ('b', 'db'), 
            ('W_q', 'dW_q'), ('W_k', 'dW_k'), ('W_v', 'dW_v'), ('W_o', 'dW_o'),
            ('gamma', 'dgamma'), ('beta', 'dbeta'),
            ('W_hx', 'dW_hx'), ('W_hh', 'dW_hh')
        ]

        for layer in layers:
            if id(layer) not in self.moments:
                self.moments[id(layer)] = {}

            for param_name, grad_name in params_to_update:
                # if the layer has the parameter and its corresponding gradient, we proceed to update
                if hasattr(layer, param_name) and hasattr(layer, grad_name):
                    param = getattr(layer, param_name)
                    grad = getattr(layer, grad_name)
                    
                    if param is not None and grad is not None:
                        # initialize moments if they don't exist for this parameter
                        if param_name not in self.moments[id(layer)]:
                            self.moments[id(layer)][param_name] = {
                                'm': np.zeros_like(param),
                                'v': np.zeros_like(param)
                            }
                        
                        m = self.moments[id(layer)][param_name]['m']
                        v = self.moments[id(layer)][param_name]['v']
                        
                        # clip the gradient to prevent exploding gradients
                        g = self.clip(grad)
                        
                        # 1. Update moments
                        m = self.beta1 * m + (1 - self.beta1) * g
                        v = self.beta2 * v + (1 - self.beta2) * (g ** 2)
                        
                        # 2. Bias correction
                        m_hat = m / (1 - self.beta1 ** self.t)
                        v_hat = v / (1 - self.beta2 ** self.t)
                        
                        # 3. Update weights dinámicamente usando setattr
                        new_param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                        setattr(layer, param_name, new_param)
                        
                        # 4. Save updated moments back to the dictionary
                        self.moments[id(layer)][param_name]['m'] = m
                        self.moments[id(layer)][param_name]['v'] = v