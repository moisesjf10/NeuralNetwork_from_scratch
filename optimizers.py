import numpy as np

class Adam:
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.t=0
        self.moments={}        

    def step(self, layers):
        self.t+=1

        for layer in layers:
            if hasattr(layer, 'dW'):
                if id(layer) not in self.moments:
                    mW=np.zeros_like(layer.W)
                    vW=np.zeros_like(layer.W)
                    mb=np.zeros_like(layer.b)
                    vb=np.zeros_like(layer.b)
                    self.moments[id(layer)] = {'mW': mW, 'vW': vW, 'mb': mb, 'vb': vb}   

                mW = self.moments[id(layer)]['mW']
                vW = self.moments[id(layer)]['vW']
                mb = self.moments[id(layer)]['mb']
                vb = self.moments[id(layer)]['vb']

                #update moments
                mW=self.beta1*mW+(1-self.beta1)*layer.dW
                vW=self.beta2*vW+(1-self.beta2)*layer.dW**2
                mb=self.beta1*mb+(1-self.beta1)*layer.db
                vb=self.beta2*vb+(1-self.beta2)*layer.db**2
                # Bias correction
                mW_hat = mW / (1 - self.beta1 ** self.t)
                vW_hat = vW / (1 - self.beta2 ** self.t)
                mb_hat = mb / (1 - self.beta1 ** self.t)
                vb_hat = vb / (1 - self.beta2 ** self.t)

                # Update weights
                layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
                layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)

                # Save updated moments back to the dictionary
                self.moments[id(layer)]['mW'] = mW
                self.moments[id(layer)]['vW'] = vW
                self.moments[id(layer)]['mb'] = mb
                self.moments[id(layer)]['vb'] = vb


