# Neural Network from Scratch

Implementation of a fully connected neural network using only NumPy, built to understand the mathematical foundations of deep learning.

## Motivation

The goal of this project is not to build the most efficient neural network, but to understand what happens inside one. Every operation — forward pass, backpropagation, weight update — is implemented from scratch without any deep learning framework.

## Project Structure

```
neural_net/
├── activations.py       # Activation functions and their derivatives
├── losses.py            # Loss functions and their derivatives
├── network.py           # Layer and NeuralNetwork classes
└── experiments/
    └── test_xor.py      # XOR problem as a sanity check
```

## Dependencies

```
numpy
matplotlib
jupyter
```

Install with:
```bash
pip install numpy matplotlib jupyter
```

---

## Mathematical Foundations

### 1. The Forward Pass

A neural network is a composition of functions. Each layer applies a linear transformation followed by a non-linear activation:

$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = f^{[l]}(Z^{[l]})$$

where:
- $A^{[l-1]}$ is the input to layer $l$, with shape $(n_{ejemplos}, n_{inputs})$
- $W^{[l]}$ is the weight matrix of layer $l$, with shape $(n_{inputs}, n_{neurons})$
- $b^{[l]}$ is the bias vector, with shape $(1, n_{neurons})$
- $f^{[l]}$ is the activation function of layer $l$

The output of the last layer is the prediction $\hat{y}$.

---

### 2. Activation Functions

#### Sigmoid
$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

Maps any real number to $(0, 1)$. Used in the output layer for binary classification.

#### ReLU
$$\text{ReLU}(z) = \max(0, z) \qquad \text{ReLU}'(z) = \mathbb{1}[z > 0]$$

Used in hidden layers. Computationally efficient and avoids the vanishing gradient problem.

#### Tanh
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \qquad \tanh'(z) = 1 - \tanh^2(z)$$

Maps to $(-1, 1)$ and is zero-centered, which makes gradient flow better than Sigmoid in hidden layers.

#### Softmax
$$\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

Used in the output layer for multiclass classification. Converts a vector of scores into a probability distribution. The $\max(z)$ subtraction is for numerical stability.

---

### 3. Loss Functions

#### Mean Squared Error (MSE)
$$L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2 \qquad \frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$$

Used for regression problems.

#### Binary Cross-Entropy (BCE)
$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

$$\frac{\partial L}{\partial \hat{y}} = -\left(\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right)$$

Used for binary classification. Always used with Sigmoid in the output layer. Numerical stability requires clipping $\hat{y}$ to $[\epsilon, 1-\epsilon]$.

---

### 4. Backpropagation

Backpropagation is an efficient application of the chain rule to compute the gradient of the loss with respect to every parameter in the network. The gradient flows from the output layer back to the input layer.

For each layer $l$, given $dA^{[l]} = \frac{\partial L}{\partial A^{[l]}}$ from the layer above, we compute:

#### Step 1 — Gradient through the activation

$$dZ^{[l]} = dA^{[l]} \odot f'^{[l]}(Z^{[l]})$$

**Proof:** By the chain rule, element-wise:

$$\frac{\partial L}{\partial Z_{ij}} = \sum_k \frac{\partial L}{\partial A_{ik}} \cdot \frac{\partial A_{ik}}{\partial Z_{ij}}$$

Since $A_{ik} = f(Z_{ik})$, the term $A_{ik}$ only depends on $Z_{ik}$, so the sum collapses to a single term:

$$\frac{\partial L}{\partial Z_{ij}} = \frac{\partial L}{\partial A_{ij}} \cdot f'(Z_{ij})$$

which in matrix form is the elementwise product $dA \odot f'(Z)$.

---

#### Step 2 — Gradient with respect to W

$$dW^{[l]} = (A^{[l-1]})^T \cdot dZ^{[l]}$$

**Proof:** From $Z_{ij} = \sum_k X_{ik} W_{kj} + b_j$, applying the chain rule:

$$\frac{\partial L}{\partial W_{kj}} = \sum_i \frac{\partial L}{\partial Z_{ij}} \cdot \frac{\partial Z_{ij}}{\partial W_{kj}} = \sum_i dZ_{ij} \cdot X_{ik}$$

This sum over examples $i$ is exactly the matrix product $X^T \cdot dZ$ at position $(k, j)$, so:

$$dW = X^T \cdot dZ$$

Shape: $(n_{inputs}, n_{ejemplos}) \times (n_{ejemplos}, n_{neurons}) = (n_{inputs}, n_{neurons})$ ✓

---

#### Step 3 — Gradient with respect to b

$$db^{[l]} = \sum_{i} dZ^{[l]}$$

**Proof:** From $Z_{ij} = \sum_k X_{ik} W_{kj} + b_j$, the bias $b_j$ does not depend on $i$, so:

$$\frac{\partial Z_{ij}}{\partial b_j} = 1$$

Applying the chain rule:

$$\frac{\partial L}{\partial b_j} = \sum_i \frac{\partial L}{\partial Z_{ij}} \cdot 1 = \sum_i dZ_{ij}$$

The sum over examples appears because in the forward pass the same bias was broadcast across all examples, so in the backward pass we accumulate the gradient from all of them.

Shape: sum of $(n_{ejemplos}, n_{neurons})$ over axis 0 → $(1, n_{neurons})$ ✓

---

#### Step 4 — Gradient with respect to the input (passed to the previous layer)

$$dA^{[l-1]} = dZ^{[l]} \cdot (W^{[l]})^T$$

**Proof:** From $Z_{ij} = \sum_k X_{ik} W_{kj} + b_j$, applying the chain rule:

$$\frac{\partial L}{\partial X_{ik}} = \sum_j \frac{\partial L}{\partial Z_{ij}} \cdot \frac{\partial Z_{ij}}{\partial X_{ik}} = \sum_j dZ_{ij} \cdot W_{kj}$$

This sum over neurons $j$ is the matrix product $dZ \cdot W^T$ at position $(i, k)$, so:

$$dX = dZ \cdot W^T$$

Shape: $(n_{ejemplos}, n_{neurons}) \times (n_{neurons}, n_{inputs}) = (n_{ejemplos}, n_{inputs})$ ✓

---

### 5. Weight Update (SGD)

Once the gradients are computed, each parameter is updated in the direction that reduces the loss:

$$W^{[l]} \leftarrow W^{[l]} - \eta \cdot dW^{[l]}$$
$$b^{[l]} \leftarrow b^{[l]} - \eta \cdot db^{[l]}$$

where $\eta$ is the learning rate.

---

### 6. Weight Initialization

Initializing all weights to zero causes all neurons in a layer to compute the same gradient and learn the same features — the symmetry problem. Weights must be initialized randomly, but the scale matters.

**Xavier initialization** — for Sigmoid and Tanh:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{inputs} + n_{neurons}}}\right)$$

**He initialization** — for ReLU:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{inputs}}}\right)$$

He uses a larger variance to compensate for the fact that ReLU kills all negative values, effectively halving the signal variance.

---

## Validation

The implementation is validated in two ways:

**XOR problem:** A 2→2→1 network trained on the 4 XOR examples. Since XOR is not linearly separable, solving it correctly confirms that the forward pass, backpropagation, and weight updates all work together.

**Gradient checking:** For each parameter $\theta$, the numerical gradient is computed as:

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$

and compared against the analytical gradient from backpropagation. If they match to within $10^{-5}$, the implementation is correct.

---

## Roadmap

- [x] Activation functions (Sigmoid, ReLU, Tanh, Softmax)
- [x] Loss functions (MSE, BCE)
- [x] Layer class with forward and backward
- [ ] NeuralNetwork class
- [ ] XOR experiment
- [ ] Gradient checking
- [ ] MNIST experiment
- [ ] Advanced optimizers (Momentum, Adam)
- [ ] Regularization (L2, Dropout)
