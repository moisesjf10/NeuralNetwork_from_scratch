# Neural Network from Scratch

Implementation of a fully connected neural network and a convolutional neural network using only NumPy, built to understand the mathematical foundations of deep learning.

## Motivation

The goal of this project is not to build the most efficient neural network, but to understand what happens inside one. Every operation — forward pass, backpropagation, weight update — is implemented from scratch without any deep learning framework.

## Project Structure

```
neural_net/
├── activations.py          # Activation functions and their derivatives
├── losses.py               # Loss functions and their derivatives
├── network.py              # NeuralNetwork class
├── layers/
│   ├── __init__.py         # Package exports
│   ├── dense.py            # Fully connected layer
│   ├── flatten.py          # FlattenLayer
│   ├── pooling.py          # PoolingLayer (Max Pooling)
│   └── conv.py             # ConvLayer (im2col)
└── experiments/
    ├── test_xor.py         # XOR sanity check
    ├── tests.py            # Full test suite with gradient checking
    ├── train_mnist.py      # Dense network on MNIST
    └── train_mnist_cnn.py  # CNN on MNIST
```

## Dependencies

```
numpy
matplotlib
jupyter
scikit-learn
```

Install with:
```bash
pip install numpy matplotlib jupyter scikit-learn
```

---

## Results

| Architecture | Dataset | Train Accuracy | Test Accuracy |
|---|---|---|---|
| Dense 784→128→64→10 | MNIST | 99.98% | 97.61% |
| CNN (in progress) | MNIST | — | — |

---

## Mathematical Foundations

### 1. The Forward Pass

A neural network is a composition of functions. Each layer applies a linear transformation followed by a non-linear activation:

$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = f^{[l]}(Z^{[l]})$$

where:
- $A^{[l-1]}$ is the input to layer $l$, with shape $(n_{examples}, n_{inputs})$
- $W^{[l]}$ is the weight matrix of layer $l$, with shape $(n_{inputs}, n_{neurons})$
- $b^{[l]}$ is the bias vector, with shape $(1, n_{neurons})$
- $f^{[l]}$ is the activation function of layer $l$

The output of the last layer is the prediction $\hat{y}$.

---

### 2. Activation Functions

All activation functions are implemented as classes with `__call__` and `derivative` methods so that each activation carries its own derivative, eliminating the need for external dictionaries or conditionals.

#### Sigmoid
$$\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

Maps any real number to $(0, 1)$. Used in the output layer for binary classification. Numerically stabilized by treating positive and negative inputs separately to avoid overflow in `exp`.

#### ReLU
$$\text{ReLU}(z) = \max(0, z) \qquad \text{ReLU}'(z) = \mathbb{1}[z > 0]$$

Used in hidden layers. Computationally efficient and avoids the vanishing gradient problem. In NumPy implemented as `x * (x > 0)`, exploiting boolean casting.

#### Tanh
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \qquad \tanh'(z) = 1 - \tanh^2(z)$$

Maps to $(-1, 1)$ and is zero-centered, which makes gradient flow better than Sigmoid in hidden layers. Uses `np.tanh` for numerical stability.

#### Softmax
$$\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

Used in the output layer for multiclass classification. Converts a vector of scores into a probability distribution. The $\max(z)$ subtraction prevents overflow — it is mathematically equivalent since the factor $e^{-\max(z)}$ cancels in numerator and denominator.

Softmax is never used alone in the backward pass. Its derivative is a Jacobian matrix which complicates backpropagation. Instead it is always combined with Categorical Cross-Entropy, whose combined derivative simplifies to $\hat{y} - y$. For this reason `Softmax.derivative` returns ones, leaving the gradient from the loss function unchanged.

---

### 3. Loss Functions

All loss functions are implemented as classes with `__call__` (returns a scalar for monitoring) and `derivative` (returns an array of the same shape as `y_pred` for backpropagation).

#### Mean Squared Error (MSE)
$$L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2 \qquad \frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$$

Used for regression problems.

#### Binary Cross-Entropy (BCE)
$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

$$\frac{\partial L}{\partial \hat{y}} = -\left(\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right)$$

Used for binary classification. Always used with Sigmoid in the output layer. Numerical stability requires clipping $\hat{y}$ to $[\epsilon, 1-\epsilon]$ with $\epsilon = 10^{-8}$ to avoid $\log(0)$.

#### Categorical Cross-Entropy (CCE)
$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

Used for multiclass classification with Softmax. Requires labels in one-hot format: a vector of zeros with a 1 at the position of the correct class.

The combined derivative of Softmax + CCE simplifies to:

$$\frac{\partial L}{\partial z} = \frac{1}{N}(\hat{y} - y)$$

**Proof of the combined Softmax + CCE derivative:**

Define $S = \sum_j e^{z_j}$ and $\hat{y}_k = e^{z_k}/S$. Starting from $\frac{\partial L}{\partial \hat{y}_k} = -y_k/\hat{y}_k$ and the Softmax Jacobian:

$$\frac{\partial \hat{y}_k}{\partial z_i} = \begin{cases} \hat{y}_i(1 - \hat{y}_i) & k = i \\ -\hat{y}_k \hat{y}_i & k \neq i \end{cases}$$

Applying the chain rule and summing over all classes $k$:

$$\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial \hat{y}_k} \cdot \frac{\partial \hat{y}_k}{\partial z_i} = -y_i(1-\hat{y}_i) + \hat{y}_i\sum_{k \neq i} y_k$$

Rearranging:

$$= -y_i + \hat{y}_i\underbrace{\sum_k y_k}_{=1} = \hat{y}_i - y_i$$

The sum collapses to 1 because $y$ is one-hot. Divided by $N$ for the batch average:

$$\frac{\partial L}{\partial z} = \frac{1}{N}(\hat{y} - y)$$

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

Shape: $(n_{inputs}, n_{examples}) \times (n_{examples}, n_{neurons}) = (n_{inputs}, n_{neurons})$ ✓

---

#### Step 3 — Gradient with respect to b

$$db^{[l]} = \sum_{i} dZ^{[l]}$$

**Proof:** From $Z_{ij} = \sum_k X_{ik} W_{kj} + b_j$, the bias $b_j$ does not depend on $i$, so:

$$\frac{\partial Z_{ij}}{\partial b_j} = 1$$

Applying the chain rule:

$$\frac{\partial L}{\partial b_j} = \sum_i \frac{\partial L}{\partial Z_{ij}} \cdot 1 = \sum_i dZ_{ij}$$

The sum over examples appears because in the forward pass the same bias was broadcast across all examples, so in the backward pass we accumulate the gradient from all of them.

Shape: sum of $(n_{examples}, n_{neurons})$ over axis 0 → $(1, n_{neurons})$ ✓

---

#### Step 4 — Gradient with respect to the input (passed to the previous layer)

$$dA^{[l-1]} = dZ^{[l]} \cdot (W^{[l]})^T$$

**Proof:** From $Z_{ij} = \sum_k X_{ik} W_{kj} + b_j$, applying the chain rule:

$$\frac{\partial L}{\partial X_{ik}} = \sum_j \frac{\partial L}{\partial Z_{ij}} \cdot \frac{\partial Z_{ij}}{\partial X_{ik}} = \sum_j dZ_{ij} \cdot W_{kj}$$

This sum over neurons $j$ is the matrix product $dZ \cdot W^T$ at position $(i, k)$, so:

$$dX = dZ \cdot W^T$$

Shape: $(n_{examples}, n_{neurons}) \times (n_{neurons}, n_{inputs}) = (n_{examples}, n_{inputs})$ ✓

---

### 5. Mini-Batch Gradient Descent

There are three variants of gradient descent depending on how many examples are used per weight update:

| Variant | Examples per update | Gradient quality | Speed |
|---|---|---|---|
| Batch GD | All examples | Exact | Slow, memory intensive |
| Stochastic GD | 1 example | Very noisy | Fast but unstable |
| Mini-Batch GD | N examples (32–128) | Good approximation | Fast and stable |

Mini-batch is the standard in practice. The training loop becomes:

```
for each epoch:
    shuffle data randomly
    split into batches of size B
    for each batch:
        forward(X_batch)
        backward(y_pred_batch, y_batch)
        update weights
    record mean loss over all batches
```

The shuffle at the start of each epoch ensures batches are different every time, preventing the network from learning the order of the data. The noise between batches also helps escape local minima.

---

### 6. Weight Update (SGD)

Once the gradients are computed, each parameter is updated in the direction that reduces the loss:

$$W^{[l]} \leftarrow W^{[l]} - \eta \cdot dW^{[l]}$$
$$b^{[l]} \leftarrow b^{[l]} - \eta \cdot db^{[l]}$$

where $\eta$ is the learning rate.

---

### 7. Weight Initialization

Initializing all weights to zero causes all neurons in a layer to compute the same gradient and learn the same features — the symmetry problem. Weights must be initialized randomly, but the scale matters.

**Xavier initialization** — for Sigmoid and Tanh:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{inputs} + n_{neurons}}}\right)$$

**He initialization** — for ReLU:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{inputs}}}\right)$$

He uses a larger variance to compensate for the fact that ReLU kills all negative values, effectively halving the signal variance. The factor 2 in the numerator corrects for this.

---

### 8. Convolutional Neural Networks

Dense layers treat each input feature independently. For images this loses all spatial structure — a pixel has no relationship to its neighbors. CNNs exploit two properties of images:

**Locality** — relevant patterns (edges, curves, corners) are formed by nearby pixels.

**Translation invariance** — a pattern is the same pattern regardless of where it appears in the image. The same filter can detect it everywhere.

#### The Convolution Operation

A filter (kernel) of shape $(C_{in}, K, K)$ slides over the input volume. At each position $(i, j)$ it computes the dot product between itself and the input patch it covers:

$$\text{out}[f, i, j] = \sum_{c=1}^{C_{in}} \sum_{kh=0}^{K-1} \sum_{kw=0}^{K-1} X[c, i+kh, j+kw] \cdot W[f, c, kh, kw] + b[f]$$

With $F$ filters the output has shape $(N, F, H_{out}, W_{out})$ where:

$$H_{out} = \frac{H - K + 2P}{S} + 1 \qquad W_{out} = \frac{W - K + 2P}{S} + 1$$

and $P$ is padding, $S$ is stride.

#### im2col

Implementing convolution with explicit loops is correct but slow. The standard technique is **im2col**: reshape all input patches that the filter covers into columns of a matrix, then perform a single matrix multiplication.

Given input $X$ of shape $(N, C, H, W)$ and kernel size $K$:

```
X_col:  (C*K*K, N*H_out*W_out)    # each column is one flattened patch
W_row:  (F, C*K*K)                 # each row is one flattened filter
Z_col:  (F, N*H_out*W_out)         # output before reshape
```

This converts the convolution into a single matrix multiplication. The patch extraction uses `np.lib.stride_tricks.as_strided`, which creates a view of shape $(N, C, H_{out}, W_{out}, K, K)$ without copying data by manipulating the memory strides of the array directly.

#### ConvLayer Backward

Three gradients are needed:

**Gradient w.r.t. filters:**
$$dW = dZ_{col} \cdot X_{col}^T \quad \text{reshaped to } (F, C, K, K)$$

**Gradient w.r.t. bias:**
$$db[f] = \sum_{n,i,j} dZ[n, f, i, j]$$

**Gradient w.r.t. input (col2im):**

The inverse of im2col. Since each input position may belong to multiple overlapping patches, in the backward pass gradients from all those patches are accumulated with `+=`:

```python
for i in range(K):
    for j in range(K):
        dX_pad[:, :, i:i+S*H_out:S, j:j+S*W_out:S] += dX_col_reshaped[:, :, i, j, :, :]
```

#### Max Pooling

Divides each feature map into non-overlapping regions of size $P \times P$ and keeps only the maximum value. Output shape: $(N, C, H/P, W/P)$.

The forward pass uses reshape to group spatial positions into regions and `np.max` over the pooling axes:

```
X reshaped:  (N, C, H/P, P, W/P, P)
max over axes (3, 5) → (N, C, H/P, W/P)
```

The backward pass requires a boolean mask recording which position was the maximum in each region. Gradients flow only through those positions; all others receive zero:

```
forward:  mask = (X == max_per_region_expanded)
backward: dX = repeat(dA, P) * mask
```

#### FlattenLayer

Bridges convolutional and dense layers. Reshapes a volume $(N, C, H, W)$ into a matrix $(N, C \cdot H \cdot W)$ in the forward pass and reverses it in the backward pass using the shape saved during the forward. No learnable parameters.

#### CNN Architecture for MNIST

```
Input:           (N, 1, 28, 28)
ConvLayer(8):    (N, 8, 26, 26)    8 filters, 3x3, no padding
MaxPool(2x2):    (N, 8, 13, 13)
ConvLayer(16):   (N, 16, 11, 11)   16 filters, 3x3, no padding
MaxPool(2x2):    (N, 16, 5, 5)
FlattenLayer:    (N, 400)          16 * 5 * 5
DenseLayer(64):  (N, 64)
DenseLayer(10):  (N, 10)           Softmax output
```

---

## NeuralNetwork class

The `NeuralNetwork` class orchestrates all layer types. All layer types (`Layer`, `ConvLayer`, `PoolingLayer`, `FlattenLayer`) share the same interface (`forward` and `backward`), so `NeuralNetwork` treats them uniformly. Layers without learnable parameters are skipped in `update` using `hasattr`.

```python
net = NeuralNetwork(
    layers=[
        ConvLayer(1, 8, 3),
        PoolingLayer(2),
        FlattenLayer(),
        Layer(400, 64, ReLU()),
        Layer(64, 10, Softmax())
    ],
    loss=CategoricalCrossEntropy()
)
net.train(X, y, epochs=30, batch_size=128, lr=0.01)
```

---

## Validation

**XOR problem:** A 2→2→1 network trained on the 4 XOR examples. Since XOR is not linearly separable, solving it correctly confirms that forward pass, backpropagation, and weight updates all work.

**Gradient checking:** For each parameter $\theta$, the numerical gradient is computed as:

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$

and compared against the analytical gradient from backpropagation. Relative error below $10^{-5}$ confirms the implementation is correct:

$$\text{relative error} = \frac{\|g_{analytical} - g_{numerical}\|}{\|g_{analytical}\| + \|g_{numerical}\|}$$

---

## Roadmap

- [x] Activation functions (Sigmoid, ReLU, Tanh, Softmax)
- [x] Loss functions (MSE, BCE, Categorical Cross-Entropy)
- [x] Dense layer with forward and backward
- [x] NeuralNetwork class with mini-batch training
- [x] XOR experiment
- [x] Gradient checking
- [x] MNIST with dense layers (97.61% test accuracy)
- [x] FlattenLayer
- [x] PoolingLayer (Max Pooling)
- [x] ConvLayer (im2col + col2im)
- [ ] MNIST with CNN
- [ ] Advanced optimizers (Momentum, Adam)
- [ ] Regularization (L2, Dropout)
