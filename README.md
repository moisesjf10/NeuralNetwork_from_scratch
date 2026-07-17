# Neural Network from Scratch

Implementation of a fully connected neural network and a convolutional neural network using only NumPy, built to understand the mathematical foundations of deep learning.

## Motivation

The goal of this project is not to build the most efficient neural network, but to understand what happens inside one. Every operation — forward pass, backpropagation, weight update — is implemented from scratch without any deep learning framework.

## Project Structure

```
neural_net/
├── app_draw.py             # Streamlit handwritten digit drawing app
├── activations.py          # Activation functions and their derivatives
├── losses.py               # Loss functions and their derivatives
├── network.py              # NeuralNetwork class
├── optimizers.py           # Adam optimizer (SGD is built-in)
├── utils.py                # CharTokenizer, BPETokenizer, and data utilities
├── layers/
│   ├── __init__.py         # Package exports
│   ├── dense.py            # Fully connected layer (supporting 3D sequential input)
│   ├── flatten.py          # FlattenLayer
│   ├── pooling.py          # PoolingLayer (Max Pooling)
│   ├── conv.py             # ConvLayer (im2col / col2im)
│   ├── rnn.py              # RNNLayer with Backpropagation Through Time
│   ├── embedding.py        # EmbeddingLayer with sparse gradient accumulation
│   ├── positional_encoding.py # Sinusoidal Positional Encoding
│   ├── layernorm.py        # Layer Normalization (LayerNorm)
│   ├── multihead_attention.py # MultiHeadAttention with causal masking and 4D backprop
│   └── transformer_block.py # TransformerBlock and PositionwiseFeedForward
├── models/                 # Saved weights (.pkl / .npz)
└── experiments/
    ├── test.py                 # Full test suite (19 tests: Dense, CNN, RNN)
    ├── train_mnist.py          # Dense network on MNIST
    ├── train_mnist_cnn.py      # CNN on MNIST
    ├── train_mnist_compare.py  # SGD vs Adam comparison
    ├── train_cifar10.py        # 3-layer CNN on CIFAR-10
    ├── train_rnn.py            # Character-level RNN text generation
    ├── train_llm.py            # Character-level Transformer text generation
    ├── train_llm_bpe.py        # BPE-level Transformer text generation with Adam
    ├── visualize_features.py   # Filters and activation maps visualizer
    └── plots/                  # Generated plots
```

## Dependencies

```
numpy
matplotlib
jupyter
scikit-learn
opencv-python
streamlit
streamlit-drawable-canvas
```

Install with:
```bash
pip install numpy matplotlib jupyter scikit-learn opencv-python streamlit streamlit-drawable-canvas
```

Run experiments from the project root with:
```bash
python3 -m experiments.train_mnist
```

---

## Results

| Architecture | Optimizer | Dataset | Train Accuracy | Test Accuracy |
|---|---|---|---|---|
| Dense 784→128→64→10 | SGD (lr=0.1) | MNIST | 99.98% | 97.61% |
| CNN (p=1) | SGD (lr=0.01) | MNIST | 98.88% | 98.16% |

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
        zero_grad()
        forward(X_batch)
        backward(y_pred_batch, y_batch)
        optimizer.step() or update(lr)
    record mean loss over all batches
```

The shuffle at the start of each epoch ensures batches are different every time, preventing the network from learning the order of the data. The noise between batches also helps escape local minima.

---

### 6. Optimizers

#### SGD — Stochastic Gradient Descent

The simplest update rule. Every parameter is updated by the same learning rate $\eta$:

$$W^{[l]} \leftarrow W^{[l]} - \eta \cdot dW^{[l]}$$
$$b^{[l]} \leftarrow b^{[l]} - \eta \cdot db^{[l]}$$

Simple and effective for well-tuned problems, but uses the same learning rate for all parameters regardless of their gradient history.

---

#### Adam — Adaptive Moment Estimation

Adam combines two ideas: **Momentum** (smooth gradient directions using a running average) and **RMSProp** (adapt the learning rate per parameter using the magnitude of past gradients).

For each parameter, Adam maintains two state vectors updated at every step $t$:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

- $m_t$ — first moment: running average of gradients (direction)
- $v_t$ — second moment: running average of squared gradients (magnitude)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ — decay rates (hyperparameters)

**Bias correction** — at the start of training $m_0 = v_0 = 0$, which biases the estimates toward zero. The correction amplifies the moments in early steps and fades as $t$ grows:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Weight update:**

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $\eta = 0.001$ and $\epsilon = 10^{-8}$.

The denominator $\sqrt{\hat{v}_t}$ normalizes each parameter's update by its gradient history — parameters with consistently large gradients receive smaller updates, and vice versa. This makes Adam robust to different learning rate scales across layers.

**Usage:**

```python
from optimizers import Adam

optimizer = Adam(lr=0.001)
net.train(X, y, epochs=30, batch_size=128, lr=0.001, optimizer=optimizer)
```

Adam is slower per step than SGD (more operations per parameter) but typically converges in fewer epochs. On simple problems like MNIST both reach similar final accuracy; on harder problems Adam's advantage is more pronounced.

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
Input:              (N, 1, 28, 28)
ConvLayer(8, p=1):  (N, 8, 28, 28)    8 filters, 3x3, padding=1
MaxPool(2x2):       (N, 8, 14, 14)
ConvLayer(16, p=1): (N, 16, 14, 14)   16 filters, 3x3, padding=1
MaxPool(2x2):       (N, 16, 7, 7)
FlattenLayer:       (N, 784)           16 * 7 * 7
DenseLayer(64):     (N, 64)
DenseLayer(10):     (N, 10)            Softmax output
```
### 9. Recurrent Neural Networks (RNN)

Unlike feedforward networks, Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state vector $h_t$ that acts as a memory of past inputs.

#### Mathematical Formulation

For each time step $t \in \{1, \dots, T\}$, given input $x_t$ of shape $(1, n_{inputs})$ and the previous hidden state $h_{t-1}$ of shape $(1, n_{hidden})$:

$$Z_t = x_t W_{hx} + h_{t-1} W_{hh} + b$$
$$h_t = \tanh(Z_t)$$

where:
- $W_{hx}$ is the input-to-hidden weight matrix of shape $(n_{inputs}, n_{hidden})$ (initialized using Xavier/Glorot Normal initialization)
- $W_{hh}$ is the hidden-to-hidden weight matrix of shape $(n_{hidden}, n_{hidden})$ (initialized using Orthogonal initialization to preserve gradient norm across time steps)
- $b$ is the bias vector of shape $(1, n_{hidden})$ (initialized to zero)
- $h_0$ is the initial hidden state, initialized to zero.

The output at each time step is the hidden state $h_t$. The output of the layer has shape $(N, T, n_{hidden})$ for a batch size $N$ and sequence length $T$.

#### Backpropagation Through Time (BPTT)

The backward pass propagates gradients back through time, from the last step $T$ to the first step $1$.

At step $t$, the total gradient of the loss with respect to the hidden state $h_t$ is the sum of the gradient coming from the layer above ($dA_t$) and the gradient coming from the future step $t+1$ ($dh_{next}$):

$$dh_t = dA_t + dh_{next}$$

The gradient of the pre-activation $Z_t$ is:

$$dZ_t = dh_t \odot \tanh'(Z_t) = dh_t \odot (1 - h_t^2)$$

Using $dZ_t$, we accumulate the parameter gradients over the sequence length $T$:

$$dW_{hx} = \sum_{t=1}^T x_t^T dZ_t$$
$$dW_{hh} = \sum_{t=1}^T h_{t-1}^T dZ_t$$
$$db = \sum_{t=1}^T \sum_{i=1}^N (dZ_t)_i$$

The gradient to be propagated to the input at step $t$ is:

$$dx_t = dZ_t W_{hx}^T$$

And the gradient to be propagated to the previous hidden state (back in time) is:

$$dh_{next\_prev} = dZ_t W_{hh}^T$$

#### Gradient Clipping

To prevent the vanishing/exploding gradient problem common in RNNs, parameter gradients are clipped element-wise to the range $[-5.0, 5.0]$ before updating:

$$g \leftarrow \max(\min(g, 5.0), -5.0) \quad \text{for } g \in \{dW_{hx}, dW_{hh}, db\}$$

---

### 10. Transformer Architecture (LLMs)

Unlike recurrent networks that process sequence elements step-by-step, the Transformer architecture processes entire sequences in parallel. It uses self-attention to draw global dependencies, bypassing recurrence entirely. Below are the technical and mathematical details of the modules implemented in the project.

#### 10.1 Embedding Layer (`layers/embedding.py`)

The `EmbeddingLayer` acts as a lookup table mapping discrete token IDs (categories) into dense continuous vector space.

*   **Initialization (`__init__(self, vocab_size, embedding_dim)`):**
    *   `vocab_size` ($V$): Total number of unique tokens/characters in the vocabulary.
    *   `embedding_dim` ($D$): The size of the dense continuous vector space.
    *   `W` (Parameters): Weight lookup table of shape `(vocab_size, embedding_dim)`. Initialized with a small random normal distribution scaled by $0.01$: `np.random.randn(vocab_size, embedding_dim) * 0.01`.
    *   `dW` (Gradients): Gradient accumulator array of shape `(vocab_size, embedding_dim)`.
*   **Forward Pass (`forward(self, X)`):**
    *   Input `X`: Matrix of integer token IDs of shape `(Batch_Size, Time_Steps)`.
    *   Output: 3D Continuous Dense Tensor of shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   **Implementation Detail:** Mathematically, this projection is equivalent to $E = X_{\text{one-hot}} W$. However, a sparse matrix multiplication is computationally inefficient. Instead, we use NumPy's advanced indexing `self.W[X]` to retrieve the rows in $O(1)$ time, bypassing floating-point operations.
*   **Backward Pass (`backward(self, dA)`):**
    *   Input `dA`: Upstream gradient of shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   Output: `None`. Since inputs are discrete integers (non-differentiable space), the gradient flow terminates at this layer.
    *   **Sparse Gradient Accumulation:** To compute `dW`, we sum all incoming gradients matching the respective token positions. Because a token can occur multiple times in the input `X`, we must accumulate these gradients atomically. Using direct assignment (`self.dW[self.inputs] += dA`) fails due to NumPy's internal memory buffering, which overwrites duplicate indices. We resolve this by using `np.add.at(self.dW, self.inputs, dA)`, which guarantees an unbuffered, sequential sum:
        $$dW_k = \sum_{\{b, t \mid X_{b, t} = k\}} dA_{b, t}$$

#### 10.2 Positional Encoding (`layers/positional_encoding.py`)

Since self-attention is permutation-invariant, it is blind to sequence order. Sinusoidal positional encoding injects positional coordinates by adding a static wave-like signal to the input embeddings.

*   **Initialization (`__init__(self, max_seq_length, embedding_dim)`):**
    *   `max_seq_length`: The maximum length of a sequence the model will support.
    *   `embedding_dim` ($D$): Dimension of features, matching the upstream `EmbeddingLayer`.
    *   `pe`: Static matrix of shape `(max_seq_length, embedding_dim)` containing precomputed values:
        $$PE(pos, 2i) = \sin\left(pos \cdot e^{-2i \cdot \frac{\ln(10000)}{D}}\right)$$
        $$PE(pos, 2i+1) = \cos\left(pos \cdot e^{-2i \cdot \frac{\ln(10000)}{D}}\right)$$
        where $pos$ is the sequence position index ($0 \dots \text{max\_seq\_length}-1$), and $2i$ / $2i+1$ are the even/odd feature dimensions.
*   **Forward Pass (`forward(self, X)`):**
    *   Input `X`: Shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   Output: Shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   **Implementation Detail:** Slices the pre-computed table to match the current sequence length and broadcasts the addition across the batch dimension:
        $$Z = X + PE_{:T, :}$$
*   **Backward Pass (`backward(self, dA)`):**
    *   Input `dA`: Upstream gradient of shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   Output: `dA`. Since the operation is a simple addition of a constant ($Z = X + C$), the partial derivative $\frac{\partial Z}{\partial X} = 1$, so gradients flow backwards completely unchanged.

#### 10.3 Layer Normalization (`layers/layernorm.py`)

`LayerNorm` stabilizes the hidden activations of deep networks by standardizing the features of each sequence element (token representation) to have zero mean and unit variance.

*   **Initialization (`__init__(self, embedding_dim, epsilon=1e-5)`):**
    *   `embedding_dim` ($D$): The dimension of the feature vectors.
    *   `gamma` (Parameters): Trainable scale parameter of shape `(embedding_dim,)`. Initialized to ones.
    *   `beta` (Parameters): Trainable shift parameter of shape `(embedding_dim,)`. Initialized to zeros.
    *   `dgamma`, `dbeta` (Gradients): Gradient arrays matching parameter shapes.
*   **Forward Pass (`forward(self, X)`):**
    *   Input `X`: Shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   Output: Shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   **Computation:** Computes mean $\mu$ and variance $\sigma^2$ along the last dimension ($D$) with `keepdims=True` (shape `(B, T, 1)`):
        $$\mu_{b,t} = \frac{1}{D} \sum_{i=1}^D X_{b,t,i} \qquad \sigma^2_{b,t} = \frac{1}{D} \sum_{i=1}^D (X_{b,t,i} - \mu_{b,t})^2$$
        $$\hat{X}_{b,t} = \frac{X_{b,t} - \mu_{b,t}}{\sqrt{\sigma^2_{b,t} + \epsilon}}$$
        $$Y_{b,t} = \gamma \odot \hat{X}_{b,t} + \beta$$
        where $\epsilon = 10^{-5}$ prevents division by zero.
*   **Backward Pass (`backward(self, d_out)`):**
    *   Input `d_out`: Shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   Output `dX`: Shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   **Parameter Gradients:** Summed across batch and time dimensions since scale and shift are shared across all tokens:
        $$d\gamma = \sum_{b=1}^B \sum_{t=1}^T dY_{b,t} \odot \hat{X}_{b,t} \qquad d\beta = \sum_{b=1}^B \sum_{t=1}^T dY_{b,t}$$
    *   **Input Gradient:** Analytical simplification of the backward pass through standard deviation and mean computations:
        $$dX\_norm = d\_out \odot \gamma$$
        $$dX = \frac{1}{D \cdot std} \left[ D \cdot dX\_norm - \sum_{k=1}^D dX\_norm_k - \hat{X} \odot \sum_{k=1}^D (dX\_norm_k \odot \hat{X}_k) \right]$$
        where the sums $\sum_{k=1}^D$ are computed along the last axis (`axis=-1, keepdims=True`).

#### 10.4 Causal Multi-Head Self-Attention (`layers/multihead_attention.py`)

Causal Multi-Head Self-Attention maps queries, keys, and values to compute contextual representations, using a causal mask to enforce autoregressive generation (tokens cannot attend to future positions).

*   **Initialization (`__init__(self, embedding_dim, num_heads)`):**
    *   `embedding_dim` ($D$): Total dimension of the model.
    *   `num_heads` ($H$): Number of parallel attention heads (must be a divisor of $D$).
    *   `d_k`: Key dimension per head ($d_k = D / H$).
    *   `W_q`, `W_k`, `W_v` (Parameters): Projection weight matrices of shape `(Embedding_Dim, Embedding_Dim)`.
    *   `W_o` (Parameters): Output projection weight matrix of shape `(Embedding_Dim, Embedding_Dim)`.
    *   All weights are initialized with Xavier/Glorot Normal distribution: standard deviation $sd = \sqrt{\frac{2}{D + D}}$.
    *   `dW_q`, `dW_k`, `dW_v`, `dW_o` (Gradients): Parameter gradient accumulators.
*   **Forward Pass (`forward(self, X)`):**
    *   Input `X`: Shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   Output `Z`: Shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   **Sequence of Operations:**
        1.  *Projection:* Project inputs to query, key, and value vectors:
            $$Q = X W_q \qquad K = X W_k \qquad V = X W_v \quad \text{shape: } (B, T, D)$$
        2.  *Multi-Head Split:* Split vectors across $H$ heads, transposing to group batch and head indices:
            $$Q_{split} = \text{transpose}(Q\text{.reshape}(B, T, H, d_k), \text{axes}=[0, 2, 1, 3]) \quad \text{shape: } (B, H, T, d_k)$$
            Same layout applies to $K_{split}$ and $V_{split}$.
        3.  *Scaled Dot-Product:* Calculate raw attention scores between queries and keys:
            $$\text{scores} = \frac{Q_{split} K_{split}^T}{\sqrt{d_k}} \quad \text{shape: } (B, H, T, T)$$
        4.  *Causal Masking:* Apply an upper triangular matrix mask (offset by $k=1$) to mask out elements where sequence index $j > i$. Causal positions receive an additive penalty of $-10^9$, which collapses to zero attention during Softmax:
            $$\text{scores}_{b,h,i,j} = \text{scores}_{b,h,i,j} - 10^9 \cdot \mathbb{1}[j > i]$$
        5.  *Attention Weights:* Compute normalized probabilities:
            $$A = \text{softmax}(\text{scores}, \text{axis}=-1) \quad \text{shape: } (B, H, T, T)$$
        6.  *Context Aggregation:* Compute weighted sum of values:
            $$\text{context} = A V_{split} \quad \text{shape: } (B, H, T, d_k)$$
        7.  *Concatenation:* Merge head outputs and project back to standard dimension:
            $$\text{context\_concat} = \text{transpose}(\text{context}, [0, 2, 1, 3])\text{.reshape}(B, T, D)$$
            $$Z = \text{context\_concat} W_o \quad \text{shape: } (B, T, D)$$
*   **Backward Pass (`backward(self, dZ)`):**
    *   Input `dZ`: Upstream gradient of shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   Output `dX`: Downstream gradient of shape `(Batch_Size, Time_Steps, Embedding_Dim)`.
    *   **Computation (4D Backpropagation):**
        1.  *Output Projection Gradients:*
            $$dZ\_flat = dZ\text{.reshape}(B \cdot T, D) \qquad \text{context\_flat} = \text{context\_concat}\text{.reshape}(B \cdot T, D)$$
            $$dW_o = \text{context\_flat}^T \cdot dZ\_flat \qquad d\_context\_concat = dZ \cdot W_o^T$$
        2.  *Reverting Multi-Head Concatenation:*
            $$d\_context = \text{transpose}(d\_context\_concat\text{.reshape}(B, T, H, d_k), [0, 2, 1, 3]) \quad \text{shape: } (B, H, T, d_k)$$
        3.  *Attention Value Gradients:*
            $$dV_{split} = A^T \cdot d\_context \quad \text{shape: } (B, H, T, d_k)$$
            $$dA = d\_context \cdot V_{split}^T \quad \text{shape: } (B, H, T, T)$$
        4.  *Softmax Derivative:*
            $$d\_scores = \frac{1}{\sqrt{d_k}} \cdot A \odot \left( dA - \sum_{j=1}^T (dA_j \odot A_j) \right) \quad \text{shape: } (B, H, T, T)$$
            where the sum and dot-products are calculated along the last axis `axis=-1`.
        5.  *Attention Query & Key Gradients:*
            $$dQ_{split} = d\_scores \cdot K_{split} \quad \text{shape: } (B, H, T, d_k)$$
            $$dK_{split} = d\_scores^T \cdot Q_{split} \quad \text{shape: } (B, H, T, d_k)$$
        6.  *Re-combining Heads & Input Projections:*
            Flatten and reshape $dQ_{split}, dK_{split}, dV_{split}$ back to `(B, T, D)`:
            $$dQ = \text{transpose}(dQ_{split}, [0, 2, 1, 3])\text{.reshape}(B \cdot T, D)$$
            $$dK = \text{transpose}(dK_{split}, [0, 2, 1, 3])\text{.reshape}(B \cdot T, D)$$
            $$dV = \text{transpose}(dV_{split}, [0, 2, 1, 3])\text{.reshape}(B \cdot T, D)$$
            Compute weight matrix gradients (by flattening sequence dimension):
            $$dW_q = X\_flat^T \cdot dQ \qquad dW_k = X\_flat^T \cdot dK \qquad dW_v = X\_flat^T \cdot dV$$
            where `X_flat` has shape `(B * T, D)`.
        7.  *Input Downstream Gradient:*
            $$dX = dQ \cdot W_q^T + dK \cdot W_k^T + dV \cdot W_v^T \quad \text{shape: } (B, T, D)$$

#### 10.5 Transformer Block (`layers/transformer_block.py`)

A pre-LayerNorm Transformer Block combining normalization, self-attention, and a position-wise feed-forward network with residual skip connections.

*   **Sub-components:**
    *   `norm1`, `norm2`: Instances of `LayerNorm` with dimension `d_model`.
    *   `attention`: An instance of `MultiHeadAttention`.
    *   `ff`: An instance of `PositionwiseFeedForward`, containing:
        *   `layer1`: A dense `Layer(d_model, d_ff, ReLU())`.
        *   `layer2`: A dense `Layer(d_ff, d_model, Linear())`.
*   **Forward Pass (`forward(self, X)`):**
    *   Input `X`: Shape `(Batch_Size, Time_Steps, d_model)`.
    *   Output: Shape `(Batch_Size, Time_Steps, d_model)`.
    *   **Computation (Pre-LayerNorm Order):**
        $$Z_1 = X + \text{attention.forward}(\text{norm1.forward}(X))$$
        $$Z_2 = Z_1 + \text{ff.forward}(\text{norm2.forward}(Z_1))$$
*   **Backward Pass (`backward(self, dZ2)`):**
    *   Input `dZ2`: Upstream gradient of shape `(Batch_Size, Time_Steps, d_model)`.
    *   Output `dX`: Downstream gradient of shape `(Batch_Size, Time_Steps, d_model)`.
    *   **Residual Flow Computation:** Gradients are copied and flow through the residual branches, accumulating at the junctions:
        1.  *Feed-Forward Backward:*
            $$d\_ff = \text{ff.backward}(dZ2)$$
            $$d\_norm2 = \text{norm2.backward}(d\_ff)$$
            $$dZ1 = dZ2 + d\_norm2$$
        2.  *Attention Backward:*
            $$d\_attn = \text{attention.backward}(dZ1)$$
            $$d\_norm1 = \text{norm1.backward}(d\_attn)$$
            $$dX = dZ1 + d\_norm1$$

---

## NeuralNetwork class

The `NeuralNetwork` class orchestrates all layer types. All layer types (`Layer`, `ConvLayer`, `PoolingLayer`, `FlattenLayer`) share the same interface (`forward` and `backward`), so `NeuralNetwork` treats them uniformly. Layers without learnable parameters are skipped in `update` using `hasattr`.

The `train` method accepts an optional `optimizer` argument. If none is provided it falls back to SGD via `update(lr)`:

```python
# SGD
net.train(X, y, epochs=30, batch_size=128, lr=0.1)

# Adam
from optimizers import Adam
net.train(X, y, epochs=30, batch_size=128, lr=0.001, optimizer=Adam(lr=0.001))
```

This follows the **dependency injection** pattern: `NeuralNetwork` does not import or instantiate any optimizer. The optimizer is passed in from outside, so the network works with any object that implements `step(layers)`.

---

## Saving and Loading Weights

Trained weights are serialized with `pickle` and saved to a `.pkl` file. The architecture must be reconstructed manually before loading — only the parameter values are stored, not the layer types or hyperparameters.

```python
# Save after training
net.save_weights('models/mnist_cnn.pkl')

# Load in a new session
net = NeuralNetwork(layers=[...], loss=...)   # same architecture
net.load_weights('models/mnist_cnn.pkl')
```

---

## Validation

**XOR problem:** A 2→2→1 network trained on the 4 XOR examples. Since XOR is not linearly separable, solving it correctly confirms that forward pass, backpropagation, and weight updates all work.

**Gradient checking:** For each parameter $\theta$, the numerical gradient is computed as:

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$

and compared against the analytical gradient from backpropagation. Relative error below $10^{-5}$ confirms the implementation is correct:

$$\text{relative error} = \frac{\|g_{analytical} - g_{numerical}\|}{\|g_{analytical}\| + \|g_{numerical}\|}$$

The gradient checker uses `hasattr` to skip layers without learnable parameters (`FlattenLayer`, `PoolingLayer`), and works with all layer types including `ConvLayer`.

## Interactive Applications & Experiments

This project includes interactive applications and more complex training scripts to test the modular components.

### 1. Interactive Drawing App (`app_draw.py`)

A real-time handwritten digit recognition application built using **Streamlit**.

- **Workflow:**
  1. Loads a pre-trained MNIST CNN model (`models/cnn_mnist.pkl`).
  2. Renders a canvas where the user can draw a digit (0–9).
  3. Preprocesses the drawn canvas image: converts RGBA to grayscale, resizes to $28 \times 28$ pixels using OpenCV, normalizes to $[0, 1]$, and shapes into a batch of size 1: `(1, 1, 28, 28)`.
  4. Runs a forward pass through the CNN.
  5. Displays the predicted digit, the model's confidence percentage, and a bar chart showing the probability distribution over all classes.
- **Run the app:**
  ```bash
  streamlit run app_draw.py
  ```

### 2. Character-Level Text Generation (`experiments/train_rnn.py`)

Uses a Recurrent Neural Network (RNN) to learn character patterns and generate new text.

- **Utilities (`utils.py`):** Includes a `CharTokenizer` that fits a vocabulary from a raw text dataset, maps characters to integers bidirectional, constructs sequence sliding windows (`X` and `Y` shifted by 1), and generates 3D one-hot tensors.
- **Network:** An `RNNLayer` (64 hidden units) followed by a Dense classification `Layer` with `Softmax` activation.
- **Sampling Strategy:** Uses statistical sampling (`np.random.choice` based on output probabilities) rather than argmax to prevent the network from getting stuck in repetitive loops.
- **Run the training:**
  ```bash
  python3 -m experiments.train_rnn
  ```

### 3. CIFAR-10 Classification CNN (`experiments/train_cifar10.py`)

A training script demonstrating a larger convolutional network trained on the 10-class CIFAR-10 dataset (RGB color images of size $3 \times 32 \times 32$).

- **Architecture:**
  `Conv(3→32, 3x3) -> MaxPool(2) -> Conv(32→64, 3x3) -> MaxPool(2) -> Conv(64→128, 3x3) -> MaxPool(2) -> Flatten -> Dense(256, ReLU) -> Dense(10, Softmax)`
- **Features:** Supports automated scikit-learn dataset download or loading the official dataset binary batches. Outputs validation accuracy, loss curves, confusion matrices, and per-class accuracy plots to `experiments/plots/cifar10/`.
- **Run the training:**
  ```bash
  python3 -m experiments.train_cifar10
  ```

### 4. Feature & Filter Visualization (`experiments/visualize_features.py`)

Inspects what the network has learned by visualizing convolutional layers.

- **Filter Visualizer:** Plots the learned $3 \times 3$ weights of the first convolutional layer as matrices using a divergent colormap (`RdBu`), exposing edge-detectors.
- **Activation Maps:** Passes a sample image (e.g., MNIST digit) and extracts the 8 resulting feature maps from `ConvLayer` to visually inspect what pixel regions activate the filters.
- **Run the script:**
  ```bash
  python3 -m experiments.visualize_features
  ```

### 5. Large Language Model (LLM) Training & BPE Tokenizer

Two experiments demonstrating character-level and subword-level text generation using a NumPy-only Pre-LayerNorm Transformer, incorporating a custom subword tokenizer.

#### 5.1 Byte-Pair Encoding (`BPETokenizer` in `utils.py`)

Subword tokenization compresses a text sequence by representing frequent character strings under singular token IDs. The `BPETokenizer` class manages this vocabulary mapping.

*   **Initialization (`__init__(self)`):**
    *   `merges`: Dictionary mapping byte-coordinate tuples `(left_id, right_id) -> new_id`.
    *   `vocab`: Dictionary mapping integer index IDs to byte literals `id -> bytes`.
    *   `vocab_size`: Initialized to $256$ to represent basic UTF-8 characters.
*   **Training (`train(self, text, vocab_size)`):**
    1.  Encodes the training text into a list of elemental bytes (numbers 0 to 255).
    2.  Iteratively searches for the most frequent adjacent pair of byte IDs in the corpus using helper `_get_stats`.
    3.  Assigns a new token ID (`256 + i`) to this pair, registers the merge rule in `self.merges`, and updates `self.vocab` with the concatenated byte string.
    4.  Replaces all occurrences of the pair in the byte list with the new token ID using `_merge`.
    5.  Repeats the cycle until the vocabulary reaches the target `vocab_size`.
*   **Encoding (`encode(self, text)`):**
    *   Translates text to UTF-8 bytes and iteratively applies the learned `merges` in order of their occurrence during training (resolving conflicts by prioritizing the merge rule with the lowest target token ID).
*   **Decoding (`decode(self, ids)`):**
    *   Joins the dictionary bytes representing each token ID in `ids` and decodes them to a string. The decoding uses `errors='replace'` to safely handle invalid byte predictions from the model.
*   **Dataset Prep (`create_dataset(self, text, seq_length, stride=1)`):**
    *   Tokenizes the input text and constructs rolling sequence windows.
    *   Returns sequence integer IDs `X` of shape `(Num_Windows, seq_length)` (passed directly to the Embedding Layer), and target sequences `Y` encoded as 3D One-Hot vectors of shape `(Num_Windows, seq_length, vocab_size)` (for CCE loss calculations).

#### 5.2 Training Experiments (`train_llm.py` & `train_llm_bpe.py`)

*   **Architectural Layout:**
    $$\text{EmbeddingLayer}(V, D) \rightarrow \text{PositionalEncoding}(T, D) \rightarrow \text{TransformerBlock}(d\_model=D, heads=H, d\_ff) \times 2 \rightarrow \text{DenseLayer}(D, V, \text{Softmax})$$
    where vocabulary size $V \approx 280$, sequence length $T = 32$, embedding size $D = 64$, attention heads $H = 4$, and feedforward dimension $d\_ff = 256$.
*   **Optimization Loop:**
    Trains using the adaptive moment estimation [Adam](file:///home/moisesjf10/GitHub/NN/optimizers.py#L12) optimizer (`lr=0.001`), saving trained weights to `transformer_weights.pkl` or `transformer_bpe_weights.pkl`.
*   **Autoregressive Text Inference:**
    Generates text by iteratively predicting the next token, appending it to the context, and rolling the window forward. To prevent repetitive text loops, inference uses a **Temperature Scaling** parameter:
    $$P(x_i) = \frac{e^{z_i / T}}{\sum_{j=1}^V e^{z_j / T}}$$
    *   `T = 0.2` (Strict/Low entropy): The distribution narrows, leading to highly deterministic, confident predictions.
    *   `T = 0.8` (Creative/High entropy): The distribution flattens, introducing diversity and creativity.
*   **Running Scripts:**
    ```bash
    python3 -m experiments.train_llm
    python3 -m experiments.train_llm_bpe
    ```

---

## Roadmap

- [x] Activation functions (Sigmoid, ReLU, Tanh, Softmax)
- [x] Loss functions (MSE, BCE, Categorical Cross-Entropy)
- [x] Dense layer with forward and backward
- [x] NeuralNetwork class with mini-batch training
- [x] XOR experiment
- [x] Gradient checking (including CNN/RNN backward checks)
- [x] MNIST with dense layers (97.61% test accuracy)
- [x] FlattenLayer
- [x] PoolingLayer (Max Pooling)
- [x] ConvLayer (im2col + col2im)
- [x] MNIST with CNN (98.16% test accuracy)
- [x] Adam optimizer
- [x] SGD vs Adam comparison experiment
- [x] Save and load weights
- [x] Recurrent Neural Network (RNNLayer & BPTT)
- [x] Text generation (Char-level RNN)
- [x] CNN on CIFAR-10 dataset
- [x] Feature/filter visualization (CNN weights & activations)
- [x] Interactive digit drawing web app (Streamlit)
- [x] Embedding Layer with sparse gradient accumulation (`np.add.at`)
- [x] Sinusoidal Positional Encoding
- [x] Layer Normalization (LayerNorm)
- [x] Causal Multi-Head Self-Attention (4D backpropagation)
- [x] Pre-LayerNorm Transformer Block & Position-wise FFN
- [x] Byte-Pair Encoding Tokenizer (BPETokenizer)
- [x] Autoregressive text generation with temperature scaling
- [ ] Regularization (L2, Dropout)
- [ ] Learning rate scheduling
- [ ] Batch Normalization
