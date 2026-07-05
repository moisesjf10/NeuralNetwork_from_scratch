import numpy as np

# ─────────────────────────────────────────────
from activations import ReLU, Sigmoid, Tanh, Softmax
from losses import MSE, BCE, CategoricalCrossEntropy
from network import NeuralNetwork
from layers import Layer, FlattenLayer, PoolingLayer, ConvLayer
# ─────────────────────────────────────────────

# ══════════════════════════════════════════════
# TEST 1 — Activation functions
# ══════════════════════════════════════════════

def test_activations():
    print("=" * 50)
    print("TEST 1 — Activation functions")
    print("=" * 50)

    sigmoid = Sigmoid()
    relu = ReLU()
    tanh = Tanh()

    # Known values computed by hand
    checks = [
        ("sigmoid(0)",          sigmoid(np.array([0.0])),          np.array([0.5])),
        ("sigmoid(2)",          sigmoid(np.array([2.0])),          np.array([0.8808])),
        ("sigmoid_deriv(0)",    sigmoid.derivative(np.array([0.0])), np.array([0.25])),
        ("relu(3)",             relu(np.array([3.0])),             np.array([3.0])),
        ("relu(-3)",            relu(np.array([-3.0])),            np.array([0.0])),
        ("relu_deriv(3)",       relu.derivative(np.array([3.0])),  np.array([1.0])),
        ("relu_deriv(-3)",      relu.derivative(np.array([-3.0])), np.array([0.0])),
        ("tanh(0)",             tanh(np.array([0.0])),             np.array([0.0])),
        ("tanh_deriv(0)",       tanh.derivative(np.array([0.0])),  np.array([1.0])),
    ]

    all_passed = True
    for name, got, expected in checks:
        passed = np.allclose(got, expected, atol=1e-4)
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: expected {expected[0]:.4f}, got {got[0]:.4f}")
        if not passed:
            all_passed = False

    print(f"\n  {'All activation tests passed' if all_passed else 'Some tests FAILED'}\n")
    return all_passed


# ══════════════════════════════════════════════
# TEST 2 — Loss functions
# ══════════════════════════════════════════════

def test_losses():
    print("=" * 50)
    print("TEST 2 — Loss functions")
    print("=" * 50)

    mse = MSE()
    bce = BCE()

    y_pred = np.array([[1.0], [0.0]])
    y_true = np.array([[1.0], [0.0]])

    y_pred_wrong = np.array([[1.0], [0.0]])
    y_true_wrong = np.array([[0.0], [1.0]])

    checks = [
        # Perfect predictions → loss should be 0
        ("mse — perfect prediction",  mse(y_pred, y_true),           0.0),
        # Completely wrong predictions → loss should be 1
        ("mse — completely wrong",     mse(y_pred_wrong, y_true_wrong), 1.0),
        # Simple known value
        ("mse — single example",       mse(np.array([[1.0]]), np.array([[0.0]])), 1.0),
    ]

    all_passed = True
    for name, got, expected in checks:
        passed = np.isclose(got, expected, atol=1e-6)
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: expected {expected:.4f}, got {got:.4f}")
        if not passed:
            all_passed = False

    # Check derivative shape
    d = mse.derivative(np.array([[0.8], [0.3]]), np.array([[1.0], [0.0]]))
    shape_ok = d.shape == (2, 1)
    print(f"  {'✅' if shape_ok else '❌'} mse_derivative shape: expected (2,1), got {d.shape}")
    if not shape_ok:
        all_passed = False

    print(f"\n  {'All loss tests passed' if all_passed else 'Some tests FAILED'}\n")
    return all_passed


# ══════════════════════════════════════════════
# TEST 3 — Forward pass shapes
# ══════════════════════════════════════════════

def test_forward_shapes():
    print("=" * 50)
    print("TEST 3 — Forward pass shapes")
    print("=" * 50)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)  # (4, 2)
    y = np.array([[0], [1], [1], [0]], dtype=float)               # (4, 1)

    net = NeuralNetwork(
        layers=[
            Layer(2, 4, ReLU()),
            Layer(4, 1, Sigmoid())
        ],
        loss=MSE()
    )

    all_passed = True

    # Check weight shapes
    checks_shapes = [
        ("Layer 0 — W shape", net.layers[0].W.shape, (2, 4)),
        ("Layer 0 — b shape", net.layers[0].b.shape, (1, 4)),
        ("Layer 1 — W shape", net.layers[1].W.shape, (4, 1)),
        ("Layer 1 — b shape", net.layers[1].b.shape, (1, 1)),
    ]

    for name, got, expected in checks_shapes:
        passed = got == expected
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: expected {expected}, got {got}")
        if not passed:
            all_passed = False

    # Check forward output shape
    y_pred = net.forward(X)
    passed = y_pred.shape == (4, 1)
    print(f"  {'✅' if passed else '❌'} forward output shape: expected (4, 1), got {y_pred.shape}")
    if not passed:
        all_passed = False

    # Check intermediate shapes saved in layers
    passed_z0 = net.layers[0].Z.shape == (4, 4)
    passed_x0 = net.layers[0].X.shape == (4, 2)
    passed_z1 = net.layers[1].Z.shape == (4, 1)
    passed_x1 = net.layers[1].X.shape == (4, 4)

    print(f"  {'✅' if passed_z0 else '❌'} Layer 0 — Z shape: expected (4, 4), got {net.layers[0].Z.shape}")
    print(f"  {'✅' if passed_x0 else '❌'} Layer 0 — X shape: expected (4, 2), got {net.layers[0].X.shape}")
    print(f"  {'✅' if passed_z1 else '❌'} Layer 1 — Z shape: expected (4, 1), got {net.layers[1].Z.shape}")
    print(f"  {'✅' if passed_x1 else '❌'} Layer 1 — X shape: expected (4, 4), got {net.layers[1].X.shape}")

    all_passed = all_passed and passed_z0 and passed_x0 and passed_z1 and passed_x1

    print(f"\n  {'All shape tests passed' if all_passed else 'Some tests FAILED'}\n")
    return all_passed


# ══════════════════════════════════════════════
# TEST 4 — Gradient checking
# ══════════════════════════════════════════════

def gradient_check(net, X, y, epsilon=1e-5):
    """
    Compares analytical gradients from backprop against
    numerical gradients computed by finite differences.

    For each parameter θ:
        numerical gradient ≈ (L(θ+ε) - L(θ-ε)) / (2ε)

    If the relative error is below 1e-5, backprop is correct.
    Works with all layer types — skips layers without learnable parameters
    (FlattenLayer, PoolingLayer) using hasattr.
    """
    # Run forward + backward to get analytical gradients
    y_pred = net.forward(X)
    net.backward(y_pred, y)

    analytical = []
    numerical  = []

    for layer in net.layers:
        # Skip layers without learnable parameters (Flatten, Pooling)
        if not hasattr(layer, 'W'):
            continue

        for param, grad in [(layer.W, layer.dW), (layer.b, layer.db)]:
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                original = param[idx]

                # L(θ + ε)
                param[idx] = original + epsilon
                y_plus = net.forward(X)
                loss_plus = net.loss(y_plus, y)

                # L(θ - ε)
                param[idx] = original - epsilon
                y_minus = net.forward(X)
                loss_minus = net.loss(y_minus, y)

                # Restore original value
                param[idx] = original

                num_grad = (loss_plus - loss_minus) / (2 * epsilon)
                analytical.append(grad[idx])
                numerical.append(num_grad)

                it.iternext()

    analytical = np.array(analytical)
    numerical  = np.array(numerical)

    # Relative error
    numerator   = np.linalg.norm(analytical - numerical)
    denominator = np.linalg.norm(analytical) + np.linalg.norm(numerical)
    relative_error = numerator / (denominator + 1e-10)

    return relative_error, analytical, numerical


def test_gradient_checking():
    print("=" * 50)
    print("TEST 4 — Gradient checking")
    print("=" * 50)

    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = NeuralNetwork(
        layers=[
            Layer(2, 3, Sigmoid()),
            Layer(3, 1, Sigmoid())
        ],
        loss=MSE()
    )

    # Use Sigmoid for gradient check — ReLU can fail at z=0 due to non-differentiability
    relative_error, analytical, numerical = gradient_check(net, X, y)

    passed = relative_error < 1e-5
    status = "✅" if passed else "❌"
    print(f"  {status} Relative error: {relative_error:.2e} (threshold: 1e-5)")

    if not passed:
        print("\n  First 10 analytical vs numerical gradients:")
        for a, n in zip(analytical[:10], numerical[:10]):
            print(f"    analytical: {a:.6f}  |  numerical: {n:.6f}")

    print(f"\n  {'Gradient check passed — backprop is correct' if passed else 'Gradient check FAILED — there is a bug in backprop'}\n")
    return passed


# ══════════════════════════════════════════════
# TEST 5 — XOR training
# ══════════════════════════════════════════════

def test_xor():
    print("=" * 50)
    print("TEST 5 — XOR training")
    print("=" * 50)

    np.random.seed(0)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = NeuralNetwork(
        layers=[
            Layer(2, 4, ReLU()),
            Layer(4, 1, Sigmoid())
        ],
        loss=MSE()
    )

    losses = net.train(X, y, epochs=5000, lr=0.1)

    y_pred = net.forward(X)
    y_binary = (y_pred > 0.5).astype(int)

    print("\n  Predictions after training:")
    inputs = ["[0,0]", "[0,1]", "[1,0]", "[1,1]"]
    for i, inp in enumerate(inputs):
        print(f"    {inp} → pred: {y_pred[i][0]:.4f}  →  {y_binary[i][0]}  (true: {int(y[i][0])})")

    accuracy = np.mean(y_binary == y.astype(int))
    loss_final = losses[-1]

    passed_acc  = accuracy == 1.0
    passed_loss = loss_final < 0.01

    print(f"\n  {'✅' if passed_acc  else '❌'} Accuracy: {accuracy*100:.1f}% (expected 100%)")
    print(f"  {'✅' if passed_loss else '❌'} Final loss: {loss_final:.6f} (expected < 0.01)")

    all_passed = passed_acc and passed_loss
    print(f"\n  {'XOR test passed' if all_passed else 'XOR test FAILED'}\n")
    return all_passed

# ══════════════════════════════════════════════
# TEST 6 — Softmax & Categorical Cross-Entropy (Forward)
# ══════════════════════════════════════════════

def test_softmax_cce():
    print("=" * 50)
    print("TEST 6 — Softmax & Categorical Cross Entropy")
    print("=" * 50)

    softmax = Softmax()
    cce = CategoricalCrossEntropy()

    # Z logits para 2 ejemplos, 3 clases
    Z = np.array([
        [1.0, 2.0, 3.0], 
        [0.0, 0.0, 0.0]
    ])
    
    # Etiquetas reales (One-hot encoding)
    Y = np.array([
        [0.0, 0.0, 1.0], # El primero es de la clase 2
        [1.0, 0.0, 0.0]  # El segundo es de la clase 0
    ])

    all_passed = True

    # 1. Probar Softmax Forward
    A = softmax(Z)
    
    # Comprobar que las probabilidades suman 1 por fila
    sums = np.sum(A, axis=1)
    passed_sums = np.allclose(sums, np.array([1.0, 1.0]))
    print(f"  {'✅' if passed_sums else '❌'} Softmax sums to 1: {sums}")
    if not passed_sums: all_passed = False

    # Comprobar un valor conocido (softmax de [0,0,0] debe ser [0.333, 0.333, 0.333])
    passed_vals = np.allclose(A[1], np.array([1/3, 1/3, 1/3]))
    print(f"  {'✅' if passed_vals else '❌'} Softmax known values: {A[1]}")
    if not passed_vals: all_passed = False

    # 2. Probar CCE Forward
    loss = cce(A, Y)
    # L = - (1 * ln(0.665) + 1 * ln(0.333)) / 2 = (0.4076 + 1.0986) / 2 = 0.7531
    expected_loss = 0.7531
    passed_loss = np.isclose(loss, expected_loss, atol=1e-3)
    print(f"  {'✅' if passed_loss else '❌'} CCE forward loss: expected ~{expected_loss:.4f}, got {loss:.4f}")
    if not passed_loss: all_passed = False

    # 3. Probar gradiente combinado (A - Y)/N
    dZ = cce.derivative(A, Y)
    expected_dZ = (A - Y) / 2 # N=2
    passed_dz = np.allclose(dZ, expected_dZ)
    print(f"  {'✅' if passed_dz else '❌'} CCE combined derivative matches (A - Y)/N")
    if not passed_dz: all_passed = False

    print(f"\n  {'Softmax & CCE tests passed' if all_passed else 'Softmax & CCE tests FAILED'}\n")
    return all_passed

# ══════════════════════════════════════════════
# TEST 7 — Gradient Check for Softmax + CCE
# ══════════════════════════════════════════════

def test_softmax_cce_gradient():
    print("=" * 50)
    print("TEST 7 — Gradient checking (Softmax + CCE)")
    print("=" * 50)

    np.random.seed(42)

    # 3 ejemplos, 4 features
    X = np.random.randn(3, 4)
    # 3 ejemplos, 3 clases (one-hot)
    y = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    net = NeuralNetwork(
        layers=[
            Layer(4, 5, Tanh()),
            Layer(5, 3, Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )

    relative_error, analytical, numerical = gradient_check(net, X, y)

    passed = relative_error < 1e-5
    status = "✅" if passed else "❌"
    print(f"  {status} Relative error: {relative_error:.2e} (threshold: 1e-5)")

    if not passed:
        print("\n  ⚠️ Esto significa que el hack de Softmax+CCE está fallando.")
        for a, n in zip(analytical[:5], numerical[:5]):
            print(f"    analytical: {a:.6f}  |  numerical: {n:.6f}")

    print(f"\n  {'Softmax+CCE Gradient check passed!' if passed else 'Softmax+CCE Gradient check FAILED'}\n")
    return passed

# ══════════════════════════════════════════════
# TEST 8 — Minibatch logic & Multiclass Training
# ══════════════════════════════════════════════

def test_minibatches():
    print("=" * 50)
    print("TEST 8 — Minibatches & Multiclass Training")
    print("=" * 50)

    # 1. Probar la función de crear batches
    net = NeuralNetwork([], MSE()) # Red dummy solo para usar el método
    X_dummy = np.arange(10).reshape(10, 1) # 10 ejemplos
    y_dummy = np.arange(10).reshape(10, 1)
    
    X_batches, y_batches = net.create_batches(X_dummy, y_dummy, batch_size=3)
    
    all_passed = True
    
    # Deberían salir 4 batches: tamaños 3, 3, 3 y 1
    passed_len = len(X_batches) == 4
    passed_shapes = (X_batches[0].shape[0] == 3) and (X_batches[-1].shape[0] == 1)
    
    print(f"  {'✅' if passed_len else '❌'} Number of batches: expected 4, got {len(X_batches)}")
    print(f"  {'✅' if passed_shapes else '❌'} Batch shapes handle remainders correctly")
    
    if not (passed_len and passed_shapes): all_passed = False

    # 2. Probar entrenamiento real con minibatches y Softmax+CCE
    np.random.seed(99)
    # Dataset sintético: 100 ejemplos, 2 features, 3 clases
    X_train = np.random.randn(100, 2)
    # Asignamos clases simples basándonos en el cuadrante de las features
    y_train_idx = np.where(X_train[:, 0] > 0, (X_train[:, 1] > 0).astype(int), 2)
    
    # Convertir a One-Hot
    y_train = np.zeros((100, 3))
    y_train[np.arange(100), y_train_idx] = 1

    net_multi = NeuralNetwork(
        layers=[
            Layer(2, 8, ReLU()),
            Layer(8, 3, Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )

    # Entrenar por pocas épocas, la loss debe bajar consistentemente
    losses = net_multi.train(X_train, y_train, epochs=20, lr=0.1, batch_size=16)
    
    loss_initial = losses[0]
    loss_final = losses[-1]
    passed_training = loss_final < loss_initial

    print(f"  {'✅' if passed_training else '❌'} Minibatch training loss decreased: {loss_initial:.4f} -> {loss_final:.4f}")
    if not passed_training: all_passed = False

    print(f"\n  {'Minibatch tests passed' if all_passed else 'Minibatch tests FAILED'}\n")
    return all_passed
# ══════════════════════════════════════════════
# TEST 9 — FlattenLayer
# ══════════════════════════════════════════════

def test_flatten_layer():
    print("=" * 50)
    print("TEST 9 — FlattenLayer")
    print("=" * 50)

    all_passed = True
    flatten = FlattenLayer()

    # Input: batch of 2 volumes (2 channels, 3x3)
    X = np.random.randn(2, 3, 3, 3)  # (N=2, C=3, H=3, W=3)

    # --- Forward ---
    out = flatten.forward(X)

    # Output shape should be (2, 3*3*3) = (2, 27)
    passed_shape = out.shape == (2, 27)
    print(f"  {'✅' if passed_shape else '❌'} Forward shape: expected (2, 27), got {out.shape}")
    if not passed_shape: all_passed = False

    # Input shape must be saved
    passed_cache = flatten.input_shape == (2, 3, 3, 3)
    print(f"  {'✅' if passed_cache else '❌'} input_shape cached: expected (2, 3, 3, 3), got {flatten.input_shape}")
    if not passed_cache: all_passed = False

    # Values must be preserved — flattening only changes shape
    passed_vals = np.allclose(out.reshape(2, 3, 3, 3), X)
    print(f"  {'✅' if passed_vals else '❌'} Forward values preserved after flatten")
    if not passed_vals: all_passed = False

    # --- Backward ---
    dA = np.random.randn(2, 27)
    dX = flatten.backward(dA)

    # Output shape must match original input shape
    passed_back_shape = dX.shape == (2, 3, 3, 3)
    print(f"  {'✅' if passed_back_shape else '❌'} Backward shape: expected (2, 3, 3, 3), got {dX.shape}")
    if not passed_back_shape: all_passed = False

    # Backward must be the exact inverse of forward — values unchanged
    passed_back_vals = np.allclose(dX.reshape(2, 27), dA)
    print(f"  {'✅' if passed_back_vals else '❌'} Backward values preserved after unflatten")
    if not passed_back_vals: all_passed = False

    print(f"\n  {'FlattenLayer tests passed' if all_passed else 'FlattenLayer tests FAILED'}\n")
    return all_passed


# ══════════════════════════════════════════════
# TEST 10 — PoolingLayer
# ══════════════════════════════════════════════

def test_pooling_layer():
    print("=" * 50)
    print("TEST 10 — PoolingLayer")
    print("=" * 50)

    all_passed = True
    pool = PoolingLayer(pool_size=2)

    # --- Forward shape ---
    X = np.random.randn(3, 4, 8, 8)  # (N=3, C=4, H=8, W=8)
    out = pool.forward(X)

    passed_shape = out.shape == (3, 4, 4, 4)
    print(f"  {'✅' if passed_shape else '❌'} Forward shape: expected (3, 4, 4, 4), got {out.shape}")
    if not passed_shape: all_passed = False

    # --- Forward values (known input) ---
    # Build a 1x1x4x4 input where we know the maxima
    X_known = np.array([[[[1, 3, 2, 0],
                           [4, 2, 1, 5],
                           [0, 1, 6, 2],
                           [3, 2, 1, 4]]]], dtype=float)  # (1, 1, 4, 4)

    out_known = PoolingLayer(pool_size=2).forward(X_known)

    # Region (0,0): [1,3,4,2] → max=4
    # Region (0,1): [2,0,1,5] → max=5
    # Region (1,0): [0,1,3,2] → max=3
    # Region (1,1): [6,2,1,4] → max=6
    expected = np.array([[[[4, 5],
                            [3, 6]]]], dtype=float)

    passed_vals = np.allclose(out_known, expected)
    print(f"  {'✅' if passed_vals else '❌'} Forward known values: expected [[4,5],[3,6]], got {out_known[0,0]}")
    if not passed_vals: all_passed = False

    # --- Mask is saved and has correct shape ---
    pool2 = PoolingLayer(pool_size=2)
    X2 = np.random.randn(2, 2, 4, 4)
    pool2.forward(X2)

    passed_mask_shape = pool2.mask.shape == (2, 2, 4, 4)
    print(f"  {'✅' if passed_mask_shape else '❌'} Mask shape: expected (2, 2, 4, 4), got {pool2.mask.shape}")
    if not passed_mask_shape: all_passed = False

    # Mask must sum to exactly 1 per 2x2 region (one max per region)
    # Sum over each 2x2 block should be 1
    mask_reshaped = pool2.mask.reshape(2, 2, 2, 2, 2, 2)
    mask_sums = mask_reshaped.sum(axis=(3, 5))
    passed_mask_sums = np.all(mask_sums >= 1)  # at least 1 per region (ties allowed)
    print(f"  {'✅' if passed_mask_sums else '❌'} Mask has at least one active position per region")
    if not passed_mask_sums: all_passed = False

    # --- Backward shape ---
    dA = np.random.randn(2, 2, 2, 2)
    dX = pool2.backward(dA)

    passed_back_shape = dX.shape == (2, 2, 4, 4)
    print(f"  {'✅' if passed_back_shape else '❌'} Backward shape: expected (2, 2, 4, 4), got {dX.shape}")
    if not passed_back_shape: all_passed = False

    # Gradient only flows through max positions
    # Non-max positions in dX must be zero
    non_max_positions = (pool2.mask == 0)
    passed_zero_grad = np.all(dX[non_max_positions] == 0)
    print(f"  {'✅' if passed_zero_grad else '❌'} Non-max positions receive zero gradient")
    if not passed_zero_grad: all_passed = False

    # --- Dimension validation ---
    try:
        pool.forward(np.random.randn(2, 3, 5, 5))  # 5 not divisible by 2
        print(f"  ❌ Dimension validation: should have raised ValueError for non-divisible H/W")
        all_passed = False
    except ValueError:
        print(f"  ✅ Dimension validation: correctly raises ValueError for non-divisible H/W")

    try:
        pool.forward(np.random.randn(2, 3, 4))  # 3D input instead of 4D
        print(f"  ❌ Dimension validation: should have raised ValueError for 3D input")
        all_passed = False
    except ValueError:
        print(f"  ✅ Dimension validation: correctly raises ValueError for non-4D input")

    print(f"\n  {'PoolingLayer tests passed' if all_passed else 'PoolingLayer tests FAILED'}\n")
    return all_passed


# ══════════════════════════════════════════════
# TEST 11 — ConvLayer shapes
# ══════════════════════════════════════════════

def test_conv_layer_shapes():
    print("=" * 50)
    print("TEST 11 — ConvLayer shapes")
    print("=" * 50)

    all_passed = True

    # --- Weight initialization shapes ---
    conv = ConvLayer(n_inputs_channels=1, n_filters=8, filter_size=3)

    passed_W = conv.W.shape == (8, 1, 3, 3)
    passed_b = conv.b.shape == (8, 1)
    print(f"  {'✅' if passed_W else '❌'} W shape: expected (8, 1, 3, 3), got {conv.W.shape}")
    print(f"  {'✅' if passed_b else '❌'} b shape: expected (8, 1), got {conv.b.shape}")
    if not passed_W: all_passed = False
    if not passed_b: all_passed = False

    # --- Forward shape, no padding, stride 1 ---
    # H_out = (28 - 3 + 0) / 1 + 1 = 26
    X = np.random.randn(4, 1, 28, 28)
    out = conv.forward(X)

    passed_out = out.shape == (4, 8, 26, 26)
    print(f"  {'✅' if passed_out else '❌'} Forward shape (28x28, f=3, no pad): expected (4, 8, 26, 26), got {out.shape}")
    if not passed_out: all_passed = False

    # --- Forward shape, with padding ---
    conv_pad = ConvLayer(n_inputs_channels=1, n_filters=4, filter_size=3, padding=1)
    out_pad = conv_pad.forward(X)

    passed_pad = out_pad.shape == (4, 4, 28, 28)
    print(f"  {'✅' if passed_pad else '❌'} Forward shape (28x28, f=3, pad=1): expected (4, 4, 28, 28), got {out_pad.shape}")
    if not passed_pad: all_passed = False

    # --- Forward shape, stride 2 ---
    conv_s2 = ConvLayer(n_inputs_channels=1, n_filters=4, filter_size=3, stride=2)
    out_s2 = conv_s2.forward(X)
    # H_out = (28 - 3) / 2 + 1 = 13
    passed_s2 = out_s2.shape == (4, 4, 13, 13)
    print(f"  {'✅' if passed_s2 else '❌'} Forward shape (28x28, f=3, stride=2): expected (4, 4, 13, 13), got {out_s2.shape}")
    if not passed_s2: all_passed = False

    # --- X_col and X_cache are saved ---
    passed_cache = conv.X_cache is not None and conv.X_col is not None
    print(f"  {'✅' if passed_cache else '❌'} X_cache and X_col saved during forward")
    if not passed_cache: all_passed = False

    # --- Backward shape ---
    dZ = np.random.randn(*out.shape)
    dX = conv.backward(dZ)

    passed_dX = dX.shape == X.shape
    passed_dW = conv.dW.shape == conv.W.shape
    passed_db = conv.db.shape == conv.b.shape
    print(f"  {'✅' if passed_dX else '❌'} dX shape: expected {X.shape}, got {dX.shape}")
    print(f"  {'✅' if passed_dW else '❌'} dW shape: expected {conv.W.shape}, got {conv.dW.shape}")
    print(f"  {'✅' if passed_db else '❌'} db shape: expected {conv.b.shape}, got {conv.db.shape}")
    if not passed_dX: all_passed = False
    if not passed_dW: all_passed = False
    if not passed_db: all_passed = False

    # --- Multi-channel input ---
    conv_rgb = ConvLayer(n_inputs_channels=3, n_filters=16, filter_size=5)
    X_rgb = np.random.randn(2, 3, 32, 32)
    out_rgb = conv_rgb.forward(X_rgb)
    # H_out = 32 - 5 + 1 = 28
    passed_rgb = out_rgb.shape == (2, 16, 28, 28)
    print(f"  {'✅' if passed_rgb else '❌'} Multi-channel (3ch input, 16 filters, 5x5): expected (2, 16, 28, 28), got {out_rgb.shape}")
    if not passed_rgb: all_passed = False

    print(f"\n  {'ConvLayer shape tests passed' if all_passed else 'ConvLayer shape tests FAILED'}\n")
    return all_passed


# ══════════════════════════════════════════════
# TEST 12 — ConvLayer known values
# ══════════════════════════════════════════════

def test_conv_known_values():
    print("=" * 50)
    print("TEST 12 — ConvLayer known values")
    print("=" * 50)

    all_passed = True

    # Single example, single channel, 4x4 input
    # Single filter, 2x2, all ones → output = sum of each 2x2 patch
    X = np.array([[[[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]]]], dtype=float)  # (1, 1, 4, 4)

    conv = ConvLayer(n_inputs_channels=1, n_filters=1, filter_size=2, stride=1, padding=0)
    conv.W = np.ones((1, 1, 2, 2))   # all-ones filter
    conv.b = np.zeros((1, 1))

    out = conv.forward(X)

    # Each output position is the sum of the corresponding 2x2 patch:
    # (0,0): 1+2+5+6=14  (0,1): 2+3+6+7=18  (0,2): 3+4+7+8=22
    # (1,0): 5+6+9+10=30 (1,1): 6+7+10+11=34 (1,2): 7+8+11+12=38
    # (2,0): 9+10+13+14=46 (2,1): 10+11+14+15=50 (2,2): 11+12+15+16=54
    expected = np.array([[[[14, 18, 22],
                            [30, 34, 38],
                            [46, 50, 54]]]], dtype=float)

    passed_vals = np.allclose(out, expected)
    print(f"  {'✅' if passed_vals else '❌'} All-ones filter forward values correct")
    if not passed_vals:
        all_passed = False
        print(f"    Expected:\n{expected[0,0]}")
        print(f"    Got:\n{out[0,0]}")

    # --- Bias is added correctly ---
    conv.b = np.array([[10.0]])
    out_bias = conv.forward(X)
    passed_bias = np.allclose(out_bias, expected + 10.0)
    print(f"  {'✅' if passed_bias else '❌'} Bias added correctly to all output positions")
    if not passed_bias: all_passed = False

    print(f"\n  {'ConvLayer value tests passed' if all_passed else 'ConvLayer value tests FAILED'}\n")
    return all_passed


# ══════════════════════════════════════════════
# TEST 13 — ConvLayer gradient checking
# ══════════════════════════════════════════════

def test_conv_gradient_check():
    print("=" * 50)
    print("TEST 13 — ConvLayer gradient checking")
    print("=" * 50)

    np.random.seed(7)

    # Small inputs to keep gradient check fast
    # (N=2, C=1, H=6, W=6) → ConvLayer(1, 3, 3) → (2, 3, 4, 4)
    X = np.random.randn(2, 1, 6, 6)
    # Flatten to (2, 48) → Dense → (2, 4) one-hot
    y = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)

    net = NeuralNetwork(
        layers=[
            ConvLayer(n_inputs_channels=1, n_filters=3, filter_size=3),
            FlattenLayer(),
            Layer(48, 4, Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )

    relative_error, analytical, numerical = gradient_check(net, X, y)

    passed = relative_error < 1e-4  # slightly looser for conv due to numerical precision
    status = "✅" if passed else "❌"
    print(f"  {status} Relative error: {relative_error:.2e} (threshold: 1e-4)")

    if not passed:
        print("\n  First 10 analytical vs numerical gradients:")
        for a, n in zip(analytical[:10], numerical[:10]):
            print(f"    analytical: {a:.6f}  |  numerical: {n:.6f}")

    print(f"\n  {'ConvLayer gradient check passed' if passed else 'ConvLayer gradient check FAILED — bug in backward'}\n")
    return passed


# ══════════════════════════════════════════════
# TEST 14 — CNN end-to-end forward shapes
# ══════════════════════════════════════════════

def test_cnn_end_to_end_shapes():
    print("=" * 50)
    print("TEST 14 — CNN end-to-end forward shapes")
    print("=" * 50)

    all_passed = True
    np.random.seed(0)

    # Simulated MNIST batch: 8 images, 1 channel, 28x28
    X = np.random.randn(8, 1, 28, 28)
    y = np.zeros((8, 10))
    y[np.arange(8), np.arange(8)] = 1  # one-hot

    net = NeuralNetwork(
        layers=[
            ConvLayer(1, 8, 3,padding=1),         # (8, 8, 26, 26)
            PoolingLayer(2),             # (8, 8, 13, 13)
            ConvLayer(8, 16, 3,padding=1),         # (8, 16, 11, 11)
            PoolingLayer(2),             # (8, 16, 5, 5)
            FlattenLayer(),              # (8, 400)
            Layer(784, 64, ReLU()),      # (8, 64)
            Layer(64, 10, Softmax()),    # (8, 10)
        ],
        loss=CategoricalCrossEntropy()
    )

    expected_intermediate = [
        (8, 8, 28, 28),
        (8, 8, 14, 14),
        (8, 16, 14, 14),
        (8, 16, 7, 7),
        (8, 784),
        (8, 64),
        (8, 10),
    ]

    # Run forward and check output at each layer
    A = X
    for i, layer in enumerate(net.layers):
        A = layer.forward(A)
        expected = expected_intermediate[i]
        passed = A.shape == expected
        layer_name = type(layer).__name__
        print(f"  {'✅' if passed else '❌'} Layer {i} ({layer_name}): expected {expected}, got {A.shape}")
        if not passed: all_passed = False

    # Output probabilities must sum to 1 per example
    sums = np.sum(A, axis=1)
    passed_sums = np.allclose(sums, np.ones(8), atol=1e-5)
    print(f"  {'✅' if passed_sums else '❌'} Final Softmax output sums to 1 per example")
    if not passed_sums: all_passed = False

    print(f"\n  {'CNN end-to-end shape test passed' if all_passed else 'CNN end-to-end shape test FAILED'}\n")
    return all_passed


# ══════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════

if __name__ == "__main__":
    results = {
        # ── Dense network ──────────────────────────────
        "Activations":              test_activations(),
        "Losses":                   test_losses(),
        "Forward shapes (dense)":   test_forward_shapes(),
        "Gradient check (MSE)":     test_gradient_checking(),
        "XOR training":             test_xor(),
        # ── Softmax + CCE ──────────────────────────────
        "Softmax & CCE logic":      test_softmax_cce(),
        "Gradient check (CCE)":     test_softmax_cce_gradient(),
        # ── Mini-batches ───────────────────────────────
        "Minibatch training":       test_minibatches(),
        # ── CNN layers ─────────────────────────────────
        "FlattenLayer":             test_flatten_layer(),
        "PoolingLayer":             test_pooling_layer(),
        "ConvLayer shapes":         test_conv_layer_shapes(),
        "ConvLayer known values":   test_conv_known_values(),
        "Gradient check (Conv)":    test_conv_gradient_check(),
        "CNN end-to-end shapes":    test_cnn_end_to_end_shapes(),
    }

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        print(f"  {'✅' if passed else '❌'} {name}")

    total = len(results)
    passed_count = sum(results.values())
    print(f"\n  {passed_count}/{total} tests passed")