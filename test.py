import numpy as np

# ─────────────────────────────────────────────
# Paste your implementations here or import them
from activations import ReLU, Sigmoid, Tanh
from losses import MSE, BCE
from network import Layer, NeuralNetwork
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
    """
    # Run forward + backward to get analytical gradients
    y_pred = net.forward(X)
    net.backward(y_pred, y)

    analytical = []
    numerical  = []

    for layer in net.layers:
        for param, grad in [(layer.W, layer.dW), (layer.b, layer.db)]:
            # Flatten to iterate element by element
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
# RUN ALL TESTS
# ══════════════════════════════════════════════

if __name__ == "__main__":
    results = {
        "Activations":        test_activations(),
        "Losses":             test_losses(),
        "Forward shapes":     test_forward_shapes(),
        "Gradient checking":  test_gradient_checking(),
        "XOR training":       test_xor(),
    }

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        print(f"  {'✅' if passed else '❌'} {name}")