import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from network import NeuralNetwork
from layers import Layer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy
from optimizers import Adam


# ══════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════

def load_and_preprocess_mnist():
    print("Loading MNIST dataset...")
    mnist   = fetch_openml('mnist_784', version=1, parser='auto')
    X       = mnist.data.values.astype(np.float32) / 255.0
    y       = mnist.target.values.astype(np.int32).reshape(-1, 1)

    encoder  = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_accuracy(net, X, y_onehot):
    probs       = net.forward(X)
    predictions = np.argmax(probs, axis=1)
    trues       = np.argmax(y_onehot, axis=1)
    return np.mean(predictions == trues)


# ══════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════

def build_network():
    """Returns a fresh network with the same architecture and random seed."""
    return NeuralNetwork(
        layers=[
            Layer(784, 128, ReLU()),
            Layer(128, 64,  ReLU()),
            Layer(64,  10,  Softmax()),
        ],
        loss=CategoricalCrossEntropy()
    )


def run_experiment(name, net, optimizer, X_train, y_train, X_test, y_test,
                   epochs, batch_size, lr):
    print(f"\n{'─' * 40}")
    print(f"  {name}")
    print(f"{'─' * 40}")

    start  = time.time()
    losses = net.train(
        X=X_train, y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer=optimizer
    )
    elapsed = time.time() - start

    train_acc = evaluate_accuracy(net, X_train, y_train)
    test_acc  = evaluate_accuracy(net, X_test,  y_test)

    print(f"\n  Train accuracy : {train_acc * 100:.3f}%")
    print(f"  Test  accuracy : {test_acc  * 100:.3f}%")
    print(f"  Time           : {elapsed:.1f}s")

    return losses, train_acc, test_acc, elapsed


# ══════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════

def plot_comparison(losses_sgd, losses_adam, results, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SGD vs Adam — MNIST Dense Network', fontsize=14)

    # ── Loss curves ────────────────────────────
    ax = axes[0]
    ax.plot(losses_sgd,  label='SGD',  color='steelblue', linewidth=2)
    ax.plot(losses_adam, label='Adam', color='coral',     linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (CCE)')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Accuracy bar chart ─────────────────────
    ax = axes[1]
    labels     = ['SGD\nTrain', 'SGD\nTest', 'Adam\nTrain', 'Adam\nTest']
    values     = [
        results['sgd']['train_acc']  * 100,
        results['sgd']['test_acc']   * 100,
        results['adam']['train_acc'] * 100,
        results['adam']['test_acc']  * 100,
    ]
    colors = ['steelblue', 'lightsteelblue', 'coral', 'lightsalmon']
    bars   = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)

    ax.set_ylim(min(values) - 2, 101)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final Accuracy')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved → {path}")

    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == '__main__':

    # ── Config ─────────────────────────────────
    EPOCHS     = 30
    BATCH_SIZE = 128

    SGD_LR  = 0.1    # SGD needs a higher lr
    ADAM_LR = 0.001  # Adam works well with smaller lr

    # ── Data ───────────────────────────────────
    X_train, X_test, y_train, y_test = load_and_preprocess_mnist()

    # ── SGD ────────────────────────────────────
    # Set the same random seed so both networks start with identical weights
    np.random.seed(42)
    net_sgd = build_network()

    losses_sgd, train_acc_sgd, test_acc_sgd, time_sgd = run_experiment(
        name      = 'SGD  (lr=0.1)',
        net       = net_sgd,
        optimizer = None,          # None → falls back to NeuralNetwork.update()
        X_train   = X_train,
        y_train   = y_train,
        X_test    = X_test,
        y_test    = y_test,
        epochs    = EPOCHS,
        batch_size= BATCH_SIZE,
        lr        = SGD_LR
    )

    # ── Adam ───────────────────────────────────
    np.random.seed(42)
    net_adam = build_network()

    losses_adam, train_acc_adam, test_acc_adam, time_adam = run_experiment(
        name      = 'Adam (lr=0.001)',
        net       = net_adam,
        optimizer = Adam(lr=ADAM_LR),
        X_train   = X_train,
        y_train   = y_train,
        X_test    = X_test,
        y_test    = y_test,
        epochs    = EPOCHS,
        batch_size= BATCH_SIZE,
        lr        = ADAM_LR
    )

    # ── Summary ────────────────────────────────
    print(f"\n{'═' * 40}")
    print(f"  SUMMARY")
    print(f"{'═' * 40}")
    print(f"  {'':20s}  {'SGD':>10}  {'Adam':>10}")
    print(f"  {'Train accuracy':20s}  {train_acc_sgd*100:>9.3f}%  {train_acc_adam*100:>9.3f}%")
    print(f"  {'Test  accuracy':20s}  {test_acc_sgd*100:>9.3f}%  {test_acc_adam*100:>9.3f}%")
    print(f"  {'Final loss':20s}  {losses_sgd[-1]:>10.4f}  {losses_adam[-1]:>10.4f}")
    print(f"  {'Time (s)':20s}  {time_sgd:>10.1f}  {time_adam:>10.1f}")
    print(f"{'═' * 40}")

    # ── Plot ───────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path  = os.path.join(script_dir, 'plots', 'sgd_vs_adam.png')

    plot_comparison(
        losses_sgd  = losses_sgd,
        losses_adam = losses_adam,
        results     = {
            'sgd':  {'train_acc': train_acc_sgd,  'test_acc': test_acc_sgd},
            'adam': {'train_acc': train_acc_adam, 'test_acc': test_acc_adam},
        },
        path = plot_path
    )
