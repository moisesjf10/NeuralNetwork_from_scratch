import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from network import NeuralNetwork
from layers import Layer, FlattenLayer, PoolingLayer, ConvLayer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy

# ══════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════

CLASS_NAMES = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

EPOCHS      = 20
BATCH_SIZE  = 64
LR          = 0.01

script_dir  = os.path.dirname(os.path.abspath(__file__))
plots_dir   = os.path.join(script_dir, 'plots', 'cifar10')
models_dir  = os.path.join(script_dir, '..', 'models')

os.makedirs(plots_dir,  exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


# ══════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════

def load_cifar10_sklearn():
    """
    Load CIFAR-10 via sklearn. Downloads automatically on first run (~150MB).
    Images come as flat arrays of 3072 values (3x32x32).
    Pixel values are normalized to [0, 1].
    Labels are one-hot encoded.
    """
    from sklearn.datasets import fetch_openml

    print("Loading CIFAR-10 (this may take a few minutes on first run)...")
    dataset = fetch_openml('CIFAR_10', version=1, parser='auto')

    X = dataset.data.values.astype(np.float32) / 255.0      # (60000, 3072)
    y = dataset.target.values.astype(np.int32).reshape(-1, 1)

    # Reshape to (N, C, H, W) — NCHW format expected by ConvLayer
    X = X.reshape(-1, 3, 32, 32)                             # (60000, 3, 32, 32)

    # One-hot encode labels
    encoder  = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)                      # (60000, 10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


def load_cifar10_official(data_dir):
    """
    Load CIFAR-10 from official pickle files.
    Download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    Extract and pass the path to the cifar-10-batches-py folder.
    """
    def load_batch(filepath):
        with open(filepath, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        X = batch[b'data'].astype(np.float32) / 255.0   # (10000, 3072)
        y = np.array(batch[b'labels'])
        X = X.reshape(-1, 3, 32, 32)
        return X, y

    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        X_b, y_b = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
        X_train_list.append(X_b)
        y_train_list.append(y_b)

    X_train = np.concatenate(X_train_list)               # (50000, 3, 32, 32)
    y_train = np.concatenate(y_train_list)               # (50000,)

    X_test, y_test_raw = load_batch(os.path.join(data_dir, 'test_batch'))

    encoder       = OneHotEncoder(sparse_output=False)
    y_train_oh    = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test_raw.reshape(-1, 1))

    print(f"  X_train: {X_train.shape}  y_train: {y_train_oh.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test_onehot.shape}")

    return X_train, X_test, y_train_oh, y_test_onehot


# ══════════════════════════════════════════════
# 2. EVALUATION
# ══════════════════════════════════════════════

def evaluate_accuracy(net, X, y_onehot, batch_size=256):
    """
    Evaluate accuracy in batches to avoid memory issues with large datasets.
    """
    n        = X.shape[0]
    correct  = 0

    for i in range(0, n, batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y_onehot[i:i + batch_size]

        probs       = net.forward(X_batch)
        predictions = np.argmax(probs, axis=1)
        trues       = np.argmax(y_batch, axis=1)
        correct    += np.sum(predictions == trues)

    return correct / n


# ══════════════════════════════════════════════
# 3. PLOTS
# ══════════════════════════════════════════════

def plot_loss_curve(losses, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(losses, color='steelblue', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (CCE)')
    ax.set_title('Training Loss — CIFAR-10 CNN')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Loss curve saved → {path}")


def plot_predictions(net, X_test, y_test, n=16, path=None):
    """
    Show n random test images with their true label, predicted label,
    and the confidence of the prediction.
    Correct predictions have a green title, wrong ones red.
    """
    indices = np.random.choice(X_test.shape[0], n, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    probs       = net.forward(X_sample)
    predictions = np.argmax(probs, axis=1)
    trues       = np.argmax(y_sample, axis=1)
    confidences = np.max(probs, axis=1)

    cols = 8
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    fig.suptitle('CIFAR-10 Predictions', fontsize=14, y=1.01)

    for i, ax in enumerate(axes.flat):
        # CIFAR-10 images are (3, 32, 32) → transpose to (32, 32, 3) for imshow
        img = X_sample[i].transpose(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')

        pred  = CLASS_NAMES[predictions[i]]
        true  = CLASS_NAMES[trues[i]]
        conf  = confidences[i] * 100
        color = 'green' if predictions[i] == trues[i] else 'red'

        ax.set_title(f'Pred: {pred}\nTrue: {true}\n{conf:.0f}%',
                     fontsize=7, color=color)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Predictions saved → {path}")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(net, X_test, y_test, path=None):
    """
    Compute and plot the 10x10 confusion matrix.
    Each row is the true class, each column the predicted class.
    The diagonal is where the network is correct.
    """
    probs       = net.forward(X_test)
    predictions = np.argmax(probs, axis=1)
    trues       = np.argmax(y_test,  axis=1)

    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(trues, predictions):
        cm[t, p] += 1

    # Normalize by row so each cell shows the fraction of that true class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — CIFAR-10 (normalized)')

    # Annotate each cell with the raw count
    for i in range(10):
        for j in range(10):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=7, color=color)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Confusion matrix saved → {path}")
    plt.show()
    plt.close(fig)


def plot_per_class_accuracy(net, X_test, y_test, path=None):
    """
    Bar chart showing accuracy per class.
    Reveals which categories are hardest for the network.
    """
    probs       = net.forward(X_test)
    predictions = np.argmax(probs, axis=1)
    trues       = np.argmax(y_test,  axis=1)

    accs = []
    for c in range(10):
        mask = (trues == c)
        acc  = np.mean(predictions[mask] == trues[mask])
        accs.append(acc)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(CLASS_NAMES, accs, color='steelblue', edgecolor='white')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-class Accuracy — CIFAR-10')
    ax.axhline(np.mean(accs), color='red', linestyle='--',
               linewidth=1.2, label=f'Mean: {np.mean(accs):.2%}')
    ax.legend()

    # Annotate bars with the value
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Per-class accuracy saved → {path}")
    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════

if __name__ == '__main__':

    # ── Load data ──────────────────────────────
    # If you have the official CIFAR-10 pickle files, use:
    #   X_train, X_test, y_train, y_test = load_cifar10_official('/path/to/cifar-10-batches-py')
    # Otherwise use sklearn (slower download, same result):
    X_train, X_test, y_train, y_test = load_cifar10_sklearn()

    # ── Build network ──────────────────────────
    # Architecture:
    #   Input: (N, 3, 32, 32)
    #   Conv(3→32,  3x3, p=1) → (N, 32, 32, 32)
    #   MaxPool(2)             → (N, 32, 16, 16)
    #   Conv(32→64, 3x3, p=1) → (N, 64, 16, 16)
    #   MaxPool(2)             → (N, 64,  8,  8)
    #   Conv(64→128,3x3, p=1) → (N, 128, 8,  8)
    #   MaxPool(2)             → (N, 128, 4,  4)
    #   Flatten                → (N, 2048)
    #   Dense(2048→256, ReLU)  → (N, 256)
    #   Dense(256→10, Softmax) → (N, 10)
    print("\nBuilding network...")
    net = NeuralNetwork(
        layers=[
            ConvLayer(3,   32,  3, padding=1),
            PoolingLayer(2),
            ConvLayer(32,  64,  3, padding=1),
            PoolingLayer(2),
            ConvLayer(64,  128, 3, padding=1),
            PoolingLayer(2),
            FlattenLayer(),
            Layer(2048, 256, ReLU()),
            Layer(256,  10,  Softmax()),
        ],
        loss=CategoricalCrossEntropy()
    )

    # ── Train ──────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  lr={LR}\n")
    losses = net.train(
        X=X_train, y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    # ── Save weights ───────────────────────────
    weights_path = os.path.join(models_dir, 'cifar10_cnn.npz')
    net.save_weights(weights_path)
    print(f"\nWeights saved → {weights_path}")

    # ── Evaluate ───────────────────────────────
    print("\nEvaluating...")
    train_acc = evaluate_accuracy(net, X_train, y_train)
    test_acc  = evaluate_accuracy(net, X_test,  y_test)
    print("-" * 35)
    print(f"  Train accuracy : {train_acc * 100:.2f}%")
    print(f"  Test  accuracy : {test_acc  * 100:.2f}%")
    print("-" * 35)

    # ── Plots ──────────────────────────────────
    print("\nGenerating plots...")

    plot_loss_curve(
        losses,
        path=os.path.join(plots_dir, 'loss_curve.png')
    )

    plot_predictions(
        net, X_test, y_test, n=16,
        path=os.path.join(plots_dir, 'predictions.png')
    )

    plot_confusion_matrix(
        net, X_test[:2000], y_test[:2000],
        path=os.path.join(plots_dir, 'confusion_matrix.png')
    )

    plot_per_class_accuracy(
        net, X_test[:2000], y_test[:2000],
        path=os.path.join(plots_dir, 'per_class_accuracy.png')
    )

    print("\nDone.")
