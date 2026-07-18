import numpy as np
import time
import os

from network import NeuralNetwork
from layers import PatchEmbedding, PositionalEncoding, TransformerBlock, GlobalAveragePooling1D, Layer
from activations import Softmax
from losses import CategoricalCrossEntropy
from optimizers import Adam

try:
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
except ImportError:
    print("[ERROR] scikit-learn is missing. Install it with: pip install scikit-learn")
    exit()

# DATASET PREPARATION (MNIST 28x28 Images)
print("=" * 60)
print("Starting Training: Vision Transformer (ViT) on MNIST")
print("=" * 60)

print("Downloading/Loading MNIST dataset (this might take a minute)...")
# fetch_openml downloads the dataset the first time and caches it
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')

# MNIST data comes flattened (70000, 784). We reshape it to (70000, 28, 28)
X_raw = mnist.data.to_numpy().reshape(-1, 28, 28)
Y_raw = mnist.target.to_numpy().astype(int)


subset_size = 20000
X_raw = X_raw[:subset_size]
Y_raw = Y_raw[:subset_size]

# ViT expects images with channels: (Batch, Channels, Height, Width)
X_data = np.expand_dims(X_raw, axis=1) # New shape: (Batch, 1, 28, 28)

# Normalize pixels between 0 and 1 (MNIST max pixel value is 255)
X_data = X_data / 255.0 

# One-Hot Encoding for labels (10 classes: digits 0 to 9)
num_classes = 10
Y_onehot = np.zeros((Y_raw.size, num_classes))
Y_onehot[np.arange(Y_raw.size), Y_raw] = 1.0

# Split into training and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_onehot, test_size=0.2, random_state=42)

print(f"Dataset ready. Training images: {X_train.shape[0]}")
print(f"Input shape (Batch, C, H, W): {X_train.shape}")

# ViT GEOMETRIC HYPERPARAMETERS
patch_size = 4
in_channels = 1
num_patches = (28 // patch_size) * (28 // patch_size) # 16 tokens

d_model = 64      # Dimension of the embedding space
num_heads = 4     # Number of attention heads
d_ff = 128        # Feed-forward network expansion

print(f"\nImage Geometry:")
print(f"- Original size: 28x28 pixels")
print(f"- Patch size: {patch_size}x{patch_size}")
print(f"- Resulting sequence length: {num_patches} tokens per image")

# ViT ARCHITECTURE ASSEMBLY
print("\nAssembling the neural network...")

vit_model = NeuralNetwork(
    layers=[
        # Visual Extraction
        PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=d_model),
        
        # Spatial Context Injection
        PositionalEncoding(max_seq_length=num_patches, embedding_dim=d_model),
        
        # The Reasoning Engine
        TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
        
        # Global Consensus
        GlobalAveragePooling1D(),
        
        # Final Classifier
        Layer(n_inputs=d_model, n_neurons=num_classes, activation=Softmax())
    ],
    loss=CategoricalCrossEntropy()
)

# TRAINING LOOP
weight_file = "vit_mnist_weights.pkl"

if os.path.exists(weight_file):
    print(f"\n[INFO] Weights found. Loading {weight_file}...")
    vit_model.load_weights(weight_file)
    loss_history = [0.1] 
else:
    start_time = time.time()

    optimizer = Adam(lr=0.001)
    loss_history = vit_model.train(X_train, Y_train, epochs=25, lr=0.001, batch_size=32, optimizer=optimizer)

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")
    
    vit_model.save_weights(weight_file)
    print(f"[INFO] Weights saved successfully in {weight_file}")

# EVALUATION (ACCURACY TEST)
print("\nEvaluating accuracy on unseen data (Test Set)...")
predictions = vit_model.forward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(Y_test, axis=1)

accuracy = np.mean(predicted_classes == true_classes) * 100
print(f"Vision Transformer Accuracy: {accuracy:.2f}%")
print("=" * 60)


# =====================================================================
# GRAPHICAL VISUALIZATION (Saving to disk)
# =====================================================================
import matplotlib.pyplot as plt

print("\nGenerating and saving visualizations to disk...")
plot_dir = os.path.join("experiments", "plots")
os.makedirs(plot_dir, exist_ok=True)

# Learning Curve
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label="Training Loss", color="#2563eb", linewidth=2, marker='o')
plt.title("Vision Transformer Learning Curve", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "vit_01_learning_curve.png")) 
plt.close()

# Prediction Grid
num_samples = 9
indices = np.random.choice(len(X_test), num_samples, replace=False)

plt.figure(figsize=(10, 10))
plt.suptitle("ViT Predictions (Green = Correct, Red = Error)", fontsize=16, y=0.95)

for i, idx in enumerate(indices):
    img = X_test[idx].reshape(28, 28)
    true_label = true_classes[idx]
    pred_label = predicted_classes[idx]
    
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray')
    color = '#16a34a' if true_label == pred_label else '#dc2626'
    plt.title(f"Pred: {pred_label} | Real: {true_label}", color=color, fontsize=14, fontweight='bold')
    plt.axis('off')
    
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "vit_02_predictions.png")) 
plt.close()

# =====================================================================
# ATTENTION MAPS (Saving to disk)
# =====================================================================
print("Generating Attention Map...")

idx = np.random.choice(len(X_test))
img = X_test[idx:idx+1]
real_label = true_classes[idx]
pred_label = predicted_classes[idx]

_ = vit_model.forward(img)

transformer_block = vit_model.layers[2]
mha_layer = getattr(transformer_block, 'mha', getattr(transformer_block, 'attention', None))

if mha_layer is not None and 'A' in mha_layer.cache:
    A = mha_layer.cache['A']
    attn_heads_mean = np.mean(A[0], axis=0)
    patch_importance = np.mean(attn_heads_mean, axis=0)
    
    grid_size = 28 // patch_size
    attn_map_2d = patch_importance.reshape(grid_size, grid_size)
    
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"ViT Attention Analysis | Pred: {pred_label} (Real: {real_label})", fontsize=16)
    
    plt.subplot(1, 3, 1)
    plt.imshow(img[0, 0], cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(attn_map_2d, cmap='inferno')
    plt.title(f"Patch Attention ({grid_size}x{grid_size})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img[0, 0], cmap='gray')
    plt.imshow(attn_map_2d, cmap='jet', alpha=0.5, extent=[0, 28, 28, 0], interpolation='bilinear')
    plt.title("Smoothed Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "vit_03_attention_map.png")) 
    plt.close()
    print("[INFO] Images saved: 'vit_01_learning_curve.png', 'vit_02_predictions.png', 'vit_03_attention_map.png'")