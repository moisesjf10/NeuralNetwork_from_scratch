import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os

# Import your architecture
from network import NeuralNetwork
from layers import Layer, FlattenLayer, PoolingLayer, ConvLayer
from activations import ReLU, Softmax
from losses import CategoricalCrossEntropy

# Get the directory where this script is located (experiments/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the output directory path (experiments/plots/)
output_dir = os.path.join(script_dir, "plots")

# Create the directory if it does not exist yet
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Output directory verified at: {output_dir}")


print("Instantiating architecture and loading weights...")
net = NeuralNetwork(
    layers=[
        ConvLayer(1, 8, 3, padding=1),
        PoolingLayer(2),           
        ConvLayer(8, 16, 3, padding=1),
        PoolingLayer(2),            
        FlattenLayer(),                  
        Layer(784, 64, ReLU()),      
        Layer(64, 10, Softmax()),   
    ],
    loss=CategoricalCrossEntropy()
)

# Make sure the name matches the file you saved during training
net.load_weights("cnn_mnist.pkl")

# Extract the first convolutional layer (index 0)
conv1 = net.layers[0]

#VISUALIZE LEARNED FILTERS (WEIGHTS)

# W shape is (F, C, f, f) -> (8, 1, 3, 3)
filters = conv1.W 

fig, axes = plt.subplots(1, 8, figsize=(15, 3))
fig.suptitle('Learned Filters in Conv1 (3x3 Matrices)', fontsize=16)

for i in range(8):
    # Select filter i, channel 0
    current_filter = filters[i, 0, :, :]
    ax = axes[i]
    
    # Use a divergent colormap where 0 is a neutral color
    # to clearly separate positive and negative weights
    cax = ax.imshow(current_filter, cmap='RdBu', vmin=-np.max(np.abs(current_filter)), vmax=np.max(np.abs(current_filter)))
    ax.axis('off')
    ax.set_title(f'Filter {i}')

plt.tight_layout()
# Save to the specific plots folder
filters_path = os.path.join(output_dir, "filtros_conv1.png")
plt.savefig(filters_path, dpi=300, bbox_inches='tight')
print(f"✅ Filters saved as: {filters_path}")
plt.close(fig)

#VISUALIZE FEATURE MAPS (ACTIVATIONS)

print("\nDownloading a test image from MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.values.astype(np.float32) / 255.0

# Take the image at index 0 (usually a handwritten 5)
original_image = X[0].reshape(1, 1, 28, 28)

# Pass the image ONLY through the first layer to see its reaction
feature_maps = conv1.forward(original_image) # Shape: (1, 8, 28, 28)

fig2, axes2 = plt.subplots(2, 4, figsize=(12, 6))
fig2.suptitle('Feature Maps (What the network sees)', fontsize=16)

for i in range(8):
    row = i // 4
    col = i % 4
    current_map = feature_maps[0, i, :, :]
    
    ax = axes2[row, col]
    # We use grayscale here because these are spatial activations (like an image)
    ax.imshow(current_map, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Filter {i} Activation')

fig2.tight_layout()
# Save to the specific plots folder
maps_path = os.path.join(output_dir, "feature_maps.png")
fig2.savefig(maps_path, dpi=300, bbox_inches='tight')
print(f"✅ Feature maps saved as: {maps_path}")
plt.close(fig2)

# Show the original image in a separate window for reference
fig3 = plt.figure(figsize=(3, 3))
plt.imshow(original_image[0, 0], cmap='gray')
plt.title("Original Image (Reference)")
plt.axis('off')

# Save to the specific plots folder
ref_path = os.path.join(output_dir, "original_image.png")
fig3.savefig(ref_path, dpi=300, bbox_inches='tight')
print(f"✅ Original image saved as: {ref_path}")
plt.close(fig3)