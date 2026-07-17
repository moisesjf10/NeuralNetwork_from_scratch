import os
import numpy as np
import time
from network import NeuralNetwork
from layers import EmbeddingLayer, PositionalEncoding, TransformerBlock, Layer
from activations import Softmax
from losses import CategoricalCrossEntropy
from utils import CharTokenizer
from optimizers import Adam

# =====================================================================
# 1. TEXT GENERATOR (Inference with Temperature)
# =====================================================================
def generate_text(model, tokenizer, seed, length=50, temperature=1.0):
    """
    Generates text autoregressively using the Transformer.
    """
    generated_text = seed
    
    for _ in range(length):
        # Encode current context
        indices = tokenizer.encode(generated_text)
        
        # We cap the context to the maximum sequence length the model was trained on
        # to prevent positional encoding out-of-bounds errors.
        if len(indices) > seq_length:
            indices = indices[-seq_length:]
            
        X = np.array([indices]) # Shape: (1, Time)
        
        # Forward pass
        predictions = model.forward(X)
        
        # Extract probabilities for the NEXT character (last time step)
        last_token_probs = predictions[0, -1, :]
        
        # Apply Temperature Scaling
        scaled_probs = np.power(last_token_probs + 1e-10, 1.0 / temperature)
        last_token_probs = scaled_probs / np.sum(scaled_probs)
        
        # Sample from the distribution
        next_id = np.random.choice(tokenizer.vocab_size, p=last_token_probs)
        next_char = tokenizer.decode([next_id])
        
        generated_text += next_char
        
    return generated_text


if __name__ == "__main__":
    print("=" * 60)
    print("Initializing Foundation Model (NumPy Transformer)")
    print("=" * 60)

    # Dataset Preparation
    raw_text = """the transformer architecture relies entirely on the self attention mechanism to draw global dependencies between input and output. 
it dispenses with recurrence and convolutions entirely. 
gradient descent optimizes the multi head matrices through backpropagation. 
linear algebra and calculus are the foundation of deep learning.""" * 4
    
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size} characters.")
    
    seq_length = 32
    print(f"Generating sliding windows of size {seq_length}...")
    X_onehot, Y_train = tokenizer.create_dataset(raw_text, seq_length=seq_length, stride=3)
    
    # Convert input from 3D One-Hot to 2D Integer IDs for the Embedding Layer
    X_train = np.argmax(X_onehot, axis=-1)
    
    # Architectural Hyperparameters
    d_model = 64      # Embedding dimension
    num_heads = 4     # Number of attention heads (d_model must be divisible by this)
    d_ff = 256        # Feed-Forward expansion dimension (usually 4x d_model)
    
    print("\nAssembling Transformer Network...")
    print(f"- Dimension: {d_model}")
    print(f"- Attention Heads: {num_heads}")
    print(f"- Layers (Blocks): 2")
    
    # Network Assembly
    model = NeuralNetwork(
        layers=[
            # Input Representation
            EmbeddingLayer(vocab_size=vocab_size, embedding_dim=d_model),
            PositionalEncoding(max_seq_length=seq_length, embedding_dim=d_model),
            
            # stack of Transformer Blocks (Pre-LayerNorm)
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            
            # Output Projection
            Layer(n_inputs=d_model, n_neurons=vocab_size, activation=Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )
    # Load Checkpoints (Transfer Learning / Continuous Training)
    weight_file = "transformer_weights.pkl"
    if os.path.exists(weight_file):
        print(f"\n[INFO] Weights found. Loading {weight_file}...")
        model.load_weights(weight_file)
    else:
        print("\n[INFO] No previous weights found. Starting with Xavier/He initialization.")

    # Cold Inference Test
    print("\n--- Generated Text BEFORE Training (Temperature: 1.0) ---")
    seed_text = "the transformer "
    print(generate_text(model, tokenizer, seed_text, length=60, temperature=1.0))
    print("-" * 60)
    
    # Training Loop
    print("\nStarting optimization loop...")
    start_time = time.time()
    
    optimizer = Adam(lr=0.001)
    # Note: Transformers in pure NumPy are computationally heavy. 
    # We use a lower learning rate (0.01) to keep the LayerNorm and Softmax gradients stable.
    loss_history = model.train(X_train, Y_train, epochs=50, lr=0.001, batch_size=16, optimizer=optimizer)
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")
    
    # Trained Inference Tests
    print("\n--- Generated Text AFTER Training (Strict, Temp: 0.2) ---")
    print(generate_text(model, tokenizer, seed_text, length=80, temperature=0.2))
    
    print("\n--- Generated Text AFTER Training (Creative, Temp: 0.8) ---")
    print(generate_text(model, tokenizer, seed_text, length=80, temperature=0.8))
    print("-" * 60)
    
    print(f"\nInitial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")
    
    # Save the master weights
    model.save_weights("transformer_weights.pkl")