import numpy as np
import time
import os
from network import NeuralNetwork
from layers import EmbeddingLayer, PositionalEncoding, TransformerBlock, Layer
from activations import Softmax
from losses import CategoricalCrossEntropy
from utils import BPETokenizer
from optimizers import Adam

# =====================================================================
# TEXT GENERATOR (BPE Version)
# =====================================================================
def generate_text(model, tokenizer, seed, length=50, temperature=1.0, seq_length=32):
    """
    Generates text autoregressively using the BPE Transformer.
    """
    # We store the generated IDs instead of just the raw text
    generated_ids = tokenizer.encode(seed)
    
    for _ in range(length):
        # We cap the context to the maximum sequence length
        context_ids = generated_ids[-seq_length:] if len(generated_ids) > seq_length else generated_ids
            
        X = np.array([context_ids]) # Shape: (1, Time)
        
        # Forward pass
        predictions = model.forward(X)
        
        # Extract probabilities for the NEXT token (last time step)
        last_token_probs = predictions[0, -1, :]
        
        # Apply Temperature Scaling safely
        scaled_probs = np.power(last_token_probs + 1e-10, 1.0 / temperature)
        last_token_probs = scaled_probs / np.sum(scaled_probs)
        
        # Sample the next token ID from the distribution
        next_id = np.random.choice(tokenizer.vocab_size, p=last_token_probs)
        
        # Add the new ID to our logical sequence
        generated_ids.append(next_id)
        
    # Decode the complete sequence at the end
    return tokenizer.decode(generated_ids)

if __name__ == "__main__":
    print("=" * 60)
    print("Initializing Foundation Model (BPE + Adam Optimizer)")
    print("=" * 60)

    # 1. Dataset Preparation
    raw_text = """the transformer architecture relies entirely on the self attention mechanism to draw global dependencies between input and output. 
it dispenses with recurrence and convolutions entirely. 
gradient descent optimizes the multi head matrices through backpropagation. 
linear algebra and calculus are the foundation of deep learning.""" * 4
    
    tokenizer = BPETokenizer()
    target_vocab = 280  # 256 base bytes + 24 BPE compression rules
    
    # Train the BPE vocabulary
    tokenizer.train(raw_text, vocab_size=target_vocab)
    vocab_size = tokenizer.vocab_size
    
    seq_length = 32
    print(f"\nGenerating sliding windows of size {seq_length}...")
    X_train, Y_train = tokenizer.create_dataset(raw_text, seq_length=seq_length, stride=3)
    
    print(f"Dataset created: {X_train.shape[0]} windows. (Notice how it might be fewer than the Char-level model!)")
    
    # Architectural Hyperparameters
    d_model = 64      
    num_heads = 4     
    d_ff = 256        
    
    print("\nAssembling Transformer Network...")
    print(f"- Dimension: {d_model}")
    print(f"- Attention Heads: {num_heads}")
    print(f"- Layers (Blocks): 2")
    
    # Network Assembly
    model = NeuralNetwork(
        layers=[
            EmbeddingLayer(vocab_size=vocab_size, embedding_dim=d_model),
            PositionalEncoding(max_seq_length=seq_length, embedding_dim=d_model),
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff),
            Layer(n_inputs=d_model, n_neurons=vocab_size, activation=Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )
    
    weight_file = "transformer_bpe_weights.pkl"
    if os.path.exists(weight_file):
        print(f"\n[INFO] BPE Weights found. Loading {weight_file}...")
        model.load_weights(weight_file)
    else:
        print("\n[INFO] No previous BPE weights found. Starting fresh (Xavier/He).")
    
    # Cold Inference Test
    print("\n--- Generated Text BEFORE Training (Temperature: 1.0) ---")
    seed_text = "the transformer "
    print(generate_text(model, tokenizer, seed_text, length=30, temperature=1.0, seq_length=seq_length))
    print("-" * 60)
    
    # Training Loop using ADAM Optimizer
    print("\nStarting optimization loop with Adam...")
    start_time = time.time()
    
    # Initialize Adam
    optimizer = Adam(lr=0.001)
    
    # We give it 150 epochs because Adam converges fast and the task is now more abstract
    loss_history = model.train(X_train, Y_train, epochs=150, lr=0.001, batch_size=16, optimizer=optimizer)
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")
    
    # Trained Inference Tests
    print("\n--- Generated Text AFTER Training (Strict, Temp: 0.2) ---")
    # We generate 30 BPE tokens (equivalent to ~50-60 characters)
    print(generate_text(model, tokenizer, seed_text, length=100, temperature=0.2, seq_length=seq_length))
    
    print("\n--- Generated Text AFTER Training (Creative, Temp: 0.8) ---")
    print(generate_text(model, tokenizer, seed_text, length=100, temperature=0.8, seq_length=seq_length))
    print("-" * 60)
    
    print(f"\nInitial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")
    
    # Save the new BPE weights
    model.save_weights(weight_file)