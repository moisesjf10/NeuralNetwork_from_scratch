import numpy as np
from network import NeuralNetwork
from layers import Layer, RNNLayer, EmbeddingLayer
from activations import Softmax
from losses import CategoricalCrossEntropy
from utils import CharTokenizer


def generate_text(model, tokenizer, seed, length=50):
    """
    Takes an initial text (seed) and predicts the future character by character.
    """
    generated_text = seed
    
    for _ in range(length):
        # Encode the text generated so far
        idx = tokenizer.encode(generated_text)
        #now we use a embedding layer
        #one_hot_matrix = tokenizer.to_one_hot(indices)
        
        # Add the Batch dimension: (1, Time, Vocabulary)
        X = np.array([idx])
        
        # Pass the sequence through the neural network
        predictions = model.forward(X)
        
        # Extract the probabilities of the last predicted character
        # predictions.shape is (1, Time, Vocabulary)
        last_char_probs = predictions[0, -1, :]

        temperature = 1.0  # Adjust this value to control randomness
        scaled_probs = np.power(last_char_probs + 1e-10, 1.0 / temperature)
        
        #stability trick: ensure probabilities sum to exactly 1.0
        last_char_probs = scaled_probs / np.sum(scaled_probs)
        
        # Statistical Sampling
        # Instead of always taking the character with the highest probability (argmax),
        # we roll a loaded die to make the text creative and avoid infinite loops.
        next_id = np.random.choice(tokenizer.vocab_size, p=last_char_probs)
        
        # 6. Decode and append to the text for the next iteration
        next_char = tokenizer.decode([next_id])
        generated_text += next_char
        
    return generated_text



if __name__ == "__main__":
    print("=" * 50)
    print("Starting RNN Engine - Text Generation")
    print("=" * 50)

    # Toy Dataset (Repeated so it memorizes patterns quickly)
    raw_text="""artificial intelligence and deep learning are fascinating branches of mathematics and computer science. 
an artificial neural network attempts to simulate human brain behavior using linear algebra and calculus operations. 
at the heart of this mathematical engine, weight matrices multiply and gradients flow backward through time. 
each iteration adjusts parameters to minimize the error.""" * 5
    
    # Tokenization
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_text)
    vocab_size = tokenizer.vocab_size
    
    seq_length = 25
    print(f"\nGenerating sliding windows of size {seq_length}...")
    # create_dataset returns 3D One-Hot tensors by default
    X_train, Y_train = tokenizer.create_dataset(raw_text, seq_length=seq_length, stride=2)
    X_train = np.argmax(X_train, axis=-1)

    embedding_dimension = 64
    hidden_size = 128
    
    # Model Setup
    # EmbeddingLayer: Projects discrete token IDs into continuous vectors.
    # RNN Layer: Processes sequential memory (input_size must match embedding_dim).
    # Dense Layer: Maps hidden states back to vocabulary distribution via Softmax.
    model = NeuralNetwork(
        layers=[
            EmbeddingLayer(vocab_size=vocab_size, embedding_dim=embedding_dimension),
            RNNLayer(input_size=embedding_dimension, hidden_size=hidden_size),
            Layer(n_inputs=hidden_size, n_neurons=vocab_size, activation=Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )
    
    #Test the untrained model
    print("\n--- Generated Text BEFORE Training ---")
    seed = "the network "
    print(generate_text(model, tokenizer, seed, length=40))
    print("-" * 40)
    
    # Training
    print("\nStarting training (BPTT)...")
    # Use a small Batch Size and an aggressive Learning Rate for this quick test
    loss_history = model.train(X_train, Y_train, epochs=100, lr=0.01, batch_size=32)
    
    # Test the trained model
    print("\n--- Generated Text AFTER Training ---")
    print(generate_text(model, tokenizer, seed, length=80))
    print("-" * 40)
    
    print(f"\nInitial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")