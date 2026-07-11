import numpy as np
from network import NeuralNetwork
from layers import Layer, RNNLayer
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
        indices = tokenizer.encode(generated_text)
        one_hot_matrix = tokenizer.to_one_hot(indices)
        
        # Add the Batch dimension: (1, Time, Vocabulary)
        X = np.expand_dims(one_hot_matrix, axis=0)
        
        # Pass the sequence through the neural network
        predictions = model.forward(X)
        
        # Extract the probabilities of the last predicted character
        # predictions.shape is (1, Time, Vocabulary)
        last_char_probs = predictions[0, -1, :]
        
        #stability trick: ensure probabilities sum to exactly 1.0
        last_char_probs = last_char_probs / np.sum(last_char_probs)
        
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
    raw_text = "to be or not to be, that is the question. " * 20
    
    # Tokenization
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_text)
    vocab_size = tokenizer.vocab_size
    
    seq_length = 15
    print(f"\nGenerating sliding windows of size {seq_length}...")
    X_train, Y_train = tokenizer.create_dataset(raw_text, seq_length=seq_length, stride=2)
    
    # Model Construction
    # - Input: character matrix (vocab_size)
    # - RNN Memory: 64 temporal context neurons
    # - Dense Output: projects from 64 to vocab_size using Softmax
    hidden_size = 64
    
    model = NeuralNetwork(
        layers=[
            RNNLayer(input_size=vocab_size, hidden_size=hidden_size),
            Layer(n_inputs=hidden_size, n_neurons=vocab_size, activation=Softmax())
        ],
        loss=CategoricalCrossEntropy()
    )
    
    #Test the untrained model
    print("\n--- Generated Text BEFORE Training ---")
    seed = "to be "
    print(generate_text(model, tokenizer, seed, length=40))
    print("-" * 40)
    
    # Training
    print("\nStarting training (BPTT)...")
    # Use a small Batch Size and an aggressive Learning Rate for this quick test
    loss_history = model.train(X_train, Y_train, epochs=100, lr=0.1, batch_size=16)
    
    # Test the trained model
    print("\n--- Generated Text AFTER Training ---")
    seed = "to be "
    print(generate_text(model, tokenizer, seed, length=40))
    print("-" * 40)
    
    print(f"\nInitial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")