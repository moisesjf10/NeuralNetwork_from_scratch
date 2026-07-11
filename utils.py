import numpy as np

class CharTokenizer:
    def __init__(self):
        """
        Initializes the empty lookup tables and vocabulary size.
        """
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0
        self.vocab = []

    def fit(self, text):
        """
        Scans the entire dataset to find all unique characters.
        Sorts them to ensure consistency across different runs.
        """
        # set() removes duplicates, list() makes it iterable, sorted() orders it
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        
        # Build the bidirectional dictionaries
        self.char_to_int = {ch: i for i, ch in enumerate(self.vocab)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        
        print(f"Tokenizer fitted. Vocabulary size: {self.vocab_size} unique characters.")

    def encode(self, text):
        """
        Converts a human readable string into a list of integer IDs.
        Example: "cab" -> [2, 0, 1]
        """
        return [self.char_to_int[ch] for ch in text]

    def decode(self, indices):
        """
        Converts a list/array of integer IDs back into a human string.
        Example: [2, 0, 1] -> "cab"
        """
        return "".join([self.int_to_char[i] for i in indices])

    def to_one_hot(self, indices):
        """
        Converts a list of integer IDs into a 2D One-Hot encoded matrix.
        Shape output: (Sequence_Length, Vocab_Size)
        """
        seq_len = len(indices)
        one_hot = np.zeros((seq_len, self.vocab_size))
        
        # NumPy advanced indexing:
        # np.arange(seq_len) selects all rows [0, 1, 2, ...]
        # indices selects the specific column for each row
        # We assign 1.0 to those exact (row, column) coordinates
        one_hot[np.arange(seq_len), indices] = 1.0
        
        return one_hot
    
    def create_dataset(self, text, seq_length, stride=1):
        """
        Slices a long text into sliding windows to train the RNN.
        Returns X and Y tensors in One-Hot format ready for the network.
        
        Args:
            text: Full string of the dataset.
            seq_length: Length of the window (e.g., 25 characters).
            stride: How many steps the window advances in each iteration (1 = max overlap).
        """
        X_indices = []
        Y_indices = []
        
        # Slide the window across the text. 
        # We subtract seq_length to ensure there is always an extra character for Y.
        for i in range(0, len(text) - seq_length, stride):
            # X: The current window the network will read
            sequence_x = text[i : i + seq_length]
            # Y: The same window, but shifted 1 position into the future
            sequence_y = text[i + 1 : i + seq_length + 1]
            
            # Convert characters to their numeric IDs
            X_indices.append(self.encode(sequence_x))
            Y_indices.append(self.encode(sequence_y))
            
        print(f"Dataset created: {len(X_indices)} windows of length {seq_length}.")
        
        # Convert lists to NumPy arrays
        X_array = np.array(X_indices)
        Y_array = np.array(Y_indices)
        
        # Convert to 3D One-Hot tensors -> Shape: (Batch, Time, Vocab)
        # Note: This can consume significant RAM if the dataset is huge.
        # If you hit MemoryErrors, you should yield these on-the-fly via a Python generator.
        batch_size = X_array.shape[0]
        X_onehot = np.zeros((batch_size, seq_length, self.vocab_size))
        Y_onehot = np.zeros((batch_size, seq_length, self.vocab_size))
        
        for b in range(batch_size):
            X_onehot[b] = self.to_one_hot(X_array[b])
            Y_onehot[b] = self.to_one_hot(Y_array[b])
            
        return X_onehot, Y_onehot