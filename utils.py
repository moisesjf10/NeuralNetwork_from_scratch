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


import numpy as np

class BPETokenizer:
    def __init__(self):
        self.merges = {}  # Dictionary: (left_id, right_id) -> new_id
        self.vocab = {}   # Dictionary: id -> bytes (for decoding)
        self.vocab_size = 256
        
    def _get_stats(self, ids):
        """Finds the absolute frequency of each pair of adjacent tokens."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, new_id):
        """Iterates over the sequence of IDs and merges occurrences of the 'pair'."""
        new_ids = []
        i = 0
        while i < len(ids):
            # If we find the pair and we are not at the last element
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size):
        """Trains the tokenizer from scratch by building the merge tree."""
        # 1. Convert raw text to a list of UTF-8 bytes (0-255)
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        
        # Initialize the base vocabulary (the 256 elemental bytes)
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Calculate how many compression operations we need
        num_merges = vocab_size - 256
        if num_merges < 0:
            raise ValueError("vocab_size must be at least 256 (base UTF-8 size)")

        print(f"[BPE] Starting training: {num_merges} target merges...")
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break # No more possible pairs
                
            # Extract the most frequent tuple (id1, id2)
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            # Register the rule for future encodings
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Compress the current sequence
            ids = self._merge(ids, best_pair, new_id)
            
        self.vocab_size = len(self.vocab)
        print(f"[BPE] Training completed. Final vocabulary size: {self.vocab_size} tokens.")

    def encode(self, text):
        """Converts text into IDs applying the learned merge rules."""
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            
            # Find the pair present in the sequence that was merged FIRST 
            # during training (the one with the lowest assigned ID)
            pair_to_merge = None
            min_idx = float('inf')
            
            for pair in stats:
                if pair in self.merges and self.merges[pair] < min_idx:
                    min_idx = self.merges[pair]
                    pair_to_merge = pair
                    
            if pair_to_merge is None:
                break # No known pairs left
                
            # Apply the merge and repeat
            ids = self._merge(ids, pair_to_merge, self.merges[pair_to_merge])
            
        return ids

    def decode(self, ids):
        """Converts a list of IDs back into a readable string."""
        b_text = b"".join(self.vocab[idx] for idx in ids)
        # Decode tolerating errors in case the network predicts an invalid byte
        return b_text.decode('utf-8', errors='replace')
        
    def create_dataset(self, text, seq_length, stride=1):
        """Shards the text into moving windows for network input."""
        encoded_text = self.encode(text)
        X, Y = [], []
        
        for i in range(0, len(encoded_text) - seq_length, stride):
            window = encoded_text[i:i + seq_length]
            target = encoded_text[i+1:i + seq_length + 1]
            X.append(window)
            Y.append(target)
            
        X = np.array(X)
        
        # One-Hot encoding for Categorical Cross Entropy Loss
        Y_arr = np.zeros((len(Y), seq_length, self.vocab_size))
        for i, target_seq in enumerate(Y):
            for j, token_id in enumerate(target_seq):
                Y_arr[i, j, token_id] = 1.0
                
        return X, Y_arr