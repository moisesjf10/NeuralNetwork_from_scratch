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