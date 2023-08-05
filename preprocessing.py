import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


class Preprocessor:
    def __init__(self, kmer_len: int = 4):
        self.kmer_len = kmer_len
        self.oov_tok: str = 'n' * kmer_len

    def kmerizer(self, sequencia: str, k: int = None) -> np.ndarray:
        """Creates a numpy array containing k-mers from a string"""

        if k is None:
            k = self.kmer_len

        k_seq = [sequencia[i:i+k] for i in range(len(sequencia)-k+1)]
        return k_seq

    def tokenizator(self, arr: list, max_words: int = None):
        """Transforms words/characters in numeric tokens"""
        words = [j for i in arr for j in i]

        if not max_words:
            max_words = len(set(words))

        tokenizer = Tokenizer(num_words=max_words, oov_token=self.oov_tok)
        tokenizer.fit_on_texts(words)

        # Tokenize os dados e converta em sequências numéricas
        sequences = tokenizer.texts_to_sequences(arr)
        return sequences

    def zero_padder(self, arr: list, pad_len: int = None) -> np.ndarray:
        """Appends zeros to inner arrays of list of arrays
        to desire padding length. If no padding length is provided,
        pads to the length of the longest array."""
        # Get max length
        max_len = max([len(i) for i in arr])
        # If padding length is provided, max_len = pad_len

        if pad_len:
            max_len = pad_len

        # Array padding
        padded = np.array([np.hstack((a, np.zeros(max_len - len(a)))) for a in arr])
        return padded
