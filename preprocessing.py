import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer


def kmerizer(sequencia: str, k: int = 4) -> np.ndarray:
    """Creates a numpy array containing k-mers from a string"""
    k_seq = np.array([sequencia[i:i+k] for i in range(len(sequencia)-k+1)])
    return k_seq


def tokenizator(arr: list[int | np.ndarray], max_words: int, oov_tok_len: int = 4):
    oov = 'n' * oov_tok_len
    tokenizer = Tokenizer(num_words=max_words, oov_token=oov)
    tokenizer.fit_on_texts(arr)

    # Tokenize os dados e converta em sequências numéricas
    sequences = tokenizer.texts_to_sequences(arr)
    return sequences


def zero_padder(arr: list[int | np.ndarray], pad_len: int = None) -> np.ndarray:
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
