import pickle
import random
import numpy as np
import pandas as pd
from collections.abc import Iterable
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer


class Preprocessor:
    def __init__(self, kmer_len: int = 4):
        self.kmer_len       = kmer_len
        self.oov_tok: str   = 'n' * kmer_len

    def cleaner(self, sequence: str) -> str:
        """Replaces any non 'actgn' base by 'n'."""
        nts = 'bdefhijklmopqrsuvwxyz'
        translation_table = str.maketrans(nts, 'n'*len(nts))
        return sequence.translate(translation_table)

    def seq_tokenizer(self, sequencia: str) -> list:
        d: dict = {'a': 1, 'c': 2, 'g': 3, 't': 4}
        return [d.get(i, 5) for i in sequencia]

    def kmerizer(self, sequencia: str, k: int = None) -> np.ndarray:
        """Creates a numpy array containing k-mers from a string."""

        if k is None:
            k = self.kmer_len

        return [sequencia[i:i+k] for i in range(len(sequencia)-k+1)]

    def tokenizator(self, arr: list, max_words: int = None):
        """Transforms words/characters in numeric tokens."""
        words = [j for i in arr for j in i]

        if not max_words:
            max_words = len(set(words))

        tokenizer = Tokenizer(num_words=max_words, oov_token=self.oov_tok)
        tokenizer.fit_on_texts(words)

        with open('tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Tokenize os dados e converta em sequências numéricas
        return tokenizer.texts_to_sequences(arr)

    def zero_padder(self, arr: list, pad_len: int = None) -> np.ndarray:
        """Appends zeros to inner arrays of list of arrays
        to desire padding length. If no padding length is provided,
        pads to the length of the longest array."""
        # Get max length
        max_len = max((len(i) for i in arr))

        # If padding length is provided, max_len = pad_len
        if pad_len:
            if pad_len < max_len:
                print(f">>> WARNING: Max len '{max_len}' longer than '{pad_len}'. Truncating.")
            max_len = pad_len

        return np.array([np.hstack((a[:pad_len], np.zeros(max_len - len(a[:pad_len])))) for a in arr])

    def transform_label(self, labels: Iterable) -> np.ndarray:
        """Transform labels in string format into numerical data."""
        encoder = LabelEncoder()
        return to_categorical(encoder.fit_transform(labels))

    def get_weight(self, label_column: pd.Series) -> dict:
        d = {l: i for i, l in enumerate(label_column.unique())}
        rotulos         = label_column.map(d).values
        rotulos_unicos  = np.unique(rotulos)
        pesos = compute_class_weight(
                class_weight    = 'balanced',
                classes         = rotulos_unicos,
                y               = rotulos)
        pesos_dict = dict(zip(rotulos_unicos, pesos))
        return pesos_dict


class Sampler:
    def seq_shuffler(self, sequencia: str, state: int = None) -> str:
        """Creates sample sequences by shuffling the input sequences."""
        if state:
            random.seed(state)

        sequencia = random.sample(sequencia, len(sequencia))
        return ''.join(sequencia)

    def create_reverse_complement(self, dna: str) -> str:
        """Creates reverse complement sequence for data augmentation."""
        complement = {'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
        return ''.join([complement.get(base, 'n') for base in dna[::-1]])

    def create_random_sequences(self, n_seqs: int = 1, lmin: int = 200, lmax: int = 10_000, state: int = None) -> Iterable:
        """Creates random DNA sequences to use as sample."""
        if state:
            random.seed(state)

        lengths     = [ random.randint(lmin, lmax) for _ in range(n_seqs) ]
        characters  = 'actg'
        for l in lengths:
            yield ''.join(random.choice(characters) for _ in range(l))
