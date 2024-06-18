from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


def create_sequences(tokenizer, max_length, vocab_size, descriptions):
    """
    Create tokenized sequences from a list of descriptions.

    Args:
        tokenizer (Tokenizer): Tokenizer object used to convert text to sequences.
        max_length (int): Maximum length of the sequences.
        vocab_size (int): Size of the vocabulary.
        descriptions (list): List of textual descriptions.

    Returns:
        list: List of tokenized sequences.
    """
    all_in_sequence = list()

    # Move through each description for the image
    for description in descriptions:
        # Encode the sequence
        in_sequence = tokenizer.texts_to_sequences([description])[0]
        all_in_sequence.append(in_sequence)
    
    return all_in_sequence


def decode_sequence(sequences, tokenizer):
    """
    Decode tokenized sequences back to text.

    Args:
        sequences (list): List of tokenized sequences.
        tokenizer (Tokenizer): Tokenizer object used to convert sequences back to text.

    Returns:
        list: List of decoded textual descriptions.
    """
    return tokenizer.sequences_to_texts(sequences)


if __name__ == "__main__":
    a = 2