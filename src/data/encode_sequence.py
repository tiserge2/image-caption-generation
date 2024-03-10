from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


def create_sequences(tokenizer, max_length, vocab_size, descriptions):
    all_in_sequence = list()

    # move through each description for the image
    for description in descriptions:
        # encode the sequence
        sequence = tokenizer.texts_to_sequences([description])[0]
        in_sequence = pad_sequences([sequence], maxlen=max_length)[0]
        all_in_sequence.append(in_sequence)
    return all_in_sequence


def decode_sequence(sequences, tokenizer):
    return tokenizer.sequences_to_texts(sequences)


if __name__ == "__main__":
    a = 2