from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


def create_sequences(tokenizer, max_length, vocab_size, descriptions):
    all_in_sequence, all_out_sequence = list(), list()

    # move through each description for the image
    for description in descriptions:
        # encode the sequence
        sequence = tokenizer.texts_to_sequences([description])[0]
        # divide one sequence into various X,y pairs
        for i in range(1, len(sequence)):
            # divide into input and output pair
            in_sequence, out_sequence = sequence[:i], sequence[i]
            # pad input sequence
            in_sequence = pad_sequences([in_sequence], maxlen=max_length)[0]
            # encode output sequence
            out_sequence = to_categorical([out_sequence], num_classes=vocab_size)[0]
            # store
            all_in_sequence.append(in_sequence)
            all_out_sequence.append(out_sequence)
    return np.array(all_in_sequence), np.array(all_out_sequence)


def decode_sequence():
    # TODO for a given sequence, transform it into readable text
    return


if __name__ == "__main__":
    a = 2