import os
import numpy as np
from pickle import dump, load
from keras.preprocessing.text import Tokenizer #for text tokenization
import json

def txt_vocab(datas):
    # To build vocab of all unique words
    vocab = set()
    for data in datas:
        [vocab.update(d.split()) for d in data['captions']]
    return vocab

def create_tokenizer(descriptions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(descriptions)
    return tokenizer


def max_length_descriptions(descriptions):
    return max(len(d.split()) for d in descriptions)


def get_all_descriptions(train_data_path):
    # read the jsons
    with open(train_data_path, "r") as f:
        data = json.load(f)
        vocab = txt_vocab(data)
        print(f"size of vocab: {len(vocab)}")
    descriptions = []
    # get all the descriptions into one unique array
    for d in data:
        descriptions += d['captions']
    return descriptions


def generate_tokens(all_desc_path, save_path):
    descriptions = get_all_descriptions(all_desc_path)
    tokenizer = create_tokenizer(descriptions)
    dump(tokenizer, open(os.path.join(save_path, 'tokenizer.p'), 'wb'))
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max_length_descriptions(descriptions)
    return {"vocab_size": vocab_size, "max_length": max_length, "tokenizer": tokenizer}

def load_tokens(all_desc_path, tokens_path):
    descriptions = get_all_descriptions(all_desc_path)
    max_length = max_length_descriptions(descriptions)
    tokenizer = load(open(tokens_path, "rb"))
    vocab_size = len(tokenizer.word_index) + 1
    return {"vocab_size": vocab_size, "max_length": max_length, "tokenizer": tokenizer}

if __name__ == "__main__":
    all_captions_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/captions_all.json"
    tokens_save_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/processed"
    tokens_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/processed/tokenizer.p"
    # tokens_params = generate_tokens(all_captions_path, tokens_save_path)
    tokens_params = load_tokens(all_captions_path, tokens_path)
    print(f"vocab size: {tokens_params['vocab_size']}, max_length: {tokens_params['max_length']}")