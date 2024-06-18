import os
import numpy as np
import joblib
from keras_preprocessing.text import Tokenizer #for text tokenization
import json
import argparse


def txt_vocab(datas):
    """
    Build a vocabulary of all unique words from the dataset.

    Args:
        datas (list): List of data, each containing captions.

    Returns:
        set: Set of unique words in the dataset.
    """
    vocab = set()
    for data in datas:
        [vocab.update(d.split()) for d in data['captions']]
    return vocab


def create_tokenizer(descriptions):
    """
    Create and fit a tokenizer on the given descriptions.

    Args:
        descriptions (list): List of textual descriptions.

    Returns:
        Tokenizer: A fitted tokenizer object.
    """
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(descriptions)
    return tokenizer


def max_length_descriptions(descriptions):
    """
    Calculate the maximum length of descriptions.

    Args:
        descriptions (list): List of textual descriptions.

    Returns:
        int: Maximum length of the descriptions.
    """
    return max(len(d.split()) for d in descriptions)


def get_all_descriptions(train_data_path):
    """
    Get all descriptions from the training data.

    Args:
        train_data_path (str): Path to the training data JSON file.

    Returns:
        list: List of all descriptions.
    """
    # Read the JSON file
    with open(train_data_path, "r") as f:
        data = json.load(f)
        vocab = txt_vocab(data)
        print(f"Size of vocab: {len(vocab)}")
    
    descriptions = []
    # Get all the descriptions into one unique array
    for d in data:
        descriptions += d['captions']
    
    return descriptions


def generate_tokens(all_desc_path, save_path):
    """
    Generate tokenizer tokens and save the tokenizer.

    Args:
        all_desc_path (str): Path to the file containing all descriptions.
        save_path (str): Path where the tokenizer should be saved.

    Returns:
        dict: A dictionary containing vocab_size, max_length, and tokenizer.
    """
    descriptions = get_all_descriptions(all_desc_path)
    tokenizer = create_tokenizer(descriptions)
    joblib.dump(tokenizer, open(os.path.join(save_path, 'tokenizer.p'), 'wb'))
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max_length_descriptions(descriptions)
    
    return {"vocab_size": vocab_size, "max_length": max_length, "tokenizer": tokenizer}


def load_tokens(all_desc_path, tokens_path): 
    """
    Load tokenizer tokens from a saved tokenizer file.

    Args:
        all_desc_path (str): Path to the file containing all descriptions.
        tokens_path (str): Path to the saved tokenizer file.

    Returns:
        dict: A dictionary containing vocab_size, max_length, and tokenizer.
    """
    descriptions = get_all_descriptions(all_desc_path)
    max_length = max_length_descriptions(descriptions)
    tokenizer = joblib.load(open(tokens_path, "rb"))
    vocab_size = len(tokenizer.word_index) + 1
    
    return {"vocab_size": vocab_size, "max_length": max_length, "tokenizer": tokenizer}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and load tokens.')
    parser.add_argument('--all_captions_path', type=str, default=r"/home/sagemaker-user/rscid/data/interim/captions_all.json",
                        help='Path to the JSON file containing all captions')
    parser.add_argument('--tokens_save_path', type=str, default=r"/home/sagemaker-user/rscid/data/processed",
                        help='Path to save tokens')
    parser.add_argument('--tokens_path', type=str, default=r"/home/sagemaker-user/rscid/data/processed/tokenizer.p",
                        help='Path to the tokenizer file')

    args = parser.parse_args()

    # Generate tokens
    tokens_params = generate_tokens(args.all_captions_path, args.tokens_save_path)

    # Load tokens
    tokens_params = load_tokens(args.all_captions_path, args.tokens_path)
    
    print(f"vocab size: {tokens_params['vocab_size']}, max_length: {tokens_params['max_length']}")