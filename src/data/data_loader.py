import sys 
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
import json
from pprint import pprint
from src.data.encode_sequence import create_sequences
from src.data.generate_token import generate_tokens, txt_vocab, load_tokens
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import random

class RSICDataset(Dataset):
    """
    Remote Sensing Image Captioning (RSIC) Dataset.

    This dataset class is used to load images and their corresponding captions for the task of remote sensing image captioning. 
    It reads the annotations from a JSON file, loads images from a directory, and applies optional transformations.

    Attributes:
        img_dir (str): Directory containing the images.
        tokens_params (dict): Parameters for tokenizing the captions, including tokenizer, max_length, and vocab_size.
        annotations (list): List of annotations loaded from the JSON file.
        transform (callable, optional): A function/transform to apply to the images.
    """
    
    def __init__(self, path_annotations, img_dir, tokens_params, transform=None) -> None:
        """
        Initialize the RSICDataset.

        Args:
            path_annotations (str): Path to the JSON file containing image annotations.
            img_dir (str): Directory where the images are stored.
            tokens_params (dict): Parameters for tokenizing captions.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.img_dir = img_dir
        self.tokens_params = tokens_params
        with open(path_annotations, "r") as f:  
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of annotations.
        """
        return len(self.annotations)
    
    def get_vocab_size(self):
        """
        Calculate the vocabulary size based on the annotations.

        Returns:
            int: Size of the vocabulary.
        """
        return len(txt_vocab(self.annotations))
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (Tensor): The transformed image.
                - in_sequences (Tensor): Tokenized caption sequences.
                - all_descriptions (list): All descriptions for the image.
                - filename (str): The filename of the image.
        """
        all_descriptions = self.annotations[idx]['captions']
        path_to_image = os.path.join(self.img_dir, self.annotations[idx]['filename'])
        filename = self.annotations[idx]['filename']
        image = io.imread(path_to_image)
        if self.transform:
            image = self.transform(image)

        descriptions = [all_descriptions[random.randint(0, 4)]]
        in_sequences = create_sequences(self.tokens_params['tokenizer'], self.tokens_params['max_length'], self.tokens_params['vocab_size'], descriptions)
        return image, torch.squeeze(torch.tensor(np.array(in_sequences), dtype=torch.long)), all_descriptions, filename
if __name__ == "__main__":
    root = r"./"
    all_captions_path = r"/home/sagemaker-user/rscid/data/interim/captions_all.json"
    tokens_save_path = r"/home/sagemaker-user/rscid/data/processed"
    tokens_path = r"/home/sagemaker-user/rscid/data/processed/tokenizer.p"
    image_dir_path = r"/home/sagemaker-user/rscid/data/interim/images"
    tokens_params = load_tokens(all_captions_path, tokens_path)

    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(11, 21), sigma=(5, 50))], p=0.5)
    ])
    
    dataset = RSICDataset(all_captions_path, image_dir_path, tokens_params, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, descriptions, _, _ in tqdm(dataloader):
        image = image.permute(0, 3, 1, 2)
        print(descriptions.to("cuda"))
        print(f"descriptions shape: {descriptions.shape}\n")
