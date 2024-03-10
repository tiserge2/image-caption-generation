from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
import json
from pprint import pprint
from src.data.encode_sequence import create_sequences
from src.data.generate_token import generate_tokens, txt_vocab
import torch
import numpy as np

class RSICDataset(Dataset):
    def __init__(self, path_annotations, img_dir, tokens_params, transform=None) -> None:
        self.img_dir = img_dir
        self.tokens_params = tokens_params
        with open(path_annotations, "r") as f:  
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)
    
    def get_vocab_size(self):
         return len(txt_vocab(self.annotations))
    
    def __getitem__(self, idx):
        descriptions = self.annotations[idx]['captions']
        path_to_image = os.path.join(self.img_dir, self.annotations[idx]['filename'])
        image = io.imread(path_to_image)
        if self.transform:
            image = self.transform(image)

        descriptions = [descriptions[0]]
        in_sequences = create_sequences(self.tokens_params['tokenizer'], self.tokens_params['max_length'], self.tokens_params['vocab_size'], descriptions)
        return image, torch.squeeze(torch.tensor(np.array(in_sequences), dtype=torch.long))

if __name__ == "__main__":
    image_dir_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/images"
    captions_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/caption_2.json"
    tokens_save_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/processed"
    path_config = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/src/configuration/config.json"
    

    tokens_params = generate_tokens(captions_path, tokens_save_path)

    tokenizer = tokens_params['tokenizer']
    word_id = tokenizer.word_index['eos']
    # print(word_id)

    # dataset = RSICDataset(captions_path, image_dir_path, tokens_params)
    # dataloader = DataLoader(dataset, batch_size=1)

    # for image, descriptions in dataloader:
    #     image = image.permute(0, 3, 1, 2)
    #     print(f"number of image size: {image.shape}")
    #     print(descriptions)
    #     print(f"descriptions shape: {descriptions.shape}\n")
