from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
import json
from pprint import pprint
from encode_sequence import create_sequences
from generate_token import generate_tokens, txt_vocab

class RSICDataset(Dataset):
    def __init__(self, path_annotations, img_dir, tokens_params, transform=None) -> None:
        self.img_dir = img_dir
        with open(path_annotations, "r") as f:  
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx):
        descriptions = self.annotations[idx]['captions']
        path_to_image = os.path.join(self.img_dir, self.annotations[idx]['filename'])
        image = io.imread(path_to_image)
        if self.transform:
            image = self.transform(image)

        descriptions = [x for x in descriptions]
        
        in_sequences, out_sequences = create_sequences(tokens_params['tokenizer'], tokens_params['max_length'], tokens_params['vocab_size'], descriptions)
        return image, in_sequences, out_sequences

if __name__ == "__main__":
        image_dir_path = r"./data/interim/images"
        captions_path = r"./data/interim/caption_2.json"
        tokens_save_path = r"./data/processed"

        tokens_params = generate_tokens(captions_path, tokens_save_path)

        dataset = RSICDataset(captions_path, image_dir_path, tokens_params)
        dataloader = DataLoader(dataset, batch_size=1)

        for image, in_sequences, out_sequences in dataloader:
            print(in_sequences.shape)
