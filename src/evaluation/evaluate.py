import os
import json
from src.models_.cnn import CNN
from src.models_.lstm import LSTM
from src.data.generate_token import load_tokens
from src.data.data_loader import RSICDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from src.data.encode_sequence import decode_sequence
from src.utilities.utils import MetricsCalculator, show_metrics


def launch_evaluation(image_dir_path, test_caption_path, all_captions_path, tokens_path, path_config, best_model_path):
    # get experimentation configurations
    with open(path_config, "r") as f:
        config = json.load(f)

    # get the tokens from training
    tokens_params = load_tokens(all_captions_path, tokens_path)
    tokenizer = tokens_params['tokenizer']

    arch = config['architecture']

    if arch in ["alexnet", "vgg", "resnet"]:
        input_size = (224, 224)
    elif arch == "googlelenet":
        input_size = (299, 299)

   
    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
    ])
    
    test_dataset = RSICDataset(test_caption_path, image_dir_path, tokens_params, transform=transform)
    # Create DataLoader instances for training and validation
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    
    # define models
    cnn = CNN(architecture=arch, freeze_weights=config['freeze_weights'], pretrained=config['pretrained'])
    lstm = LSTM(vocabulary_size=tokens_params['vocab_size'], encoder_dim=cnn.dim, training=False)
    # Load the state dictionary
    lstm.load_state_dict(torch.load(best_model_path))

    predictions_data = {}
    references_data = {}

    for image, descriptions, all_descriptions, filename in tqdm(test_loader):
        all_descriptions = [x[0].replace("<start> ", "").replace(" <eos>", "") for x in all_descriptions]
        # print(all_descriptions, "\n")
        with torch.no_grad():
            image = torch.tensor(image, dtype=torch.float32)
            image_features = cnn(image)
            descriptions = torch.tensor(descriptions)
            preds, alphas = lstm(image_features, descriptions)
            targets = descriptions[:, 1:]
            
            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
            word_idxs = torch.max(preds, dim=1)[1].tolist()
            initial_text = decode_sequence(descriptions.tolist(), tokens_params['tokenizer'])
            reconstructed_text = decode_sequence([word_idxs], tokens_params['tokenizer'])

            # print(initial_text, "\n", reconstructed_text)

            predictions_data[filename] = reconstructed_text
            references_data[filename] = all_descriptions

    # Create an instance of MetricsCalculator
    metrics_calculator = MetricsCalculator(references_data, predictions_data)
    # Compute BLEU scores for n=1, 2, 3 and 4
    metrics_calculator.compute_bleu()
    # Compute ROUGE-N score for n=1
    metrics_calculator.compute_rouge_l()
    # Compute ROUGE-L score
    metrics_calculator.compute_rouge_l()
    # Compute CIDEr score
    metrics_calculator.compute_cider()
    # Compute Meteor score
    metrics_calculator.compute_meteor()
    scores = metrics_calculator.scores
    show_metrics(scores)
    

if __name__ == "__main__":
    image_dir_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/images"
    all_captions_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/captions_all.json"
    test_captions_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/captions_test.json"
    tokens_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/processed/tokenizer.p"
    best_model_folder = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/models_/EXP_2024-03-14_16-09-02"
    path_config =  best_model_folder +  "/config.json"
    best_model_path = best_model_folder + "/" + [x for x in os.listdir(best_model_folder) if x.split('.')[-1] == "pth"][0]
    launch_evaluation(image_dir_path, test_captions_path, all_captions_path, tokens_path, path_config, best_model_path)