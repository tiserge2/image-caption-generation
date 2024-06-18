import sys 
import os
import json
import re
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
import argparse



def launch_evaluation(image_dir_path, test_caption_path, all_captions_path, tokens_path, config_path, best_model_path, best_model_folder):
    """
    Launch the evaluation of a trained image captioning model on a test dataset.

    Args:
        image_dir_path (str): Path to the directory containing test images.
        test_caption_path (str): Path to the JSON file containing test captions.
        all_captions_path (str): Path to the JSON file containing all captions.
        tokens_path (str): Path to the saved tokenizer file.
        config_path (str): Path to the configuration JSON file.
        best_model_path (str): Path to the saved best model checkpoint.
        best_model_folder (str): Path to the folder where evaluation results will be saved.

    Returns:
        tuple: A tuple containing references_data, predictions_data, and evaluation scores.
    """
    # Get experimentation configurations
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get the tokens from training
    tokens_params = load_tokens(all_captions_path, tokens_path)
    
    # Setting the training running device
    use_cuda = config['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Selected device: {device}")

    arch = config['architecture']
    if arch in ["alexnet", "vgg", "resnet18"]:
        input_size = (224, 224)
    elif arch == "googlelenet":
        input_size = (299, 299)

    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
    ])
    
    test_dataset = RSICDataset(test_caption_path, image_dir_path, tokens_params, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    save_att_vis_path = os.path.join(best_model_folder, "attention_viz")
    os.makedirs(save_att_vis_path, exist_ok=True)

    # Define models
    cnn = CNN(architecture=arch, freeze_weights=config['freeze_weights'], pretrained=config['pretrained'])
    lstm = LSTM(
        vocabulary_size=tokens_params['vocab_size'], 
        encoder_dim=cnn.dim, 
        training=False, 
        device=device, 
        show_attention=True, 
        tokenizer=tokens_params['tokenizer'], 
        save_att_vis_path=save_att_vis_path
    )

    cnn.to(device)
    lstm.to(device)

    # Load the model
    checkpoint = torch.load(best_model_path)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    lstm.load_state_dict(checkpoint['lstm_state_dict'])

    predictions_data = {}
    references_data = {}

    for image, descriptions, all_descriptions, filename in tqdm(test_loader):
        all_descriptions = [re.sub(' +', ' ', x[0].replace("<start>", "").replace("<eos>", "").replace("<pad>", "").strip()) for x in all_descriptions]
        with torch.no_grad():
            image = image.to(device, dtype=torch.float32)
            image_features = cnn(image)
            descriptions = descriptions.to(device)
            preds, alphas = lstm(image_features, descriptions, images=image)
            targets = descriptions[:, 1:]
            
            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
            word_idxs = torch.max(preds, dim=1)[1].tolist()
            initial_text = decode_sequence(descriptions.tolist(), tokens_params['tokenizer'])
            reconstructed_text = decode_sequence([word_idxs], tokens_params['tokenizer'])

            predictions_data[filename] = [re.sub(' +', ' ', x.split("<eos>")[0].replace("<start>", "").replace("<pad>", "").strip()) for x in reconstructed_text]
            references_data[filename] = all_descriptions

    # Create an instance of MetricsCalculator
    metrics_calculator = MetricsCalculator(references_data, predictions_data)

    # Compute evaluation metrics
    metrics_calculator.compute_bleu()
    metrics_calculator.compute_rouge_n()
    metrics_calculator.compute_rouge_l()
    metrics_calculator.compute_cider()
    metrics_calculator.compute_meteor()

    scores = metrics_calculator.scores
    metrics_results = show_metrics(scores)

    # Save the metrics in a text file
    with open(os.path.join(best_model_folder, "test_results.txt"), "w") as f:
        f.write(metrics_results)

    return references_data, predictions_data, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on test data and generate evaluation results.")
    parser.add_argument('--root', type=str, default=r"/home/sagemaker-user/rscid/", help='Root directory')
    args = parser.parse_args()

    root = args.root
    image_dir_path = os.path.join(root, "data/interim/images")
    all_captions_path = os.path.join(root, "data/interim/captions_all.json")
    test_captions_path = os.path.join(root, "data/interim/captions_test.json")
    tokens_path = os.path.join(root, "data/processed/tokenizer.p")
    all_experience_path = os.path.join(root, "experience_history/CONFIG_ce1662f6-957e-46dc-bd48-a3841c476a7d/")
    result_csv = ""

    for i, exp in enumerate([x for x in os.listdir(all_experience_path) if "EXP" in x]):
        best_model_folder = os.path.join(all_experience_path, exp)
        path_config = os.path.join(best_model_folder, "config.json")

        try:
            with open(path_config, "r") as f:
                used_conf = json.load(f)
            best_model_path = os.path.join(best_model_folder, [x for x in os.listdir(best_model_folder) if x.split('.')[-1] == "pth"][0])
        except Exception as e:
            print("No model found in that folder")
            continue
            
        refs, preds, scores = launch_evaluation(image_dir_path, test_captions_path, all_captions_path, tokens_path, path_config, best_model_path, best_model_folder)

        if i == 0:
            result_csv += "folder;" + ";".join(scores.keys()) + ";" + "Optimizer;Learning rate;Architecture" + "\n"
        result_csv += exp + ";" +  ";".join([str(x) for x in scores.values()]) + f";{used_conf['optimizer']};{used_conf['lr']};{used_conf['architecture']}" "\n"
            
        print(result_csv)
        new_data_refs = {}
        for key, value in refs.items():
            new_key = key[0]  # Assuming each tuple key contains only one element
            new_data_refs[new_key] = value
        
        new_data_preds = {}
        for key, value in preds.items():
            new_key = key[0]  # Assuming each tuple key contains only one element
            new_data_preds[new_key] = value
        
        text_lines_empty = []
        text_lines_non_empty = []
        empty = 0
        count = 0
        for key_refs, keys_preds in zip(new_data_refs, new_data_preds):
            count += 1
            if new_data_preds[key_refs][0] != "": # all the predictions which aren't empty goes here
                text_lines_non_empty.append(f"***************** {key_refs} *****************\n")
                for index, ref in enumerate(new_data_refs[key_refs]):
                    text_lines_non_empty.append(f"{index}- {ref}\n")
                text_lines_non_empty.append(f"\n--> {new_data_preds[key_refs]}\n\n")
            else: # empty prediction here
                empty += 1
                text_lines_empty.append(f"***************** {key_refs} *****************\n")
                for index, ref in enumerate(new_data_refs[key_refs]):
                    text_lines_empty.append(f"{index}- {ref}\n")
                text_lines_empty.append(f"\n--> {new_data_preds[key_refs]}\n\n")
        
        print(f"Empty pred: {empty}")
        print(f"Non empty preds: {count - empty}")
        
        text_lines_non_empty.append("\n\n\n\n\n")
        
        text_lines = text_lines_non_empty + text_lines_empty
        
        with open(os.path.join(best_model_folder, "test_predictions.txt"), "w") as f:
            f.writelines(text_lines)
            
    with open(os.path.join(all_experience_path, "result_csv.csv"), "w") as f:
        f.writelines(result_csv)