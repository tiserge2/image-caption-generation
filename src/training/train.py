import sys 
from src.models_.cnn import CNN
from src.models_.lstm import LSTM
from src.data.generate_token import load_tokens
from src.data.data_loader import RSICDataset
from src.utilities.utils import AverageMeter, accuracy, calculate_caption_lengths
from src.data.encode_sequence import decode_sequence
from src.utilities.plotting import plot_losses
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import json
from tqdm import tqdm
import os
import datetime
import shutil
import math
from torch.utils.tensorboard import SummaryWriter
import uuid
import argparse



def remove_existing_saved_model(folder_path):
  """
  Deletes existing best model saved within a folder (use with caution).

  Args:
      folder_path (str): The path to the folder to be emptied.
  """

  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and "best_model" in filename:
      os.remove(file_path)



def train_loop(dataloader_train, optimizer, cnn, lstm, train_losses, cross_entropy_loss, word_indexer, device):
    """
    Executes the training loop over the entire training dataset.

    Args:
        dataloader_train (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        cnn (torch.nn.Module): Convolutional Neural Network model (e.g., AlexNet, VGG, ResNet).
        lstm (torch.nn.Module): LSTM model for generating captions.
        train_losses (list): List to store training losses for each epoch.
        cross_entropy_loss (torch.nn.CrossEntropyLoss): Loss function for computing sequence prediction loss.
        word_indexer (dict): Dictionary mapping words in the vocabulary to indices.
        device (torch.device): Device where tensors are allocated ('cpu' or 'cuda').
    """

    losses = AverageMeter()
    for image, descriptions, _, _ in tqdm(dataloader_train):
        image = torch.tensor(image, dtype=torch.float32).to(device)
        image_features = cnn(image)
        descriptions = descriptions.to(device)
        preds, alphas = lstm(image_features, descriptions)
        targets = descriptions[:, 1:]
        
        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        att_regularization = 1 * ((1 - alphas.sum(1))**2).mean()

        loss = cross_entropy_loss(preds.to(device), targets.to(device))
        loss += att_regularization
        loss.backward()
        optimizer.step()
        total_caption_length = calculate_caption_lengths(word_indexer, descriptions)
        losses.update(loss.item(), total_caption_length) 
    print(f"Total caption: {losses.count}")
    print(f"Total loss: {losses.sum}")
    train_losses.append(losses.avg)


def validation_loop(dataloader_val, cnn, lstm, tokens_params, val_losses, cross_entropy_loss, word_indexer, device, show_example=False):
    """
    Executes the validation loop over the entire validation dataset.

    Args:
        dataloader_val (DataLoader): DataLoader for the validation dataset.
        cnn (torch.nn.Module): Convolutional Neural Network model (e.g., AlexNet, VGG, ResNet).
        lstm (torch.nn.Module): LSTM model for generating captions.
        tokens_params (dict): Parameters related to tokens (e.g., tokenizer, vocabulary size).
        val_losses (list): List to store validation losses for each epoch.
        cross_entropy_loss (torch.nn.CrossEntropyLoss): Loss function for computing sequence prediction loss.
        word_indexer (dict): Dictionary mapping words in the vocabulary to indices.
        device (torch.device): Device where tensors are allocated ('cpu' or 'cuda').
        show_example (bool, optional): Flag to print example predictions and references. Default is False.
    """
    losses_v = AverageMeter()
    with torch.no_grad():
        for image, descriptions, _, _ in tqdm(dataloader_val):
            image = torch.tensor(image, dtype=torch.float32).to(device)
            image_features = cnn(image)
            descriptions = descriptions.to(device)
            preds, alphas = lstm(image_features, descriptions)
            targets = descriptions[:, 1:]
            
            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization = 1 * ((1 - alphas.sum(1))**2).mean()

            loss = cross_entropy_loss(preds.to(device), targets.to(device))
            loss += att_regularization


            total_caption_length = calculate_caption_lengths(word_indexer, descriptions)
            losses_v.update(loss.item(), total_caption_length)

            word_idxs = torch.max(preds, dim=1)[1].tolist()
            initial_text = decode_sequence(descriptions.tolist(), tokens_params['tokenizer'])
            reconstructed_text = decode_sequence([word_idxs], tokens_params['tokenizer'])
        if show_example:
            print("\n", initial_text)
            print(reconstructed_text, "\n") 
        val_losses.append(losses_v.avg)


def launch_training(image_dir_path, train_captions_path, all_captions_path, tokens_path, config_json, best_models_path):
    """
    Orchestrates the entire training process for the image captioning model.

    Args:
        image_dir_path (str): Path to the directory containing images for training.
        train_captions_path (str): Path to the file containing training captions.
        all_captions_path (str): Path to the file containing all captions for tokenization.
        tokens_path (str): Path to the file containing tokens for training.
        config_json (dict): Configuration settings for the training process.
        best_models_path (str): Path to the directory where best model checkpoints will be saved.
    """

    # get experimentation configurations
    config = config_json

    # get the tokens from training
    tokens_params = load_tokens(all_captions_path, tokens_path)
    tokenizer = tokens_params['tokenizer']
    word_indexer = tokenizer.word_index

    # setting the training running device
    use_cuda = config['use_cuda'] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Selected device: {device}")

    arch = config['architecture']

    if arch in ["alexnet", "vgg", "resnet18"]:
        input_size = (224, 224)
    elif arch == "googlelenet":
        input_size = (299, 299)

   
    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Set a random seed for reproducibility
    seed = 42  # You can choose any integer value
    torch.manual_seed(seed)
    
    dataset = RSICDataset(train_captions_path, image_dir_path, tokens_params, transform=transform)
    # Define the size of the validation set
    train_set_size = config['train_split']  
    # Calculate the size of the validation set
    train_size = int(train_set_size * len(dataset))
    val_size = len(dataset) - train_size
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create DataLoader instances for training and validation
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # define models
    cnn = CNN(architecture=arch, freeze_weights=config['freeze_weights'], pretrained=config['pretrained'])
    lstm = LSTM(vocabulary_size=tokens_params['vocab_size'], encoder_dim=cnn.dim, device=device)
    cnn.to(device)
    lstm.to(device)
    parameters = list(cnn.parameters()) + list(lstm.parameters())
    
    if config['optimizer'] == "SGD":
        optimizer = optim.SGD(parameters, lr=config['lr'])
    elif config['optimizer'] == "AdamW":
        optimizer = optim.AdamW(parameters, lr=config['lr'])
    elif config['optimizer'] == "RAdam":
        optimizer = optim.RAdam(parameters, lr=config['lr'])
    elif config['optimizer'] == "Adam":
        optimizer = optim.Adam(parameters, lr=config['lr'])
    else:
        raise ValueError(f"The optimizer {config['optimizer']} doesn't match any implemented ones.")
        
    cross_entropy_loss = nn.CrossEntropyLoss().cuda()
    epochs = config['epochs']

    train_losses = []
    val_losses = []


    # Get current date and time
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    time_str = now.strftime("%H:%M:%S")  # Format: HH:MM:SS

    best_model_folder_name = f"EXP_{date_str}_{time_str.replace(':', '-')}"
    best_model_folder = os.path.join(best_models_path, best_model_folder_name)
    os.makedirs(best_model_folder, exist_ok=True)

    best_loss = math.inf
    best_lost_epoch = 1
    patience = config['patience']

    show_plot = config['show_loss_plot']
    
    path_tensor_logs = os.path.join(best_model_folder, "tensor_logs")
    writer = SummaryWriter(path_tensor_logs)
    
    for epoch in range(1, epochs + 1): # start the training
        print(f"\n\n=========> Epoch: {epoch}/{epochs} <=========")
        train_loop(train_loader, optimizer, cnn, lstm, train_losses, cross_entropy_loss, word_indexer, device)
        validation_loop(val_loader, cnn, lstm, tokens_params, val_losses, cross_entropy_loss, word_indexer, device, show_example=config['show_example'])
        writer.add_scalars("Losses", {"train": train_losses[epoch - 1], "validation": val_losses[epoch - 1]}, epoch - 1)
        print("train loss", train_losses)
        print("val losses", val_losses)
        if epoch > 1:
            plot_losses(train_losses, val_losses, best_model_folder, show_plot=show_plot)
            if val_losses[-1] < best_loss:
                best_lost_epoch = epoch
                best_loss = val_losses[-1]
                model_file = f'best_model_epoch-{str(epoch)}_loss-{best_loss}_.pth'
                # check if there is a model already saved 
                remove_existing_saved_model(best_model_folder)
                torch.save({
                    'cnn_state_dict': cnn.state_dict(),
                    'lstm_state_dict': lstm.state_dict(),
                    'config': config
                }, os.path.join(best_model_folder, model_file))
                # save json configuration also
                with open(os.path.join(best_model_folder, "config.json"), "w") as f:
                    json.dump(config, f, indent=4)
                print('Saved model to ' + model_file)
            # early stopping
            if epoch - best_lost_epoch > patience:
                print("====> Stopping training because of early stopping...")
                break
    writer.close()


if __name__ == "__main__":
    # load data here ..
    parser = argparse.ArgumentParser(description='Launch training with configurations.')
    parser.add_argument('--root', type=str, default=r"/home/sagemaker-user/rscid/",
                        help='Root directory path')
    parser.add_argument('--image_dir_path', type=str, default=r"data/interim/images",
                        help='Path to the directory containing images')
    parser.add_argument('--all_captions_path', type=str, default=r"data/interim/captions_all.json",
                        help='Path to the JSON file containing all captions')
    parser.add_argument('--train_captions_path', type=str, default=r"data/interim/captions_train.json",
                        help='Path to the JSON file containing training captions')
    parser.add_argument('--tokens_path', type=str, default=r"data/processed/tokenizer.p",
                        help='Path to the tokenizer file')
    parser.add_argument('--path_config', type=str, default=r"src/configuration/config.json",
                        help='Path to the configuration JSON file')
    parser.add_argument('--best_models_path', type=str, default=r"experience_history/",
                        help='Path to save the best models')

    args = parser.parse_args()

    # Create a unique folder for this training configuration
    config_folder = uuid.uuid4()
    best_models_path = os.path.join(args.root, args.best_models_path, f"CONFIG_{config_folder}")
    os.makedirs(best_models_path, exist_ok=True)

    print(f"\n\n=========> Starting new training configuration test. Saved at: {best_models_path}\n\n")

    # Load configurations from JSON
    with open(args.path_config, "r") as f:
        all_configs = json.load(f)

    # Iterate through each configuration and launch training
    for i, config in enumerate(all_configs):
        print(f"\n\n=========> Configuration {i + 1}/{len(all_configs)} started...\n\n")
        launch_training(args.image_dir_path, args.train_captions_path, args.all_captions_path, args.tokens_path, config, best_models_path)