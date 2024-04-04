from src.models_.cnn import CNN
from src.models_.lstm import LSTM
from src.data.generate_token import load_tokens
from src.data.data_loader import RSICDataset
from src.utilities.utils import AverageMeter, accuracy, calculate_caption_lengths
from src.data.encode_sequence import decode_sequence
from src.utilities.plotting import plot_losses, write
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



def train_loop(dataloader_train, optimizer, cnn, lstm, train_losses, cross_entropy_loss, word_indexer):
    losses = AverageMeter()
    for image, descriptions, _, _ in tqdm(dataloader_train):
        image = torch.tensor(image, dtype=torch.float32)
        image_features = cnn(image)
        descriptions = torch.tensor(descriptions)
        preds, alphas = lstm(image_features, descriptions)
        targets = descriptions[:, 1:]
        
        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        att_regularization = 1 * ((1 - alphas.sum(1))**2).mean()

        loss = cross_entropy_loss(preds, targets)
        loss += att_regularization
        loss.backward()
        optimizer.step()

        total_caption_length = calculate_caption_lengths(word_indexer, descriptions)
        losses.update(loss.item(), total_caption_length)        
    train_losses.append(losses.avg)


def validation_loop(dataloader_val, cnn, lstm, tokens_params, val_losses, cross_entropy_loss, word_indexer, show_example=False):
    losses_v = AverageMeter()
    with torch.no_grad():
        for image, descriptions, _, _ in tqdm(dataloader_val):
            image = torch.tensor(image, dtype=torch.float32)
            image_features = cnn(image)
            descriptions = torch.tensor(descriptions)
            preds, alphas = lstm(image_features, descriptions)
            targets = descriptions[:, 1:]
            
            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization = 1 * ((1 - alphas.sum(1))**2).mean()

            loss = cross_entropy_loss(preds, targets)
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


def launch_training(image_dir_path, train_captions_path, all_captions_path, tokens_path, path_config, best_models_path):
    # get experimentation configurations
    with open(path_config, "r") as f:
        config = json.load(f)

    # get the tokens from training
    tokens_params = load_tokens(all_captions_path, tokens_path)
    tokenizer = tokens_params['tokenizer']
    word_indexer = tokenizer.word_index

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
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # define models
    cnn = CNN(architecture=arch, freeze_weights=config['freeze_weights'], pretrained=config['pretrained'])
    lstm = LSTM(vocabulary_size=tokens_params['vocab_size'], encoder_dim=cnn.dim)
    parameters = list(cnn.parameters()) + list(lstm.parameters())
    optimizer = optim.SGD(parameters, lr=config['lr'])
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

    show_plot = config['show_loss_plot']
    
    path_tensor_logs = os.path.join(best_model_folder, "tensor_logs")
    writer = SummaryWriter(path_tensor_logs)

    for epoch in range(1, epochs + 1): # start the training
        print(f"\n\n=========> Epoch: {epoch}/{epochs} <=========")
        train_loop(train_loader, optimizer, cnn, lstm, train_losses, cross_entropy_loss, word_indexer)
        validation_loop(val_loader, cnn, lstm, tokens_params, val_losses, cross_entropy_loss, word_indexer, show_example=config['show_example'])
        writer.add_scalars("Losses", {"train": train_losses[epoch - 1], "validation": val_losses[epoch - 1]}, epoch - 1)

        if epoch > 1:
            plot_losses(train_losses, val_losses, best_model_folder, show_plot=show_plot)
            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                model_file = f'best_model_epoch-{str(epoch)}_loss-{best_loss}_.pth'
                # check if there is a model already saved 
                remove_existing_saved_model(best_model_folder)
                torch.save(lstm.state_dict(), os.path.join(best_model_folder, model_file))
                # save json configuration also
                with open(os.path.join(best_model_folder, "config.json"), "w") as f:
                    json.dump(config, f, indent=4)
                print('Saved model to ' + model_file)
    writer.close()
if __name__ == "__main__":
    # load data here ..
    root = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/"
    image_dir_path = root + r"data/interim/images"
    all_captions_path = root + r"data/interim/captions_all.json"
    train_captions_path = root + r"data/interim/captions_200_train.json"
    tokens_path = root + r"data/processed/tokenizer.p"
    path_config = root + r"src/configuration/config.json"
    best_models_path = root + r"models_"
    launch_training(image_dir_path, train_captions_path, all_captions_path, tokens_path, path_config, best_models_path)