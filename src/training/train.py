from src.models_.cnn import CNN
from src.models_.lstm import LSTM
from src.data.generate_token import generate_tokens
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


def train_loop(dataloader_train, optimizer, cnn, lstm, tokens_params, train_losses):
    losses = AverageMeter()
    for image, descriptions in tqdm(dataloader_train):
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
        
        word_idxs = torch.max(preds, dim=1)[1].tolist()
        initial_text = decode_sequence(descriptions.tolist(), tokens_params['tokenizer'])
        reconstructed_text = decode_sequence([word_idxs], tokens_params['tokenizer'])
        # print("\n", initial_text)
        # print(reconstructed_text, "\n")
    train_losses.append(losses.avg)


def validation_loop(dataloader_val, cnn, lstm, tokens_params, val_losses):
    losses_v = AverageMeter()
    with torch.no_grad():
        for image, descriptions in tqdm(dataloader_val):
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
            # print("\n", initial_text)
            # print(reconstructed_text, "\n")
        val_losses.append(losses_v.avg)


def main(image_dir_path, captions_path, tokens_save_path, path_config):
    # get experimentation configurations
    with open(path_config, "r") as f:
        config = json.load(f)

    # get the tokens from training
    tokens_params = generate_tokens(captions_path, tokens_save_path)
    tokenizer = tokens_params['tokenizer']
    word_indexer = tokenizer.word_index
   
    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
    ])
    
    dataset = RSICDataset(captions_path, image_dir_path, tokens_params, transform=transform)
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

    vocab_size = dataset.get_vocab_size()
    
    arch = config['architecture']

    if arch in ["alexnet", "vgg", "resnet"]:
        input_size = (224, 224)
    elif arch == "googlelenet":
        input_size = (299, 299)

    # define models
    cnn = CNN(architecture=arch, freeze_weights=config['freeze_weights'], pretrained=config['pretrained'])
    lstm = LSTM(vocabulary_size=vocab_size, encoder_dim=cnn.dim)
    optimizer = optim.Adam(lstm.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1)
    cross_entropy_loss = nn.CrossEntropyLoss().cuda()
    epochs = config['epochs']

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}")
        scheduler.step()
        train_loop(train_loader, optimizer, scheduler, cnn, lstm, tokenizer, train_losses)
        validation_loop(val_loader, cnn, lstm, tokenizer, train_losses)
        plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    # load data here ..
    image_dir_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/images"
    captions_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/caption_2.json"
    tokens_save_path = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/processed"
    path_config = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/src/configuration/config.json"
    
    main(image_dir_path, captions_path, tokens_save_path, path_config)