import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

def plot_losses(train_losses, val_losses, path_to_save_loss, show_plot=False):
    """
    Plot training and validation losses over epochs.

    Parameters:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
    """

    if show_plot:
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss', marker='o')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(epochs)  # Set x-coordinates to integers
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close() 