import torch.nn as nn
import torch.nn.functional as F
import torch

# soft attention

class Attention(nn.Module): 
    def __init__(self, encoder_dim):
        """
        Initialize the Attention module.

        Args:
            encoder_dim (int): The dimension of the encoder's output features.
        """
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)
        self.W = nn.Linear(encoder_dim, 512)
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img_features, hidden_state):
        """
        Forward pass for the Attention module.

        Args:
            img_features (torch.Tensor): The image features from the encoder. Shape (batch_size, num_features, encoder_dim).
            hidden_state (torch.Tensor): The hidden state from the LSTM. Shape (batch_size, hidden_dim).

        Returns:
            context (torch.Tensor): The context vector computed as a weighted sum of the image features. Shape (batch_size, encoder_dim).
            alpha (torch.Tensor): The attention weights. Shape (batch_size, num_features).
        """
        U_h = self.U(hidden_state).unsqueeze(1)  # Shape (batch_size, 1, 512)
        W_s = self.W(img_features)  # Shape (batch_size, num_features, 512)
        att = self.relu(W_s + U_h)  # Shape (batch_size, num_features, 512)
        e = self.v(att).squeeze(2)  # Shape (batch_size, num_features)
        alpha = self.softmax(e)  # Shape (batch_size, num_features)
        context = (img_features * alpha.unsqueeze(2)).sum(1)  # Shape (batch_size, encoder_dim)
        return context, alpha
