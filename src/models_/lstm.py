import torch
import torch.nn as nn
from src.models_.attention import Attention
from src.utilities.utils import save_visualize_attention, get_attention_heatmap
import numpy as np


class DeepOutputLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the DeepOutputLayer module.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer, typically the vocabulary size.
        """
        super(DeepOutputLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context_vector, lstm_state, prev_word_embedding):
        """
        Forward pass for the DeepOutputLayer.

        Args:
            context_vector (torch.Tensor): Context vector from the attention mechanism.
            lstm_state (torch.Tensor): Hidden state from the LSTM.
            prev_word_embedding (torch.Tensor): Embedding of the previous word.

        Returns:
            torch.Tensor: Output probabilities for the next word.
        """
        input_concat = context_vector + lstm_state + prev_word_embedding
        output = self.fc1(input_concat)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        
        return output


class LSTM(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, training=True, device="gpu", show_attention=False, tokenizer=None, save_att_vis_path=None):
        """
        Initialize the LSTM module.

        Args:
            vocabulary_size (int): Size of the vocabulary.
            encoder_dim (int): Dimension of the encoder (CNN) output.
            training (bool): Whether the model is in training mode.
            device (str): Device to run the model on ('cpu' or 'cuda').
            show_attention (bool): Whether to visualize attention maps.
            tokenizer: Tokenizer for decoding sequences (optional).
            save_att_vis_path (str): Path to save attention visualizations (optional).
        """
        super(LSTM, self).__init__()

        self.training = training
        self.device = device
        self.show_attention = show_attention
        self.tokenizer = tokenizer
        self.save_att_vis_path = save_att_vis_path

        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.f_beta = nn.Linear(512, 512)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, vocabulary_size)
        # self.deep_output = DeepOutputLayer(encoder_dim, 512, vocabulary_size)
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim)
        self.embedding = nn.Embedding(vocabulary_size, 512)
        self.lstm = nn.LSTMCell(512 + encoder_dim, 512)

    def forward(self, img_features, captions, images=None):
        """
        Forward pass for the LSTM module.

        Args:
            img_features (torch.Tensor): Features from the CNN encoder.
            captions (torch.Tensor): Input captions for training or inference.
            images (torch.Tensor, optional): Images for visualization (default: None).

        Returns:
            torch.Tensor: Predicted word probabilities for each time step.
            torch.Tensor: Attention weights for each time step.
        """
        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1
        prev_words = torch.full((batch_size, 1), 2, dtype=torch.long, device=self.device)
        
        if self.training:
            embedding = self.embedding(captions)
        else:
            embedding = self.embedding(prev_words)

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1))
        
        if not self.training and self.show_attention and images is not None:
            image_show = np.array(255 * images.squeeze(0).cpu()).transpose((1, 2, 0)).astype(np.uint8)
            holder_attention_over = {"image": image_show, "word_attentions": []}
        
        for t in range(max_timespan):
            if captions[:, t][0] == 3:
                break
            gated_context, alpha = self.attention(img_features, h)
            if self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)
            
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))
                
            if not self.training and self.show_attention and images is not None:
                word_index = np.array(output.max(1)[1].cpu()).astype(np.uint8)
                res = get_attention_heatmap(image_show, gated_context.cpu(), word_index, tokenizer=self.tokenizer)
                holder_attention_over["word_attentions"].append(res)

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
            
            if self.training and output.max(1)[1].cpu().numpy()[0] == 3:
                break

        if not self.training and self.show_attention and images is not None:
            save_visualize_attention(holder_attention_over, self.save_att_vis_path)
        
        return preds, alphas

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.deep_output.bias.data.fill_(0)
        self.deep_output.weight.data.uniform_(-0.1, 0.1)
        
    def get_init_lstm_state(self, img_features):
        """
        Initializes the hidden state and cell state of the LSTM.

        Args:
            img_features (torch.Tensor): Features from the CNN encoder.

        Returns:
            torch.Tensor: Initialized hidden state.
            torch.Tensor: Initialized cell state.
        """
        avg_features = img_features.mean(dim=1)
        c = self.init_c(avg_features)
        h = self.init_h(avg_features)
        return h, c
