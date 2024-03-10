import torch
import torch.nn as nn
from src.models_.attention import Attention

class LSTM(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim):
        super(LSTM, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, vocabulary_size)
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim)
        self.embedding = nn.Embedding(vocabulary_size, 512)
        self.lstm = nn.LSTMCell(512 + encoder_dim, 512)

    def forward(self, img_features, captions):
        """
        We can use teacher forcing during training. For reference, refer to
        https://www.deeplearningbook.org/contents/rnn.html

        """
        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1
        # print(f"\n\ncaptions size: {captions.shape}")
        embedding = self.embedding(captions)

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1))


        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            # print(f"image feature size: {img_features.shape}")
            # print(f"context size: {context.shape}")
            # print(f"gate size: {gate.shape}")

            # print(f"gated context size: {gated_context.shape}")
            # print(f"gated context unsqueez size: {gated_context.unsqueeze(1).shape}")
            # print(f"embeddings size: {embedding.shape}")
            # print(f"embeddings sliced size: {embedding[:, t].squeeze(1).shape}")
            lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            # print(f"lstm input size: {lstm_input.shape}\n\n")

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c