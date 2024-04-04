import torch
import torch.nn as nn
from src.models_.attention import Attention

class LSTM(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, training=True):
        super(LSTM, self).__init__()

        self.training = training

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
        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1
        # print(f"\n\ncaptions size: {captions.shape}")

        prev_words = torch.zeros(batch_size, 1).long()
        if self.training:
            embedding = self.embedding(captions)
        else:
            embedding = self.embedding(prev_words)

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1))


        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            if self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training :
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))

        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)
        # print(f"size avg_features: {avg_features.shape}")
        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c