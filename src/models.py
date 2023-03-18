import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, config: list, act: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx, layer in enumerate(config):
            self.layers.append(nn.Linear(layer[0], layer[1]))
            if idx != len(config) - 1:
                self.layers.append(act())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class RecurrentEncoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim):
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, self.seq_len, self.n_features)
        x, _ = self.rnn1(x)
        x = self.tanh(x)
        x, (hidden, _) = self.rnn2(x)

        hidden = hidden.view(-1, self.hidden_dim)

        return hidden


class RecurrentDecoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim):
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.rnn1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.tanh = nn.Tanh()

        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(1, self.seq_len)
        x = x.view(x.shape[0], self.seq_len, -1)
        x, _ = self.rnn1(x)
        x = self.tanh(x)
        x, _ = self.rnn2(x)

        x = self.output_layer(x)

        return x


class RecurrentAE(nn.Module):
    def __init__(self, n_features, hidden_dim, seq_len):
        super().__init__()

        self.encoder = RecurrentEncoder(seq_len=seq_len, n_features=n_features, hidden_dim=hidden_dim)
        self.decoder = RecurrentDecoder(seq_len=seq_len, n_features=n_features, hidden_dim=hidden_dim)


    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x


if __name__ == "__main__":
    import torch

    encoder = RecurrentEncoder(2, 3, 1)
    decoder = RecurrentDecoder(2, 3, 1)

    x = torch.zeros(( 1, 2, 3))
    
    latent = encoder(x)
    # print(latent.shape)
    x_h = decoder(latent)

    print(x_h.shape)
