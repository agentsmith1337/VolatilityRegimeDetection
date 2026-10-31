"""PyTorch sequence networks used by `zoo.SequenceClassifier` (separate module so torch loads lazily)."""
import torch
import torch.nn as nn


class RecurrentNet(nn.Module):
    """LSTM or GRU over a window of bars; the last hidden state -> dropout -> 3 regime logits."""

    def __init__(self, cell, n_features, hidden, dropout):
        super().__init__()
        self.rnn = (nn.LSTM if cell == 'lstm' else nn.GRU)(n_features, hidden, batch_first=True)
        self.drop, self.out = nn.Dropout(dropout), nn.Linear(hidden, 3)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(self.drop(h[:, -1]))


class TransformerNet(nn.Module):
    """Small Transformer encoder over a window of bars: linear embedding + learned positions, self-attention
    layers, last position -> 3 regime logits. No causal mask needed: every bar in the window is in the past."""

    def __init__(self, n_features, window, d_model, dropout, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, 3)

    def forward(self, x):
        return self.out(self.encoder(self.embed(x) + self.pos)[:, -1])


class ConvNet(nn.Module):
    """Temporal convolution network: dilated causal 1-D convolutions (kernel 3, dilations 1, 2, 4, receptive
    field 15 bars) with residual connections, last position -> 3 regime logits."""

    def __init__(self, n_features, hidden, dropout, dilations=(1, 2, 4)):
        super().__init__()
        self.inp = nn.Conv1d(n_features, hidden, 1)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, dilation=d) for d in dilations)
        self.drop, self.out = nn.Dropout(dropout), nn.Linear(hidden, 3)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))                                  # (batch, hidden, time)
        for conv in self.convs:
            padded = nn.functional.pad(h, (2 * conv.dilation[0], 0))     # pad on the left only: causal
            h = h + self.drop(torch.relu(conv(padded)))
        return self.out(h[:, :, -1])


def n_params(net):
    return sum(p.numel() for p in net.parameters())
