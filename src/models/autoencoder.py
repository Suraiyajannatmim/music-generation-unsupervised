"""Task 1 LSTM Autoencoder model.

The complete Colab implementation is in notebooks/music_generation_full_4_tasks_colab.ipynb
and mirrored in src/full_pipeline_reference.py.
"""
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=512, latent_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.encoder(emb)
        z = self.to_latent(h[-1])
        h0 = self.from_latent(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        dec, _ = self.decoder(emb, (h0, c0))
        return self.output(dec), z
