"""Task 2 conditional VAE model."""
import torch
from torch import nn


class ConditionalMusicVAE(nn.Module):
    def __init__(self, vocab_size, num_genres, emb_dim=256, genre_dim=32, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.genre_emb = nn.Embedding(num_genres, genre_dim)
        self.encoder = nn.LSTM(emb_dim + genre_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim + genre_dim, hidden_dim)
        self.decoder = nn.LSTM(emb_dim + genre_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, genre_ids):
        g = self.genre_emb(genre_ids)
        gseq = g.unsqueeze(1).expand(-1, x.size(1), -1)
        enc_in = torch.cat([self.token_emb(x), gseq], dim=-1)
        _, (h, _) = self.encoder(enc_in)
        mu, logvar = self.mu(h[-1]), self.logvar(h[-1])
        z = self.reparameterize(mu, logvar)
        h0 = self.latent_to_hidden(torch.cat([z, g], dim=-1)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        dec_in = torch.cat([self.token_emb(x), gseq], dim=-1)
        dec, _ = self.decoder(dec_in, (h0, c0))
        return self.output(dec), mu, logvar
