"""Task 3 genre-conditioned Transformer decoder."""
import math
import torch
from torch import nn


class GenreConditionedTransformer(nn.Module):
    def __init__(self, vocab_size, num_genres, d_model=256, nhead=4, num_layers=4, dim_feedforward=512, max_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.genre_emb = nn.Embedding(num_genres, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, genre_ids):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.token_emb.embedding_dim)
        h = h + self.pos_emb(positions) + self.genre_emb(genre_ids).unsqueeze(1)
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        return self.output(self.transformer(h, mask=mask))
