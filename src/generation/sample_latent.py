"""Latent sampling helpers."""
import torch


def sample_normal(batch_size, latent_dim, device=None):
    return torch.randn(batch_size, latent_dim, device=device)
