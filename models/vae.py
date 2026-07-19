import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesVAE(nn.Module):
    """
    GRU-based VAE for time series.
    """
    def __init__(self, feature_dim, hidden_dim=128, latent_dim=16, num_layers=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_rnn = nn.GRU(feature_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        self.enc_mu = nn.Linear(2 * hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

        # Decoder
        self.dec_rnn = nn.GRU(latent_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True)
        self.dec_out = nn.Linear(hidden_dim, feature_dim)

    def encode(self, x):
        # x: [B, T, D]
        h, _ = self.enc_rnn(x)                # [B, T, 2H]
        mu = self.enc_mu(h)                   # [B, T, L]
        logvar = self.enc_logvar(h)           # [B, T, L]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: [B, T, L]
        h, _ = self.dec_rnn(z)                # [B, T, H]
        x_hat = self.dec_out(h)               # [B, T, D]
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


def vae_loss(x, x_hat, mu, logvar, beta=1.0, mask=None):
    """
    mask: [B, T, D] or [B, T] indicating valid entries (e.g. MIMIC-III).
    """
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand_as(x)
        recon = ((x_hat - x) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    else:
        recon = F.mse_loss(x_hat, x, reduction='mean')
    # KL per element, then mean
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld
