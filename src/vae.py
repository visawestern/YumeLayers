# YumeLayers — мини-VAE. Сжатие ×8: изображение N×N -> латент N/8×N/8×8 каналов.
import torch, torch.nn as nn

def conv3x3(ch_in, ch_out, stride=1):
    return nn.Conv2d(ch_in, ch_out, 3, stride, 1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
    def forward(self, x):
        return x + self.net(x)

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base=32, latent_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(in_ch, base), nn.SiLU(),
            self._down(base, base*2),       # /2
            self._down(base*2, base*4),     # /4
            self._down(base*4, base*8),     # /8
            nn.Conv2d(base*8, latent_ch*2, 3, 1, 1),  # mu, logvar
        )
    def _down(self, c, co):
        return nn.Sequential(ResBlock(c), conv3x3(c, co, 2), nn.SiLU())
    def forward(self, x):
        p = self.net(x)
        mu, logvar = p.chunk(2, dim=1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_ch=3, base=32, latent_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch, base*8, 3, 1, 1), nn.SiLU(),
            self._up(base*8, base*4),
            self._up(base*4, base*2),
            self._up(base*2, base),
            nn.Conv2d(base, out_ch, 3, 1, 1),
        )
    def _up(self, c, co):
        return nn.Sequential(ResBlock(c), nn.Upsample(scale_factor=2, mode="nearest"), conv3x3(c, co), nn.SiLU())
    def forward(self, z):
        return self.net(z)

class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(3, cfg.vae_base_ch, cfg.vae_latent_ch)
        self.decoder = Decoder(3, cfg.vae_base_ch, cfg.vae_latent_ch)
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        return mu + torch.randn_like(std) * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def vae_loss(x, recon, mu, logvar, beta=1e-4):
    recon_loss = torch.nn.functional.mse_loss(recon, x)
    # сжатие до латента
    z = torch.randn_like(mu)
    std = torch.exp(0.5 * logvar.clamp(-10, 10))
    z2 = mu + z * std
    # KL-дивергенция
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3]).mean()
    return recon_loss + beta * kl, recon_loss, kl
