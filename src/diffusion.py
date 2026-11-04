# YumeLayers — мини-DiT (патч-трансформер) + DDPM-расписание.
# Работает в латентном пространстве latent_ch × (H/8) × (W/8).
import torch, torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period, dtype=t.dtype, device=t.device)) * torch.arange(half, dtype=t.dtype, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.adaLN = nn.Linear(dim, dim*6)
    def forward(self, x, t_emb):
        shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = self.adaLN(t_emb).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        h, _ = self.attn(h, h, h)
        x = x + gate_msa[:, None] * h
        h = self.norm2(x) * (1 + scale_ff[:, None]) + shift_ff[:, None]
        x = x + gate_ff[:, None] * self.ff(h)
        return x

class DiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch = cfg.image_size // 8 // cfg.latent_size  # 1 (24x24 патч 1)
        self.patch = 1
        in_ch = cfg.vae_latent_ch * self.patch * self.patch
        self.proj = nn.Linear(in_ch, cfg.dif_dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.latent_size * cfg.latent_size, cfg.dif_dim))
        self.time_mlp = nn.Sequential(nn.Linear(cfg.dif_dim, cfg.dif_dim), nn.SiLU(), nn.Linear(cfg.dif_dim, cfg.dif_dim))
        self.blocks = nn.ModuleList([DiTBlock(cfg.dif_dim, cfg.dif_heads) for _ in range(cfg.dif_depth)])
        self.norm = nn.LayerNorm(cfg.dif_dim)
        self.head = nn.Linear(cfg.dif_dim, cfg.vae_latent_ch * self.patch * self.patch)
        if cfg.dif_use_text:
            self.text_proj = nn.Linear(cfg.dif_text_dim, cfg.dif_dim)
    def forward(self, z, t, text_emb=None):
        B, C, H, W = z.shape
        # z: B,C,H,W -> B, L, C (патч 1x1)
        x = z.flatten(2).transpose(1, 2)  # B, L, C
        x = self.proj(x) + self.pos
        t_emb = self.time_mlp(timestep_embedding(t, self.cfg.dif_dim))
        if text_emb is not None and self.cfg.dif_use_text:
            t_emb = t_emb + self.text_proj(text_emb)
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.norm(x)
        x = self.head(x)
        return x.transpose(1, 2).view(B, C, H, W)

class DDPM:
    def __init__(self, num_steps=200, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, 0)
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise
    def sample(self, model, shape, steps=None, device="cpu", text_emb=None):
        steps = steps or self.num_steps
        x = torch.randn(shape, device=device)
        betas = (1 - self.alphas_cumprod).to(device)
        for i in reversed(range(steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred = model(x, t, text_emb)
            alpha = self.alphas_cumprod[i].to(device)
            alpha_prev = self.alphas_cumprod[i-1].to(device) if i > 0 else torch.tensor(1.0, device=device)
            beta = betas[i]
            noise = (x - torch.sqrt(alpha) * pred) / torch.sqrt(1 - alpha + 1e-6)
            x = torch.sqrt(alpha_prev) * pred + torch.sqrt(1 - alpha_prev) * noise
            if i > 0:
                sigma = torch.sqrt(beta * (1 - alpha_prev) / (1 - alpha + 1e-6))
                x = x + sigma * torch.randn_like(x)
        return x
