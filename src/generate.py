# YumeLayers — инференс: текст/шум -> латент -> декодирование VAE -> картинка (PNG).
import os, torch
import numpy as np
from src.vae import VAE
from src.diffusion import DiT, DDPM
from src.text_encoder import TextEncoder

def generate(cfg, prompt="", steps=None, seed=None, return_np=False):
    device = torch.device(cfg.resolve_device())
    vae = VAE(cfg).to(device)
    vae_path = os.path.join(cfg.ckpt_dir, "vae.pt")
    if not os.path.exists(vae_path):
        raise RuntimeError("VAE не обучен. Сначала обучи VAE.")
    vae.load_state_dict(torch.load(vae_path, map_location=device))

    model = DiT(cfg).to(device)
    diff_path = os.path.join(cfg.ckpt_dir, "yume_layers.pt")
    if os.path.exists(diff_path):
        model.load_state_dict(torch.load(diff_path, map_location=device))

    if seed is not None:
        torch.manual_seed(seed)
    ddpm = DDPM(cfg.dif_steps)
    steps = steps or cfg.dif_sampling_steps
    text_emb = None
    if prompt and cfg.dif_use_text:
        text_enc = TextEncoder(cfg, device)
        text_emb = text_enc.encode([prompt])

    L = cfg.latent_size
    shape = (1, cfg.vae_latent_ch, L, L)
    z = ddpm.sample(model, shape, steps=steps, device=device, text_emb=text_emb)
    with torch.no_grad():
        img = vae.decode(z)   # 1,3,S,S в [0,1] (приближённо)
    img = img.clamp(0, 1).permute(0, 2, 3, 1)[0].cpu().numpy()
    if return_np:
        return img
    # сохранить PNG
    os.makedirs(cfg.out_dir, exist_ok=True)
    from PIL import Image
    path = os.path.join(cfg.out_dir, f"out_{len(os.listdir(cfg.out_dir)):04d}.png")
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path
