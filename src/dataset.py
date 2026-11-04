# YumeLayers — датасет.
# 1) Синтетические сцены (фон + объекты + свет) с автоматической разметкой слоёв.
# 2) Реальные видео -> кадры (ffmpeg) -> VAE-латент.
import os, math, random, subprocess, tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def _vstack(*arrays):
    return np.concatenate(arrays, axis=0)

def gen_synthetic_frame(size=192, rng=None):
    """Возвращает (rgb, layers) где layers: [фон_rgb, объекты_rgb, маска, свет]"""
    rng = rng or random.Random()
    # фон: радиальный/линейный градиент
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    c_top = np.array([rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.9)])
    c_bot = np.array([rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.5)])
    t = y / size
    bg = c_top[None, None, :] * t[..., None] + c_bot[None, None, :] * (1 - t[..., None])
    mask = np.zeros((size, size, 1), np.float32)
    obj = np.zeros((size, size, 3), np.float32)
    # объекты: 1-3 фигуры
    n_obj = rng.randint(1, 3)
    for _ in range(n_obj):
        cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        r = rng.uniform(size * 0.05, size * 0.2)
        color = np.array([rng.uniform(0.3, 1), rng.uniform(0.3, 1), rng.uniform(0.3, 1)])
        if rng.random() < 0.5:
            m = ((x - cx*size)**2 + (y - cy*size)**2) < r**2
        else:
            m = (np.abs(x - cx*size) < r) & (np.abs(y - cy*size) < r * 0.6)
        mask[..., 0] = np.maximum(mask[..., 0], m.astype(np.float32))
        obj += color[None, None, :] * m[..., None]
    # свет: направленный градиент слева-сверху
    light = np.clip(0.4 + 0.6 * (1 - (x + y) / (2 * size)), 0, 1).astype(np.float32)
    light = light[..., None]
    layers = np.concatenate([bg, obj, mask, light], axis=-1)  # S,S,8
    rgb = bg * (1 - mask) + obj * mask
    rgb = np.clip(rgb * light, 0, 1)
    return rgb, layers

class SyntheticDataset(Dataset):
    def __init__(self, cfg, n=None):
        self.cfg = cfg
        self.n = n or cfg.synthetic_count
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        rgb, layers = gen_synthetic_frame(self.cfg.image_size)
        return torch.tensor(rgb.transpose(2, 0, 1)), torch.tensor(layers.transpose(2, 0, 1))

def video_to_frames(video_path, out_dir, fps=3, max_w=192, max_h=192):
    """Извлекает кадры через ffmpeg, возвращает список путей к PNG."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    pat = os.path.join(out_dir, f"{base}_%05d.png")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf",
           f"fps={fps},scale='min({max_w},iw)':'min({max_h},ih)':force_original_aspect_ratio=decrease",
           "-q:v", "2", pat]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted([os.path.join(out_dir, p) for p in os.listdir(out_dir) if p.startswith(base + "_")])

def filter_frames(paths, min_std=12.0, max_black=0.2):
    """Отбрасывает тёмные/дубликатные кадры."""
    import cv2
    out = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.std() < min_std:
            continue
        if (img < 8).mean() > max_black:
            continue
        out.append(p)
    return out

def encode_frames_to_latents(paths, vae, device, size=192, out_path=None):
    """Прогоняет кадры через VAE-энкодер -> тензор латентов [N, 8, size/8, size/8]."""
    import cv2
    latents = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        with torch.no_grad():
            mu, lv = vae.encode(t.unsqueeze(0).to(device))
            z = vae.reparameterize(mu, lv)
        latents.append(z.cpu())
    if not latents:
        return None
    z = torch.cat(latents, 0)
    if out_path:
        torch.save(z, out_path)
    return z

class LatentDataset(Dataset):
    def __init__(self, latent_tensor):
        self.z = latent_tensor
    def __len__(self):
        return len(self.z)
    def __getitem__(self, i):
        return self.z[i]

class VideoFramesDataset(Dataset):
    """Кадры из видео, нормализованные в [0,1]."""
    def __init__(self, frame_paths, size=192):
        self.paths = frame_paths
        self.size = size
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        import cv2
        img = cv2.imread(self.paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return t
