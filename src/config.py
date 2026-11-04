# YumeLayers — параметры эксперимента.
import os, json, dataclasses
from dataclasses import dataclass, field

@dataclass
class Config:
    # --- пути ---
    project_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")
    ckpt_dir: str = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    out_dir: str = os.path.join(os.path.dirname(__file__), "..", "outputs")

    # --- устройство ---
    device: str = "auto"            # auto | cuda | mps | cpu

    # --- данные ---
    video_files: list = field(default_factory=list)   # пути к видео
    video_fps: int = 3               # кадры в секунду извлечения
    video_max_w: int = 192
    video_max_h: int = 192
    synthetic_count: int = 1500
    image_size: int = 192            # сторона кадра
    latent_size: int = 24            # image_size // 8

    # --- VAE ---
    vae_latent_ch: int = 8
    vae_base_ch: int = 32
    vae_epochs: int = 80
    vae_lr: float = 1e-3
    vae_batch: int = 8

    # --- Диффузия ---
    dif_epochs: int = 300
    dif_lr: float = 3e-4
    dif_batch: int = 8
    dif_steps: int = 200            # шагов шума при обучении
    dif_sampling_steps: int = 60
    dif_dim: int = 128
    dif_depth: int = 4
    dif_heads: int = 4
    dif_text_dim: int = 64
    dif_use_text: bool = False

    # --- обучение ---
    grad_checkpoint: bool = True
    use_fp16: bool = True
    log_every: int = 20
    save_every: int = 500
    seed: int = 42

    def resolve_device(self):
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def ensure_dirs(self):
        for d in (self.data_dir, self.ckpt_dir, self.out_dir):
            os.makedirs(d, exist_ok=True)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
