# YumeLayers — ленивый бэкенд. Позволяет UI стартовать даже без установленного torch,
# и показывает понятное сообщение, чего не хватает.
import os, json

class _BackendBase:
    error = None
    def load_history(self):
        try:
            p = os.path.join(self.cfg.ckpt_dir, "history.json")
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)[-500:]
        except Exception:
            pass
        return []
    def train_vae(self, callback=None):
        raise NotImplementedError
    def train_diffusion(self, callback=None, use_text=False):
        raise NotImplementedError
    def generate(self, cfg, prompt="", steps=None, seed=None):
        raise NotImplementedError

def construct_backend(cfg):
    try:
        import torch  # noqa: F401
    except Exception as e:
        class Missing(_BackendBase):
            error = f"PyTorch не установлен. Выполни: pip install -r requirements.txt ({e})"
            def __init__(self, cfg): self.cfg = cfg
        return Missing(cfg)
    try:
        from src.trainer import Trainer
        from src.generate import generate

        class Real(_BackendBase):
            def __init__(self, cfg):
                self.cfg = cfg
                self.trainer = Trainer(cfg)
                self._generate = generate
            def train_vae(self, callback=None):
                return self.trainer.train_vae(callback=callback)
            def train_diffusion(self, callback=None, use_text=False):
                return self.trainer.train_diffusion(callback=callback, use_text=use_text)
            def generate(self, cfg, prompt="", steps=None, seed=None):
                return self._generate(cfg, prompt=prompt, steps=steps, seed=seed)
        return Real(cfg)
    except Exception as e:
        class Broken(_BackendBase):
            error = f"Ошибка инициализации бэкенда: {e}"
            def __init__(self, cfg): self.cfg = cfg
        return Broken(cfg)
