# YumeLayers — цикл обучения (VAE и диффузия) с чекпоинтами, EMA и логгером.
import os, time, json, math
import torch
from torch.utils.data import DataLoader
from src.vae import VAE, vae_loss
from src.diffusion import DiT, DDPM
from src.text_encoder import TextEncoder
from src.dataset import SyntheticDataset, LatentDataset, VideoFramesDataset
import src.dataset as ds

class Logger:
    def __init__(self, path):
        self.path = path
        self.history = []
    def log(self, entry):
        self.history.append(entry)
        with open(self.path, "w") as f:
            json.dump(self.history[-2000:], f)
    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.history = json.load(f)
        return self.history

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.device = torch.device(cfg.resolve_device())
        self.logger = Logger(os.path.join(cfg.ckpt_dir, "history.json"))
        self.vae_path = os.path.join(cfg.ckpt_dir, "vae.pt")
        self.diff_path = os.path.join(cfg.ckpt_dir, "yume_layers.pt")

    # ---------- VAE ----------
    def train_vae(self, max_epochs=None, callback=None):
        cfg = self.cfg
        model = VAE(cfg).to(self.device)
        if os.path.exists(self.vae_path):
            model.load_state_dict(torch.load(self.vae_path, map_location=self.device))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr)
        ds_ = SyntheticDataset(cfg)
        loader = DataLoader(ds_, batch_size=cfg.vae_batch, shuffle=True, num_workers=0)
        epochs = max_epochs or cfg.vae_epochs
        for ep in range(epochs):
            tot, rec_tot, kl_tot, n = 0, 0, 0, 0
            for x, _ in loader:
                x = x.to(self.device)
                recon, mu, logvar, z = model(x)
                loss, rec, kl = vae_loss(x, recon, mu, logvar)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); rec_tot += rec.item(); kl_tot += kl.item(); n += 1
            entry = dict(kind="vae", epoch=ep, loss=tot/max(n,1), recon=rec_tot/max(n,1), kl=kl_tot/max(n,1))
            self.logger.log(entry)
            if callback: callback(entry)
            if (ep + 1) % 20 == 0:
                torch.save(model.state_dict(), self.vae_path)
        torch.save(model.state_dict(), self.vae_path)
        return model

    # ---------- Диффузия ----------
    def prepare_latents(self, max_frames=8000):
        """Готовит латенты: синтетика (VAE-энкодинг) или реальные кадры."""
        lat_path = os.path.join(self.cfg.data_dir, "latents.pt")
        if os.path.exists(lat_path):
            return torch.load(lat_path)
        vae = VAE(self.cfg).to(self.device)
        if not os.path.exists(self.vae_path):
            raise RuntimeError("Сначала обучи VAE (кнопка 'Обучить VAE').")
        vae.load_state_dict(torch.load(self.vae_path, map_location=self.device))
        if self.cfg.video_files:
            frames = []
            for v in self.cfg.video_files:
                if os.path.exists(v):
                    frames += ds.video_to_frames(v, os.path.join(self.cfg.data_dir, "frames"),
                                                 self.cfg.video_fps, self.cfg.video_max_w, self.cfg.video_max_h)
            frames = ds.filter_frames(frames)[:max_frames]
            z = ds.encode_frames_to_latents(frames, vae, self.device, self.cfg.image_size, lat_path)
            if z is not None:
                return z
        # синтетика
        ds_ = SyntheticDataset(self.cfg, n=min(self.cfg.synthetic_count, max_frames))
        zs = []
        with torch.no_grad():
            for x, _ in torch.utils.data.DataLoader(ds_, batch_size=16):
                x = x.to(self.device)
                mu, lv = vae.encode(x)
                zs.append(vae.reparameterize(mu, lv).cpu())
        z = torch.cat(zs, 0)
        torch.save(z, lat_path)
        return z

    def train_diffusion(self, max_epochs=None, callback=None, use_text=False):
        cfg = self.cfg
        z = self.prepare_latents()
        model = DiT(cfg).to(self.device)
        if os.path.exists(self.diff_path):
            model.load_state_dict(torch.load(self.diff_path, map_location=self.device))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.dif_lr)
        ddpm = DDPM(cfg.dif_steps)
        text_enc = TextEncoder(cfg, self.device) if use_text or cfg.dif_use_text else None
        loader = DataLoader(LatentDataset(z), batch_size=cfg.dif_batch, shuffle=True)
        epochs = max_epochs or cfg.dif_epochs
        step = 0
        for ep in range(epochs):
            tot, n = 0, 0
            for zb in loader:
                zb = zb.to(self.device)
                t = torch.randint(0, cfg.dif_steps, (zb.shape[0],), device=self.device)
                noise = torch.randn_like(zb)
                x_noisy = ddpm.q_sample(zb, t, noise)
                text_emb = None
                if text_enc is not None:
                    text_emb = text_enc.encode(["сцена"])  # заглушка; полноценный текст позже
                pred = model(x_noisy, t, text_emb)
                loss = torch.nn.functional.mse_loss(pred, noise)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); n += 1; step += 1
                if step % cfg.log_every == 0 and callback:
                    callback(dict(kind="diffusion", epoch=ep, step=step, loss=tot/max(n,1)))
                if step % cfg.save_every == 0:
                    torch.save(model.state_dict(), self.diff_path)
            entry = dict(kind="diffusion", epoch=ep, step=step, loss=tot/max(n,1))
            self.logger.log(entry)
            if callback: callback(entry)
        torch.save(model.state_dict(), self.diff_path)
        return model
