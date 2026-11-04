# YumeLayers — текстовый энкодер.
# Основной путь: CLIP (заморожен). Запасной: мини-энкодер поверх хэш-словаря.
import os, torch, torch.nn as nn

class MiniTextEncoder(nn.Module):
    """Офлайн-мини-энкодер: словарь из hash-бакетов + Linear-проекция."""
    def __init__(self, vocab_buckets=4096, emb_dim=64, out_dim=64):
        super().__init__()
        self.buckets = vocab_buckets
        self.table = nn.Embedding(vocab_buckets, emb_dim)
        self.proj = nn.Linear(emb_dim, out_dim)
    def forward(self, texts):
        # texts: список строк
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        for t in texts:
            h = 0
            for ch in t.lower():
                h = (h * 31 + ord(ch)) % self.buckets
            ids.append(h)
        idx = torch.tensor(ids, device=next(self.parameters()).device)
        v = self.table(idx).mean(dim=0, keepdim=True)
        return self.proj(v)

class TextEncoder:
    """Обёртка: CLIP если доступен, иначе MiniTextEncoder."""
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.kind = "mini"
        self.model = MiniTextEncoder(out_dim=cfg.dif_text_dim).to(device)
        try:
            from transformers import CLIPModel, CLIPTokenizer
            hf_cache = os.path.join(cfg.data_dir, "hf")
            os.makedirs(hf_cache, exist_ok=True)
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=hf_cache).to(device).eval()
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=hf_cache)
            self.kind = "clip"
        except Exception:
            self.clip = None
    def encode(self, texts):
        if self.kind == "clip" and self.clip is not None:
            with torch.no_grad():
                toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                return self.clip.get_text_features(**toks)
        return self.model(texts)
