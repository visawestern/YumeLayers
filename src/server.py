# YumeLayers — HTTP-сервер: обучение, продолжение, генерация, статус, галерея.
import os, json, threading, io, time
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
from src.config import Config
from src.server_lazy import construct_backend

class YumeAPI:
    def __init__(self, cfg_path=None):
        self.cfg = cfg_path and Config.load(cfg_path) or Config()
        self.backend = construct_backend(self.cfg)
        self.lock = threading.Lock()
        self.state = {"running": False, "job": None, "progress": 0.0, "message": "", "kind": None}
        self._th = None

    # ---------- фоновая задача ----------
    def _run(self, kind, kwargs):
        def cb(entry):
            with self.lock:
                self.state["message"] = f"эпоха {entry.get('epoch')}, loss={entry.get('loss', 0):.4f}"
                self.state["kind"] = kind
        with self.lock:
            self.state.update(running=True, job=kind, progress=0.0, message="старт")
        try:
            if kind == "vae":
                self.backend.train_vae(callback=cb)
            else:
                self.backend.train_diffusion(callback=cb, use_text=kwargs.get("use_text", False))
            with self.lock:
                self.state["progress"] = 1.0
                self.state["message"] = "завершено"
        except Exception as e:
            with self.lock:
                self.state["message"] = f"ошибка: {e}"
        finally:
            with self.lock:
                self.state["running"] = False

    def start(self, kind, kwargs):
        if self.backend.error:
            return False, self.backend.error
        if self.state["running"]:
            return False, "Обучение уже идёт"
        self._th = threading.Thread(target=self._run, args=(kind, kwargs), daemon=True)
        self._th.start()
        return True, "запущено"

    def status(self):
        with self.lock:
            st = dict(self.state)
        st["backend_error"] = self.backend.error
        st["checkpoints"] = sorted(os.listdir(self.cfg.ckpt_dir)) if os.path.isdir(self.cfg.ckpt_dir) else []
        st["outputs"] = sorted(os.listdir(self.cfg.out_dir)) if os.path.isdir(self.cfg.out_dir) else []
        st["history"] = self.backend.load_history() if not self.backend.error else []
        return st

    def generate(self, data):
        if self.backend.error:
            return {"ok": False, "error": self.backend.error}
        try:
            path = self.backend.generate(self.cfg, prompt=data.get("prompt", ""),
                            steps=int(data.get("steps", 0)) or None,
                            seed=int(data.get("seed", 0)) or None)
            return {"ok": True, "path": os.path.basename(path)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

class Handler(http.server.BaseHTTPRequestHandler):
    api = None  # назначается в main()

    def log_message(self, *a):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self._send_json({"error": "not found"}, 404)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p == "/api/status":
            return self._send_json(self.api.status())
        if p.startswith("/outputs/"):
            return self._send_file(os.path.join(self.api.cfg.out_dir, os.path.basename(p)), "image/png")
        if p == "/" or p == "/index.html":
            return self._send_file(os.path.join(self.api.cfg.project_dir, "ui", "index.html"), "text/html; charset=utf-8")
        self._send_json({"error": "no route"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        p = u.path
        ln = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(ln) if ln else b""
        if p == "/api/train":
            try:
                data = json.loads(raw.decode())
            except Exception:
                data = {}
            kind = data.get("mode", "diffusion")
            cfg = self.api.cfg
            if data.get("image_size"):
                cfg.image_size = int(data["image_size"])
                cfg.latent_size = cfg.image_size // 8
            if data.get("video_paths"):
                cfg.video_files = [v for v in data["video_paths"] if v.strip()]
            if data.get("epochs"):
                cfg.dif_epochs = int(data["epochs"])
            ok, msg = self.api.start(kind, data)
            return self._send_json({"ok": ok, "message": msg})
        if p == "/api/generate":
            try:
                data = json.loads(raw.decode())
            except Exception:
                data = {}
            return self._send_json(self.api.generate(data))
        if p == "/api/upload":
            # простой multipart не разбираем; принимаем base64 как fallback не нужен — см. UI
            return self._send_json({"ok": False, "error": "upload через путь к файлу на диске"})
        self._send_json({"error": "no route"}, 404)

def main(port=7860, cfg_path=None):
    YumeAPI._instance = YumeAPI(cfg_path)
    Handler.api = YumeAPI._instance
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"YumeLayers UI: http://127.0.0.1:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    main(port)
