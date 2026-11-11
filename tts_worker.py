#!/usr/bin/env python3
"""
Qwen3-TTS worker. Runs in its OWN venv (transformers 4.57.x), never inside ComfyUI.

Two modes, same generation code:
  one-shot:  python tts_worker.py --text "..." --out /path/out.wav [--speaker Ryan --language English]
  service:   python tts_worker.py --serve [--host 127.0.0.1 --port 8190]

Service endpoints:
  GET  /health    -> {"ok": true, "loaded": bool}
  POST /generate  -> JSON {text, speaker, language, instruct, model, dtype, attn, out, unload_after}
  POST /unload    -> frees the model's VRAM

Only CustomVoice models (generate_custom_voice) are supported for now.
"""
import argparse
import gc
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_MODEL = "/<ComfyUI>/models/qwen-tts/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
IDLE_SECONDS = int(os.environ.get("QWEN_TTS_IDLE_SECONDS", "300"))  # 0 disables idle unload

_lock = threading.Lock()
_state = {"model": None, "key": None, "last_used": 0.0}


def _log(msg):
    sys.stderr.write(f"[qwen-tts] {msg}\n")
    sys.stderr.flush()


def _unload():
    if _state["model"] is None:
        return False
    _state["model"] = None
    _state["key"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _log("model unloaded")
    return True


def _load(model_path, dtype_name, attn):
    key = (model_path, dtype_name, attn)
    if _state["model"] is not None and _state["key"] == key:
        return _state["model"]
    _unload()
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_path, device_map="cuda:0", dtype=dtype, attn_implementation=attn
    )
    _log(f"loaded {os.path.basename(model_path)} ({dtype_name}, {attn}) in {time.time() - t0:.1f}s")
    _state["model"] = model
    _state["key"] = key
    return model


def generate(req):
    import soundfile as sf

    text = (req.get("text") or "").strip()
    if not text:
        raise ValueError("text is empty")
    out = req.get("out")
    if not out:
        raise ValueError("out path is required")

    model = _load(
        req.get("model") or DEFAULT_MODEL,
        req.get("dtype") or "bf16",
        req.get("attn") or "sdpa",
    )
    kwargs = {
        "text": text,
        "language": req.get("language") or "English",
        "speaker": req.get("speaker") or "Ryan",
    }
    if req.get("instruct"):
        kwargs["instruct"] = req["instruct"]

    t0 = time.time()
    wavs, sr = model.generate_custom_voice(**kwargs)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sf.write(out, wavs[0], sr, subtype="PCM_16")  # PCM_16 so the node can read it with stdlib wave
    _state["last_used"] = time.time()
    _log(f"generated {len(text)} chars in {time.time() - t0:.1f}s -> {out}")

    if req.get("unload_after"):
        _unload()
    return {"path": out, "sample_rate": int(sr)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "loaded": _state["model"] is not None})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            with _lock:  # one GPU job at a time
                if self.path == "/generate":
                    return self._send(200, generate(req))
                if self.path == "/unload":
                    return self._send(200, {"unloaded": _unload()})
            self._send(404, {"error": "not found"})
        except Exception as e:
            _log(f"error: {type(e).__name__}: {e}")
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        _log(fmt % args)


def _idle_reaper():
    while True:
        time.sleep(30)
        if IDLE_SECONDS <= 0:
            continue
        with _lock:
            if _state["model"] is not None and time.time() - _state["last_used"] > IDLE_SECONDS:
                _log(f"idle for {IDLE_SECONDS}s")
                _unload()


def main():
    ap = argparse.ArgumentParser(description="Qwen3-TTS worker (one-shot or service)")
    ap.add_argument("--serve", action="store_true", help="run as a local HTTP service")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.serve:
        threading.Thread(target=_idle_reaper, daemon=True).start()
        _log(f"serving on http://{args.host}:{args.port} (idle unload: {IDLE_SECONDS}s)")
        HTTPServer((args.host, args.port), Handler).serve_forever()
        return

    if not args.text or not args.out:
        ap.error("--text and --out are required in one-shot mode")
    result = generate({**vars(args), "unload_after": True})
    print(json.dumps(result))  # last stdout line is the result; the node parses it


if __name__ == "__main__":
    main()
