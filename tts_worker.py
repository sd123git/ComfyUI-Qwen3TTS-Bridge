#!/usr/bin/env python3
"""
Qwen3-TTS worker. Runs in its OWN venv (transformers 4.57.x), never inside ComfyUI.

Two modes, same generation code:
  one-shot:  python tts_worker.py --job job.json           (how the ComfyUI nodes call it)
             python tts_worker.py --task custom --model <dir> --text "..." --out out.wav
             python tts_worker.py --list-speakers --model <CustomVoice dir>
  service:   python tts_worker.py --serve [--host 127.0.0.1 --port 8190]

Tasks and the model type each needs:
  custom    -> *-CustomVoice   preset speakers
  design    -> *-VoiceDesign   voice from a text description
  clone     -> *-Base          voice from reference audio
  dialogue  -> *-Base          many lines, each cloned from its own reference, one model load

Service endpoints:
  GET  /health    -> {"ok": true, "loaded": bool, "model": path|null, "cached_voices": n}
  POST /generate  -> the same JSON job the --job file holds
  POST /unload    -> frees the model's VRAM

Sampling keys passed through to generation: seed, max_new_tokens, top_p, top_k,
temperature, repetition_penalty.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

IDLE_SECONDS = int(os.environ.get("QWEN_TTS_IDLE_SECONDS", "300"))  # 0 disables idle unload
TASKS = ("custom", "design", "clone", "dialogue")
SAMPLING_KEYS = ("max_new_tokens", "top_p", "top_k", "temperature", "repetition_penalty")
MODEL_KIND = {"custom": "customvoice", "design": "voicedesign", "clone": "base", "dialogue": "base"}

_lock = threading.Lock()
_state = {"model": None, "key": None, "last_used": 0.0, "prompts": {}}


def _log(msg):
    sys.stderr.write(f"[qwen3-tts] {msg}\n")
    sys.stderr.flush()


# ---------- model lifecycle ----------

def _unload():
    if _state["model"] is None:
        return False
    _state["model"] = None
    _state["key"] = None
    _state["prompts"] = {}
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _log("model unloaded")
    return True


def _load(model_path, dtype_name, attn, task):
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(f"model folder not found: {model_path}")
    kind = MODEL_KIND[task]
    name = os.path.basename(model_path.rstrip("/")).lower()
    if (kind == "base" and not name.endswith("-base")) or (kind != "base" and kind not in name):
        _log(f"warning: task '{task}' expects a {kind} model, got {os.path.basename(model_path)}")

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


# ---------- helpers ----------

def _seed(seed):
    if seed is None or int(seed) < 0:
        return
    import random
    import numpy as np
    import torch
    s = int(seed)
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s % (2 ** 63))


def _gen_kwargs(req):
    return {k: req[k] for k in SAMPLING_KEYS if req.get(k) is not None}


def _file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _audio_seconds(path):
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return 0.0


def _clone_kwargs(model, ref_audio, ref_text, x_only):
    """Arguments for generate_voice_clone. Builds the reusable voice prompt once per
    reference and caches it until the model is unloaded."""
    if not ref_audio or not os.path.exists(ref_audio):
        raise ValueError(f"reference audio not found: {ref_audio}")
    ref_text = (ref_text or "").strip()
    x_only = bool(x_only)
    if not ref_text and not x_only:
        raise ValueError("voice clone needs ref_text (the words spoken in the reference), "
                         "or x_vector_only")

    direct = {"ref_audio": ref_audio, "x_vector_only_mode": x_only}
    if ref_text:
        direct["ref_text"] = ref_text
    if not hasattr(model, "create_voice_clone_prompt"):
        return direct

    key = (_file_hash(ref_audio), ref_text, x_only)
    prompt = _state["prompts"].get(key)
    if prompt is None:
        prompt = model.create_voice_clone_prompt(**direct)
        _state["prompts"][key] = prompt
        _log(f"voice prompt built from {os.path.basename(ref_audio)} "
             f"({_audio_seconds(ref_audio):.1f}s reference)")
    return {"voice_clone_prompt": prompt}


def _write(out, wav, sr):
    import soundfile as sf
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sf.write(out, wav, sr, subtype="PCM_16")  # PCM_16 so the node can read it with stdlib wave
    return len(wav) / float(sr)


# ---------- generation ----------

def _run(model, req):
    task = req["task"]
    language = req.get("language") or "English"
    gk = _gen_kwargs(req)
    _seed(req.get("seed"))

    if task == "dialogue":
        lines = req.get("lines") or []
        out_dir = req.get("out_dir")
        if not lines:
            raise ValueError("dialogue has no lines")
        if not out_dir:
            raise ValueError("out_dir is required for dialogue")
        paths, total, sr = [], 0.0, 0
        for i, ln in enumerate(lines):
            text = (ln.get("text") or "").strip()
            if not text:
                raise ValueError(f"dialogue line {i + 1} is empty")
            ck = _clone_kwargs(model, ln.get("ref_audio"), ln.get("ref_text"), ln.get("x_vector_only"))
            wavs, sr = model.generate_voice_clone(text=text, language=language, **ck, **gk)
            path = os.path.join(out_dir, f"line_{i:04d}.wav")
            total += _write(path, wavs[0], sr)
            paths.append(path)
        return {"paths": paths, "sample_rate": int(sr), "seconds": total}

    text = (req.get("text") or "").strip()
    out = req.get("out")
    if not text:
        raise ValueError("text is empty")
    if not out:
        raise ValueError("out path is required")
    instruct = (req.get("instruct") or "").strip()

    if task == "custom":
        kw = {"text": text, "language": language, "speaker": req.get("speaker") or "Ryan"}
        if instruct:
            kw["instruct"] = instruct
        wavs, sr = model.generate_custom_voice(**kw, **gk)
    elif task == "design":
        if not instruct:
            raise ValueError("voice design needs a voice description")
        wavs, sr = model.generate_voice_design(text=text, language=language, instruct=instruct, **gk)
    else:  # clone
        ck = _clone_kwargs(model, req.get("ref_audio"), req.get("ref_text"), req.get("x_vector_only"))
        wavs, sr = model.generate_voice_clone(text=text, language=language, **ck, **gk)

    seconds = _write(out, wavs[0], sr)
    return {"path": out, "sample_rate": int(sr), "seconds": seconds}


def generate(req):
    import torch

    task = req.get("task") or "custom"
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}' (expected one of {TASKS})")
    req["task"] = task
    model = _load(req.get("model"), req.get("dtype") or "bf16", req.get("attn") or "sdpa", task)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        result = _run(model, req)
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        _unload()
        raise RuntimeError(
            f"CUDA out of memory during {task} (peak {peak:.1f} GiB); model unloaded. "
            "Check that ref_text matches the reference audio word for word, lower "
            "max_new_tokens, turn on free_comfy_vram_first, or try a 0.6B model.")
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    _state["last_used"] = time.time()
    _log(f"{task}: {result.get('seconds', 0):.1f}s of audio in {time.time() - t0:.1f}s, "
         f"peak VRAM {peak:.1f} GiB")

    if req.get("unload_after"):
        _unload()
    return result


# ---------- service ----------

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
            key = _state["key"]
            return self._send(200, {"ok": True, "loaded": _state["model"] is not None,
                                    "model": key[0] if key else None,
                                    "cached_voices": len(_state["prompts"])})
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
    ap.add_argument("--job", help="JSON job file (what the ComfyUI nodes send)")
    ap.add_argument("--list-speakers", action="store_true", help="print a CustomVoice model's speakers")
    ap.add_argument("--task", default="custom", choices=TASKS[:3])
    ap.add_argument("--model", help="model folder")
    ap.add_argument("--text")
    ap.add_argument("--language", default="English")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--instruct", default="")
    ap.add_argument("--ref-audio", dest="ref_audio")
    ap.add_argument("--ref-text", dest="ref_text", default="")
    ap.add_argument("--x-vector-only", dest="x_vector_only", action="store_true")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.serve:
        threading.Thread(target=_idle_reaper, daemon=True).start()
        _log(f"serving on http://{args.host}:{args.port} (idle unload: {IDLE_SECONDS}s)")
        HTTPServer((args.host, args.port), Handler).serve_forever()
        return

    if args.job:
        with open(args.job, encoding="utf-8") as f:
            req = json.load(f)
        req["unload_after"] = True
        print(json.dumps(generate(req)))  # last stdout line is the result
        return

    if args.list_speakers:
        model = _load(args.model, args.dtype, args.attn, "custom")
        print(json.dumps(model.get_supported_speakers()))
        return

    if not args.model or not args.text or not args.out:
        ap.error("--model, --text and --out are required (or use --job)")
    print(json.dumps(generate({**vars(args), "unload_after": True})))


if __name__ == "__main__":
    main()
