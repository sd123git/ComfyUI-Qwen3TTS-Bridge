"""
Qwen3-TTS Bridge for ComfyUI.

Runs Qwen3-TTS in a separate venv (transformers 4.57.x) so ComfyUI's own
transformers 5.x is never touched. The node only bridges: it sends the job to
tts_worker.py and returns the result as an AUDIO output.

  mode = subprocess : starts the worker per run; VRAM is always freed on exit.
  mode = service    : talks to a running worker service (qwen-tts.service);
                      the model stays loaded between runs unless unload_after is on.

Set QWEN_TTS_PYTHON in the ComfyUI service environment to change the venv's python.
"""
import json
import os
import subprocess
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np
import torch

import folder_paths

QWEN_PY = os.environ.get("QWEN_TTS_PYTHON", "<TTS_ENV>/bin/python")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_worker.py")
DEFAULT_MODEL = "/<ComfyUI>/models/qwen-tts/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_URL = "http://127.0.0.1:8190"
LANGUAGES = ["English", "Chinese", "Japanese", "Korean", "German", "French",
             "Russian", "Portuguese", "Spanish", "Italian", "Auto"]
TIMEOUT = 900


def _read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    audio = audio.reshape(-1, channels).T  # [C, T]
    return {"waveform": torch.from_numpy(audio.copy()).unsqueeze(0), "sample_rate": sr}


def _clean_env():
    # Keep ComfyUI's Python settings from leaking into the worker's venv.
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_CONSTRAINT")}
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


class QwenTTSBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True,
                                    "default": "She said she would be here by noon, but the train was running late."}),
                "speaker": ("STRING", {"default": "Ryan",
                                       "tooltip": "CustomVoice speaker name, e.g. Ryan, Vivian."}),
                "language": (LANGUAGES, {"default": "English"}),
                "mode": (["subprocess", "service"], {"default": "subprocess",
                         "tooltip": "subprocess: load per run, always frees VRAM. service: use the running qwen-tts service."}),
                "unload_after": ("BOOLEAN", {"default": True,
                                 "tooltip": "Service mode: free the TTS model's VRAM after this run. Subprocess mode always frees it."}),
                "free_comfy_vram_first": ("BOOLEAN", {"default": False,
                                          "tooltip": "Unload ComfyUI's own models before TTS runs, to make room on the GPU."}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": "",
                                        "tooltip": "Optional style instruction, e.g. 'Very happy.'"}),
                "model_path": ("STRING", {"default": DEFAULT_MODEL}),
                "service_url": ("STRING", {"default": DEFAULT_URL}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "audio/qwen-tts-bridge"

    def run(self, text, speaker, language, mode, unload_after, free_comfy_vram_first,
            precision, instruct="", model_path=DEFAULT_MODEL, service_url=DEFAULT_URL):
        if not text.strip():
            raise ValueError("Qwen TTS Bridge: text is empty")

        if free_comfy_vram_first:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        out = os.path.join(temp_dir, f"qwen_tts_{uuid.uuid4().hex}.wav")

        req = {
            "text": text, "speaker": speaker, "language": language, "instruct": instruct,
            "model": model_path, "dtype": precision, "attn": "sdpa",
            "out": out, "unload_after": unload_after,
        }
        if mode == "service":
            self._via_service(service_url, req)
        else:
            self._via_subprocess(req)

        if not os.path.exists(out):
            raise RuntimeError(f"Qwen TTS Bridge: worker reported success but {out} is missing")
        audio = _read_wav(out)
        try:
            os.remove(out)
        except OSError:
            pass
        return (audio,)

    @staticmethod
    def _via_subprocess(req):
        if not os.path.exists(QWEN_PY):
            raise RuntimeError(f"Qwen TTS Bridge: venv python not found at {QWEN_PY} (set QWEN_TTS_PYTHON)")
        cmd = [QWEN_PY, WORKER,
               "--text", req["text"], "--speaker", req["speaker"], "--language", req["language"],
               "--model", req["model"], "--dtype", req["dtype"], "--attn", req["attn"],
               "--out", req["out"]]
        if req["instruct"]:
            cmd += ["--instruct", req["instruct"]]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=_clean_env())
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise RuntimeError(f"Qwen TTS Bridge: worker failed (exit {proc.returncode}):\n{tail}")

    @staticmethod
    def _via_service(url, req):
        data = json.dumps(req).encode()
        request = urllib.request.Request(url.rstrip("/") + "/generate", data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
                json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("error", "")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"Qwen TTS Bridge: service error: {detail}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Qwen TTS Bridge: service not reachable at {url} ({e.reason}). "
                               f"Start qwen-tts.service or switch mode to subprocess.")


NODE_CLASS_MAPPINGS = {"QwenTTSBridge": QwenTTSBridge}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenTTSBridge": "Qwen3-TTS Bridge (external venv)"}
