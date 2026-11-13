"""
Qwen3-TTS Bridge for ComfyUI.

Runs Qwen3-TTS in a separate venv (transformers 4.57.x) so ComfyUI's own
transformers 5.x is never touched. The nodes only bridge: they send jobs to
tts_worker.py and return AUDIO.

Generation nodes
  Qwen3-TTS Bridge (external venv)  Custom Voice: preset speakers        (*-CustomVoice)
  Qwen3-TTS Bridge: Voice Design     voice from a description             (*-VoiceDesign)
  Qwen3-TTS Bridge: Voice Clone      voice from reference audio           (*-Base)
  Qwen3-TTS Bridge: Dialogue         multi-character script with pauses   (*-Base)

Voice nodes
  Qwen3-TTS Bridge: Make Voice   reference audio + its transcript -> VOICE (optionally saved)
  Qwen3-TTS Bridge: Load Voice   saved voice from models/qwen-tts/voices -> VOICE
  Qwen3-TTS Bridge: Role Bank    up to 8 named VOICEs -> ROLE BANK for Dialogue

Modes
  subprocess : starts the worker per run; VRAM is always freed on exit.
  service    : uses the running qwen3-tts service; the model stays loaded between runs
               unless unload_after is on.

Models: <ComfyUI>/models/qwen-tts/Qwen/<model folder>
Voices: <ComfyUI>/models/qwen-tts/voices/<name>.wav + <name>.txt (transcript)
Set QWEN_TTS_PYTHON in ComfyUI's environment to the TTS venv's python.
"""
import hashlib
import json
import os
import re
import shutil
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
QWEN_DIR = os.path.join(folder_paths.models_dir, "qwen-tts")
MODELS_DIR = os.path.join(QWEN_DIR, "Qwen")
VOICES_DIR = os.path.join(QWEN_DIR, "voices")
DEFAULT_URL = "http://127.0.0.1:8190"
TIMEOUT = 3600
NONE_FOUND = "(no matching model found)"
NO_VOICES = "(no saved voices)"

VOICE_TYPE = "QWEN3_BRIDGE_VOICE"
ROLE_BANK_TYPE = "QWEN3_BRIDGE_ROLE_BANK"
CATEGORY = "audio/qwen3-tts-bridge"

LANGUAGES = ["English", "Chinese", "Japanese", "Korean", "German", "French",
             "Russian", "Portuguese", "Spanish", "Italian", "Auto"]
# Preset speakers of the official CustomVoice models (model card).
SPEAKERS = ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
            "Ono_Anna", "Sohee", "custom"]
SAMPLING_KEYS = ("max_new_tokens", "top_p", "top_k", "temperature", "repetition_penalty")
DEFAULT_TEXT = "She said she would be here by noon, but the train was running late."


# ---------- models and voices on disk ----------

def _list_models(task):
    def match(name):
        n = name.lower()
        if task == "custom":
            return "customvoice" in n
        if task == "design":
            return "voicedesign" in n
        return n.endswith("-base")
    try:
        names = [d for d in os.listdir(MODELS_DIR)
                 if os.path.isdir(os.path.join(MODELS_DIR, d)) and match(d)]
    except FileNotFoundError:
        names = []
    names.sort(key=lambda d: (0 if "1.7b" in d.lower() else 1, d))  # larger model first
    return names or [NONE_FOUND]


def _model_dir(model, model_path=""):
    if model_path and model_path.strip():
        return model_path.strip()
    if not model or model == NONE_FOUND:
        raise ValueError(f"Qwen3-TTS Bridge: no matching model found in {MODELS_DIR}")
    return os.path.join(MODELS_DIR, model)


def _list_voices():
    try:
        names = sorted(f for f in os.listdir(VOICES_DIR) if f.lower().endswith(".wav"))
    except FileNotFoundError:
        names = []
    return names or [NO_VOICES]


def _safe_name(name):
    name = re.sub(r"[^\w\- ]", "_", (name or "").strip()).strip()
    return name or "voice"


# ---------- audio helpers ----------

def _temp_dir():
    d = folder_paths.get_temp_directory()
    os.makedirs(d, exist_ok=True)
    return d


def _load_wav_np(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, channels).T, sr  # [C, T]


def _to_audio(data, sr):
    return {"waveform": torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0),
            "sample_rate": int(sr)}


def _prepare_ref(audio, max_seconds):
    """ComfyUI AUDIO -> mono int16 PCM, trimmed to max_seconds."""
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0]
    mono = wav.float().mean(dim=0).clamp(-1.0, 1.0).cpu().numpy()
    sr = int(audio["sample_rate"])
    if max_seconds and max_seconds > 0:
        mono = mono[: int(max_seconds * sr)]
    return (mono * 32767.0).astype(np.int16), sr


def _save_pcm(pcm, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _next_name(base):
    i = 1
    while os.path.exists(os.path.join(VOICES_DIR, f"{base}_{i:02d}.wav")):
        i += 1
    return f"{base}_{i:02d}"


def _voice_from_audio(audio, ref_text, x_vector_only, max_seconds, name="", save=False, increment=False):
    pcm, sr = _prepare_ref(audio, max_seconds)
    final = _safe_name(name)
    if save:
        os.makedirs(VOICES_DIR, exist_ok=True)
        if increment:
            final = _next_name(final)
        base = os.path.join(VOICES_DIR, final)
        path = base + ".wav"
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(ref_text.strip())
    else:
        digest = hashlib.sha1(pcm.tobytes() + str(sr).encode()).hexdigest()[:16]
        path = os.path.join(_temp_dir(), f"qwen3_voice_{digest}.wav")
    _save_pcm(pcm, sr, path)
    return {"name": final, "ref_audio": path, "ref_text": ref_text.strip(),
            "x_vector_only": bool(x_vector_only), "seconds": len(pcm) / float(sr)}


# ---------- dispatch to the worker ----------

def _clean_env():
    # Keep ComfyUI's Python settings from leaking into the worker's venv.
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_CONSTRAINT")}
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _via_subprocess(req):
    if not os.path.exists(QWEN_PY):
        raise RuntimeError(f"Qwen3-TTS Bridge: venv python not found at {QWEN_PY} (set QWEN_TTS_PYTHON)")
    job = os.path.join(_temp_dir(), f"qwen3_job_{uuid.uuid4().hex}.json")
    with open(job, "w", encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False)
    try:
        proc = subprocess.run([QWEN_PY, WORKER, "--job", job], capture_output=True, text=True,
                              timeout=TIMEOUT, env=_clean_env())
    finally:
        try:
            os.remove(job)
        except OSError:
            pass
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"Qwen3-TTS Bridge: worker failed (exit {proc.returncode}):\n{tail}")
    lines = [l for l in (proc.stdout or "").splitlines() if l.strip()]
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError("Qwen3-TTS Bridge: could not read the worker's result:\n"
                           + (proc.stdout or "")[-1000:])


def _via_service(url, req):
    data = json.dumps(req, ensure_ascii=False).encode()
    request = urllib.request.Request(url.rstrip("/") + "/generate", data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("error", "")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"Qwen3-TTS Bridge: service error: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Qwen3-TTS Bridge: service not reachable at {url} ({e.reason}). "
                           f"Start qwen3-tts.service or switch mode to subprocess.")


def _dispatch(req, kw):
    if kw.get("free_comfy_vram_first"):
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    if kw.get("mode") == "service":
        return _via_service(kw.get("service_url") or DEFAULT_URL, req)
    return _via_subprocess(req)


def _base_req(task, kw):
    req = {
        "task": task,
        "model": _model_dir(kw.get("model"), kw.get("model_path", "")),
        "language": kw.get("language", "English"),
        "dtype": kw.get("precision", "bf16"),
        "attn": "sdpa",
        "unload_after": kw.get("unload_after", True),
        "seed": kw.get("seed", 0),
    }
    for k in SAMPLING_KEYS:
        if kw.get(k) is not None:
            req[k] = kw[k]
    return req


def _run_segments(segments, kw, line_pause, speaker_change_pause):
    """Generate cloned segments in one worker job and lay them on a timeline.

    segments: [(line_index, role, text, pause_after_within_line, voice)]
    Returns (mix [T], {role: stem [T]}, sample_rate). Stems are time-aligned with the mix,
    silent wherever another role speaks.
    """
    out_dir = os.path.join(_temp_dir(), f"qwen3_segments_{uuid.uuid4().hex}")
    os.makedirs(out_dir, exist_ok=True)
    req = _base_req("dialogue", kw)
    req["out_dir"] = out_dir
    req["lines"] = [{"text": t, "ref_audio": v["ref_audio"], "ref_text": v["ref_text"],
                     "x_vector_only": v["x_vector_only"]} for (_, _, t, _, v) in segments]
    try:
        res = _dispatch(req, kw)
        paths = res.get("paths") or []
        if len(paths) != len(segments):
            raise RuntimeError(f"Qwen3-TTS Bridge: expected {len(segments)} segments, got {len(paths)}")
        timeline, cursor, sr = [], 0, None
        for k, (li, role, _, pause, _) in enumerate(segments):
            data, s = _load_wav_np(paths[k])
            sr = sr or s
            mono = data.mean(axis=0)
            timeline.append((cursor, mono, role))
            cursor += len(mono)
            if k + 1 < len(segments):
                nli, nrole = segments[k + 1][0], segments[k + 1][1]
                gap = pause if nli == li else (speaker_change_pause if nrole != role else line_pause)
                cursor += int(round(max(gap, 0.0) * sr))
        mix = np.zeros(cursor, dtype=np.float32)
        stems = {}
        for start, mono, role in timeline:
            mix[start:start + len(mono)] += mono
            stems.setdefault(role, np.zeros(cursor, dtype=np.float32))[start:start + len(mono)] += mono
        return mix, stems, sr
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _single(task, kw, **fields):
    if not (kw.get("text") or "").strip():
        raise ValueError("Qwen3-TTS Bridge: text is empty")
    out = os.path.join(_temp_dir(), f"qwen3_tts_{uuid.uuid4().hex}.wav")
    req = _base_req(task, kw)
    req.update(text=kw["text"], out=out, **fields)
    try:
        _dispatch(req, kw)
        if not os.path.exists(out):
            raise RuntimeError(f"Qwen3-TTS Bridge: worker reported success but {out} is missing")
        return _to_audio(*_load_wav_np(out))
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


# ---------- shared inputs ----------

def _model_input(task):
    models = _list_models(task)
    kind = {"custom": "CustomVoice", "design": "VoiceDesign"}.get(task, "Base")
    return (models, {"default": models[0],
                     "tooltip": f"{kind} models found in {MODELS_DIR}. 1.7B: better quality; 0.6B: faster, less VRAM."})


def _run_inputs():
    return {
        "language": (LANGUAGES, {"default": "English",
                                 "tooltip": "Set it explicitly; Auto can give odd accents."}),
        "mode": (["subprocess", "service"], {"default": "subprocess",
                 "tooltip": "subprocess: load per run, always frees VRAM. service: use the running qwen3-tts service."}),
        "unload_after": ("BOOLEAN", {"default": True,
                         "tooltip": "Service mode: free the TTS model's VRAM after this run. Subprocess mode always frees it."}),
        "free_comfy_vram_first": ("BOOLEAN", {"default": False,
                                  "tooltip": "Unload ComfyUI's own models before TTS runs, to make room on the GPU."}),
        "precision": (["bf16", "fp32"], {"default": "bf16"}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True,
                         "tooltip": "Fixed seed = repeatable delivery for the same text and voice."}),
    }


def _sampling_inputs():
    return {
        "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64,
                                   "tooltip": "Cap on generated audio tokens (about 12 per second of speech). "
                                              "Stops runaway generation."}),
        "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
        "top_k": ("INT", {"default": 20, "min": 0, "max": 200}),
        "temperature": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 2.0, "step": 0.05,
                                  "tooltip": "Lower = steadier, flatter delivery. Higher = more varied."}),
        "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
        "model_path": ("STRING", {"default": "", "tooltip": "Overrides the model dropdown when set."}),
        "service_url": ("STRING", {"default": DEFAULT_URL}),
    }


# ---------- generation nodes ----------

class QwenTTSBridge:
    """Custom Voice: preset speakers. Class name kept from v0.1 so saved workflows still find it."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "text": ("STRING", {"multiline": True, "default": DEFAULT_TEXT}),
            "speaker": (SPEAKERS, {"default": "Ryan",
                                   "tooltip": "Preset speaker. Pick 'custom' to type a fine-tuned speaker name below."}),
            "model": _model_input("custom"),
        }
        req.update(_run_inputs())
        opt = {
            "instruct": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Optional style instruction, e.g. 'Very happy.'"}),
            "custom_speaker": ("STRING", {"default": "", "tooltip": "Used when speaker is 'custom'."}),
        }
        opt.update(_sampling_inputs())
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kw):
        speaker = kw.get("speaker", "Ryan")
        if speaker == "custom":
            speaker = (kw.get("custom_speaker") or "").strip()
            if not speaker:
                raise ValueError("Qwen3-TTS Bridge: speaker is 'custom' but custom_speaker is empty")
        return (_single("custom", kw, speaker=speaker, instruct=kw.get("instruct", "")),)


class QwenTTSBridgeVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "text": ("STRING", {"multiline": True, "default": DEFAULT_TEXT}),
            "voice_description": ("STRING", {"multiline": True,
                                             "default": "A calm, warm middle-aged male voice with a slight rasp.",
                                             "tooltip": "Describe the voice: age, gender, tone, pace, accent, emotion."}),
            "model": _model_input("design"),
        }
        req.update(_run_inputs())
        return {"required": req, "optional": _sampling_inputs()}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "text")
    OUTPUT_TOOLTIPS = ("Generated speech.",
                       "The text spoken, for wiring straight into Make Voice as the transcript.")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kw):
        desc = (kw.get("voice_description") or "").strip()
        if not desc:
            raise ValueError("Qwen3-TTS Bridge: voice_description is empty")
        return (_single("design", kw, instruct=desc), kw.get("text", ""))


class QwenTTSBridgeVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "text": ("STRING", {"multiline": True, "default": DEFAULT_TEXT}),
            "model": _model_input("clone"),
        }
        req.update(_run_inputs())
        opt = {
            "voice": (VOICE_TYPE, {"tooltip": "From Make Voice or Load Voice. Takes priority over ref_audio."}),
            "ref_audio": ("AUDIO", {"tooltip": "5-15 s of clean speech, if no voice is connected."}),
            "ref_text": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "The exact words spoken in ref_audio."}),
            "x_vector_only": ("BOOLEAN", {"default": False,
                                          "tooltip": "Speaker embedding only; no ref_text needed, lower quality."}),
            "ref_max_seconds": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 60.0, "step": 0.5,
                                          "tooltip": "Trim ref_audio to this length. 0 = no trim."}),
        }
        opt.update(_sampling_inputs())
        # Appended last so saved v0.3 workflows keep their widget positions.
        opt.update({
            "line_pause": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.05,
                                     "tooltip": "If > 0, each line of text is generated separately with this silence between."}),
            "sentence_pause": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                         "tooltip": "If > 0, split at sentence ends and add this silence."}),
            "comma_pause": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                      "tooltip": "If > 0, also split at commas and add this silence."}),
        })
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kw):
        voice = kw.get("voice")
        if voice is None:
            if kw.get("ref_audio") is None:
                raise ValueError("Qwen3-TTS Bridge: connect a voice, or ref_audio with its ref_text")
            if not kw.get("x_vector_only") and not (kw.get("ref_text") or "").strip():
                raise ValueError("Qwen3-TTS Bridge: ref_text is empty. Enter the words spoken in "
                                 "ref_audio, or turn on x_vector_only.")
            voice = _voice_from_audio(kw["ref_audio"], kw.get("ref_text", ""), kw.get("x_vector_only", False),
                                      kw.get("ref_max_seconds", 15.0))

        lp, sp, cp = kw.get("line_pause", 0.0), kw.get("sentence_pause", 0.0), kw.get("comma_pause", 0.0)
        if lp <= 0 and sp <= 0 and cp <= 0:
            return (_single("clone", kw, ref_audio=voice["ref_audio"], ref_text=voice["ref_text"],
                            x_vector_only=voice["x_vector_only"]),)

        lines = [l.strip() for l in (kw.get("text") or "").splitlines() if l.strip()]
        if not lines:
            raise ValueError("Qwen3-TTS Bridge: text is empty")
        segments = [(li, "voice", t, p, voice)
                    for li, line in enumerate(lines) for t, p in _split_fragments(line, sp, cp)]
        mix, _, sr = _run_segments(segments, kw, lp, lp)
        return (_to_audio(mix[None, :], sr),)


# ---------- voice nodes ----------

class QwenTTSBridgeMakeVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Clean speech, ideally 5-15 s. Voice Design output works well."}),
                "ref_text": ("STRING", {"multiline": True, "default": "",
                                        "tooltip": "The exact words spoken in the audio."}),
                "name": ("STRING", {"default": "my_voice"}),
                "save_to_library": ("BOOLEAN", {"default": False,
                                    "tooltip": f"Save to {VOICES_DIR} for Load Voice."}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "ref_max_seconds": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 60.0, "step": 0.5,
                                              "tooltip": "Trim to this length. 0 = no trim."}),
                "on_existing": (["increment", "overwrite"], {"default": "increment",
                                "tooltip": "increment: save as name_01, name_02, ... overwrite: replace name.wav."}),
            }
        }

    RETURN_TYPES = (VOICE_TYPE, "STRING")
    RETURN_NAMES = ("voice", "saved_name")
    OUTPUT_NODE = True  # runs on its own, without a downstream node
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, audio, ref_text, name, save_to_library, x_vector_only, ref_max_seconds,
            on_existing="increment"):
        if not x_vector_only and not ref_text.strip():
            raise ValueError("Qwen3-TTS Bridge: ref_text is empty. Enter the words spoken in the audio, "
                             "or turn on x_vector_only.")
        voice = _voice_from_audio(audio, ref_text, x_vector_only, ref_max_seconds, name, save_to_library,
                                  increment=(on_existing == "increment"))
        print(f"[Qwen3-TTS Bridge] voice '{voice['name']}': {voice['seconds']:.1f}s reference"
              + (f", saved to {voice['ref_audio']}" if save_to_library else ""))
        return (voice, voice["name"] if save_to_library else "")


class QwenTTSBridgeLoadVoice:
    @classmethod
    def INPUT_TYPES(cls):
        voices = _list_voices()
        return {"required": {"voice_file": (voices, {"default": voices[0],
                             "tooltip": f"Saved voices in {VOICES_DIR}. A .txt with the same name holds the "
                                        "transcript. Newly saved voices appear after a refresh (press R)."})}}

    RETURN_TYPES = (VOICE_TYPE, "AUDIO", "STRING")
    RETURN_NAMES = ("voice", "audio", "ref_text")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, voice_file):
        path = os.path.join(VOICES_DIR, voice_file)
        txt = os.path.splitext(path)[0] + ".txt"
        return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (path, txt))

    def run(self, voice_file):
        if voice_file == NO_VOICES:
            raise ValueError(f"Qwen3-TTS Bridge: no saved voices in {VOICES_DIR}")
        path = os.path.join(VOICES_DIR, voice_file)
        txt = os.path.splitext(path)[0] + ".txt"
        ref_text = ""
        if os.path.exists(txt):
            with open(txt, encoding="utf-8") as f:
                ref_text = f.read().strip()
        x_only = not ref_text
        if x_only:
            print(f"[Qwen3-TTS Bridge] {voice_file}: no transcript found, using x_vector_only")
        data, sr = _load_wav_np(path)
        voice = {"name": os.path.splitext(voice_file)[0], "ref_audio": path, "ref_text": ref_text,
                 "x_vector_only": x_only, "seconds": data.shape[1] / float(sr)}
        return (voice, _to_audio(data, sr), ref_text)


class QwenTTSBridgeRoleBank:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for i in range(1, 9):
            opt[f"role_name_{i}"] = ("STRING", {"default": f"Role{i}"})
            opt[f"voice_{i}"] = (VOICE_TYPE,)
        return {"required": {}, "optional": opt}

    RETURN_TYPES = (ROLE_BANK_TYPE,)
    RETURN_NAMES = ("role_bank",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kw):
        bank = {}
        for i in range(1, 9):
            voice = kw.get(f"voice_{i}")
            if voice is None:
                continue
            name = (kw.get(f"role_name_{i}") or f"Role{i}").strip()
            if name in bank:
                raise ValueError(f"Qwen3-TTS Bridge: role name '{name}' is used twice")
            bank[name] = {**voice, "slot": i}
        if not bank:
            raise ValueError("Qwen3-TTS Bridge: connect at least one voice to the role bank")
        return (bank,)


# ---------- dialogue ----------

_ROLE_RE = re.compile(r"^\s*([^:：]{1,40}?)\s*[:：]\s*(.+)$")
_SENT_END = ".!?。！？…"
_CJK_SENT = "。！？"
_COMMA = ",;，；、"
_CJK_COMMA = "，；、"


def _parse_script(script, bank):
    items, prev = [], None
    for n, raw in enumerate(script.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = _ROLE_RE.match(line)
        if m and m.group(1).strip() in bank:
            prev = m.group(1).strip()
            items.append((prev, m.group(2).strip()))
        elif prev is not None:
            if m:
                print(f"[Qwen3-TTS Bridge] script line {n}: '{m.group(1).strip()}' is not in the role bank; "
                      f"spoken by {prev}")
            items.append((prev, line))
        else:
            raise ValueError(f"Qwen3-TTS Bridge: script line {n} doesn't start with a known role. "
                             f"Roles: {', '.join(bank)}")
    if not items:
        raise ValueError("Qwen3-TTS Bridge: script is empty")
    return items


def _split_fragments(text, sentence_pause, comma_pause):
    """Split a line at sentence ends / commas when their pause is > 0.
    Returns [(fragment, pause_after)], last pause 0."""
    if sentence_pause <= 0 and comma_pause <= 0:
        return [(text, 0.0)]
    frags, buf = [], ""
    for i, ch in enumerate(text):
        buf += ch
        nxt = text[i + 1] if i + 1 < len(text) else ""
        ends = nxt == "" or nxt.isspace()
        pause = None
        if sentence_pause > 0 and ch in _SENT_END and nxt not in _SENT_END and (ch in _CJK_SENT or ends):
            pause = sentence_pause
        elif comma_pause > 0 and ch in _COMMA and (ch in _CJK_COMMA or ends):
            pause = comma_pause
        if pause is not None and buf.strip():
            frags.append((buf.strip(), pause))
            buf = ""
    if buf.strip():
        frags.append((buf.strip(), 0.0))
    if frags:
        frags[-1] = (frags[-1][0], 0.0)
    return frags or [(text, 0.0)]


class QwenTTSBridgeDialogue:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "role_bank": (ROLE_BANK_TYPE,),
            "script": ("STRING", {"multiline": True,
                                  "default": "Role1: Hello there. How was the trip?\nRole2: Long. But worth it.",
                                  "tooltip": "One line per turn: 'RoleName: text'. Lines without a known role "
                                             "continue the previous speaker."}),
            "model": _model_input("clone"),
            "line_pause": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.05,
                                     "tooltip": "Silence between lines by the same speaker (seconds)."}),
            "speaker_change_pause": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.05,
                                               "tooltip": "Silence when the speaker changes (seconds)."}),
            "sentence_pause": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                         "tooltip": "If > 0, split lines at sentence ends and add this silence. "
                                                    "Splitting can change the flow between sentences."}),
            "comma_pause": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
                                      "tooltip": "If > 0, also split at commas and add this silence."}),
        }
        req.update(_run_inputs())
        return {"required": req, "optional": _sampling_inputs()}

    RETURN_TYPES = ("AUDIO",) + ("AUDIO",) * 8 + ("STRING",)
    RETURN_NAMES = ("audio",) + tuple(f"stem_{i}" for i in range(1, 9)) + ("roles",)
    OUTPUT_TOOLTIPS = (("Full mixed dialogue.",)
                       + tuple(f"Role bank slot {i} only: same length as the mix, silent while others speak."
                               for i in range(1, 9))
                       + ("Which role is on which stem.",))
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kw):
        bank = kw["role_bank"]
        items = _parse_script(kw.get("script", ""), bank)
        segments = []  # (line index, role, text, pause after within the line, voice)
        for li, (role, text) in enumerate(items):
            for t, p in _split_fragments(text, kw.get("sentence_pause", 0.0), kw.get("comma_pause", 0.0)):
                segments.append((li, role, t, p, bank[role]))
        print(f"[Qwen3-TTS Bridge] dialogue: {len(items)} lines, {len(segments)} segments, "
              f"{len(set(s[1] for s in segments))} voices")

        mix, stems, sr = _run_segments(segments, kw, kw.get("line_pause", 0.4),
                                       kw.get("speaker_change_pause", 0.8))
        by_slot = {v.get("slot"): name for name, v in bank.items()}
        silent = np.zeros_like(mix)
        outs, labels = [], []
        for i in range(1, 9):
            name = by_slot.get(i)
            outs.append(_to_audio(stems.get(name, silent)[None, :], sr))
            if name:
                labels.append(f"stem_{i}: {name}")
        return (_to_audio(mix[None, :], sr), *outs, "\n".join(labels))


NODE_CLASS_MAPPINGS = {
    "QwenTTSBridge": QwenTTSBridge,
    "QwenTTSBridgeVoiceDesign": QwenTTSBridgeVoiceDesign,
    "QwenTTSBridgeVoiceClone": QwenTTSBridgeVoiceClone,
    "QwenTTSBridgeMakeVoice": QwenTTSBridgeMakeVoice,
    "QwenTTSBridgeLoadVoice": QwenTTSBridgeLoadVoice,
    "QwenTTSBridgeRoleBank": QwenTTSBridgeRoleBank,
    "QwenTTSBridgeDialogue": QwenTTSBridgeDialogue,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTSBridge": "Qwen3-TTS Bridge (external venv)",
    "QwenTTSBridgeVoiceDesign": "Qwen3-TTS Bridge: Voice Design",
    "QwenTTSBridgeVoiceClone": "Qwen3-TTS Bridge: Voice Clone",
    "QwenTTSBridgeMakeVoice": "Qwen3-TTS Bridge: Make Voice",
    "QwenTTSBridgeLoadVoice": "Qwen3-TTS Bridge: Load Voice",
    "QwenTTSBridgeRoleBank": "Qwen3-TTS Bridge: Role Bank",
    "QwenTTSBridgeDialogue": "Qwen3-TTS Bridge: Dialogue",
}
