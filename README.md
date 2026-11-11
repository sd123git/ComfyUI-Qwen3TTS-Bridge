# ComfyUI-Qwen3TTS-Bridge

Run **Qwen3-TTS** inside ComfyUI without touching ComfyUI's own Python packages.

## Why a bridge?

Current ComfyUI installs ship **transformers 5.x**. Qwen3-TTS needs **transformers 4.57.3**:

- The official `qwen-tts` package does not import on transformers 5.
- Nodes that patch around this can load, but may produce garbled or incoherent audio.

A Python process can only load one version of a package, so both cannot live in ComfyUI's environment. This node keeps Qwen3-TTS in its **own virtual environment** and runs it as a separate process. The node is the bridge: you work in ComfyUI as usual, and it hands each job to the TTS environment and returns the audio.

Nothing is installed into ComfyUI's environment.

> **⚠️ Not a copy-and-paste install.**
> The commands below use placeholders. Replace them with your own paths:
>
> - `<COMFYUI>` — your ComfyUI folder, e.g. `/home/you/ComfyUI`
> - `<TTS_ENV>` — where the TTS environment will live, e.g. `/home/you/qwen3-tts-env`
>
> Developed and tested on Linux, with ComfyUI running as a systemd service, on an NVIDIA RTX 3060 (12 GB).

## Requirements

- Linux with an NVIDIA GPU and a working driver
- Python 3.10 or newer (`python3 --version`)
- About 10 GB of free disk space for the environment and one model
- ComfyUI with a CUDA build of PyTorch

## 1. Install the node

```
cd <COMFYUI>/custom_nodes
git clone https://github.com/sd123git/ComfyUI-Qwen3TTS-Bridge.git
```

## 2. Create the TTS environment

**Match ComfyUI's PyTorch.** Check which torch and CUDA version ComfyUI uses:

```
<COMFYUI>/venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Use the same torch version and CUDA index below. For example, `2.14.1+cu132` means torch `2.14.1` from the `cu132` index.

```
python3 -m venv <TTS_ENV>
<TTS_ENV>/bin/pip install torch==<VERSION> torchaudio --index-url https://download.pytorch.org/whl/<CUDA_INDEX>
<TTS_ENV>/bin/pip install qwen-tts transformers==4.57.3 soundfile
```

You don't need to activate the environment. Calling `<TTS_ENV>/bin/pip` by its full path installs into that environment.

Confirm torch landed in the right place:

```
<TTS_ENV>/bin/pip show torch | grep -E "Version|Location"
```

The location should be inside `<TTS_ENV>`.

## 3. Download a model

The node supports **CustomVoice** models (preset speakers). Models must go in `models/qwen-tts/Qwen/`, including the `Qwen/` level:

```
<TTS_ENV>/bin/hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir <COMFYUI>/models/qwen-tts/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

The 0.6B model is fast and fits easily on 12 GB cards. For higher quality, use `Qwen3-TTS-12Hz-1.7B-CustomVoice` the same way.

If `hf` is not found, use `<TTS_ENV>/bin/huggingface-cli download` with the same arguments.

## 4. Test the TTS environment on its own

Do this before touching ComfyUI. It shows the environment works.

**Before running:** stop ComfyUI if it is running, so the test doesn't compete for VRAM.

```
cd /tmp
HF_HUB_OFFLINE=1 <TTS_ENV>/bin/python - <<'EOF'
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
p = "<COMFYUI>/models/qwen-tts/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
m = Qwen3TTSModel.from_pretrained(p, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
print("speakers:", m.get_supported_speakers())
wavs, sr = m.generate_custom_voice(text="She said she would be here by noon, but the train was running late.",
                                   language="English", speaker="Ryan")
sf.write("qwen3-tts-test.wav", wavs[0], sr)
print("wrote /tmp/qwen3-tts-test.wav", sr)
EOF
```

Listen to `/tmp/qwen3-tts-test.wav`. It should be clear English speech. The printed speaker list shows the names you can use in the node.

A "flash-attn is not installed" warning is normal and harmless.

## 5. Point the node at your paths

The node needs two paths. Defaults are set at the top of `__init__.py`:

| Setting | What it is | How to change it |
|---|---|---|
| `QWEN_PY` | The TTS environment's Python | Set `QWEN_TTS_PYTHON=<TTS_ENV>/bin/python` in ComfyUI's environment, or edit the default |
| `DEFAULT_MODEL` | Path to the model folder | Use the node's `model_path` field, or edit the default |

If ComfyUI runs as a systemd service, add this to its `[Service]` section:

```
Environment=QWEN_TTS_PYTHON=<TTS_ENV>/bin/python
```

Then run `sudo systemctl daemon-reload` and restart ComfyUI.

## 6. Use it

Restart ComfyUI, then add **Qwen3-TTS Bridge (external venv)** from the node library. Connect its `AUDIO` output to a preview or save node.

| Input | Notes |
|---|---|
| `text` | What to say. Full sentences work much better than one or two words. |
| `speaker` | A CustomVoice speaker name, e.g. `Ryan` or `Vivian`. |
| `language` | Set it explicitly. `Auto` can give odd accents. |
| `mode` | `subprocess` or `service` (see below). |
| `unload_after` | Service mode: free VRAM after this run. |
| `free_comfy_vram_first` | Unload ComfyUI's own models before TTS runs, to make room. |
| `precision` | `bf16` is recommended. Use `fp32` only if bf16 misbehaves on your GPU. |
| `instruct` | Optional style hint, e.g. `Very happy.` |
| `model_path` | Model folder to load. |
| `service_url` | Service address, default `http://127.0.0.1:8190`. |

### Choosing a mode

**`subprocess` (default).** Each run starts the TTS environment, generates, and exits. VRAM is always freed afterwards. Expect roughly 10 seconds per run for model loading. Best for occasional TTS between other GPU jobs.

**`service`.** Runs Qwen3-TTS as a background service that keeps the model loaded, so repeated runs are fast. ComfyUI cannot unload a model held by another process, so the service frees its own VRAM:

- `unload_after` in the node frees it after a run.
- The model is released after 5 idle minutes. Change this with `QWEN_TTS_IDLE_SECONDS`; `0` disables it.
- `POST /unload` frees it on demand.

## Optional: run as a service (Linux, systemd)

Copy the example `qwen3-tts.service` to `/etc/systemd/system/`, then edit:

- `User=` to your username
- the `ExecStart=` paths to your `<TTS_ENV>` and `<COMFYUI>` folders

```
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-tts.service
curl -s http://127.0.0.1:8190/health
```

The health check should return `{"ok": true, ...}`. Set the node's `mode` to `service`.

The service uses `HF_HUB_OFFLINE=1`, so it never downloads anything. Models must already be in place (step 3).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `venv python not found` | `QWEN_TTS_PYTHON` or `QWEN_PY` doesn't point at `<TTS_ENV>/bin/python`. |
| `service not reachable` | The service isn't running. Check `systemctl status qwen3-tts.service`, or switch to `subprocess`. |
| Model not found, or offline errors | The model is not at `models/qwen-tts/Qwen/<model>`. Check the `Qwen/` level. |
| Garbled audio | Check that `<TTS_ENV>` has `transformers==4.57.3`. Try a full sentence with the language set explicitly. |
| Out of memory | Turn on `free_comfy_vram_first`, use the 0.6B model, or use `subprocess` mode. |
| Error shown in the node | The node shows the last lines of the worker's output. For service mode, see `journalctl -u qwen3-tts.service`. |

## Limitations

- CustomVoice models only. Voice cloning and voice design are not supported yet.
- One TTS job at a time in service mode.

## License

Node code: Apache 2.0 LICENSE, in this repository.

Qwen3-TTS model weights are covered by the Qwen team's own license. Review it before commercial use: https://github.com/QwenLM/Qwen3-TTS
