<div align="center">

# 🎙️ Voice Cloner

**Free, open-source voice cloning for Urdu and English.** Clone a voice from a 10-second clip, stream it in real time, or train it on your own recordings.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qasimrazamalik-coder/voice-cloner/blob/main/notebooks/Voice_Cloner_Colab.ipynb)
[![tests](https://github.com/qasimrazamalik-coder/voice-cloner/actions/workflows/tests.yml/badge.svg)](https://github.com/qasimrazamalik-coder/voice-cloner/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

</div>

Voice Cloner wraps **[OmniVoice](https://github.com/k2-fsa/OmniVoice)**, an open zero-shot TTS model covering 646 languages (Urdu included, with about 211 hours of Urdu training data), in a complete studio. It writes Urdu in normal Urdu script, checks its own output for accuracy, and runs free on Google Colab.

---

## ✨ Features

| | |
|---|---|
| 🎬 **Studio** | Script → mastered audio. Each part is rendered several times in one GPU batch, and the best take is picked by **Whisper** (were the words right?) and **WavLM speaker verification** (does it sound like the voice?). Every part gets a 🟢🟡🔴 score, and you can edit and retake a single part. Draft / Final / Max quality, style presets, −14 LUFS mastering, **WAV + MP3 + synced SRT subtitles**. |
| 📦 **Batch** | Paste many scripts separated by `---` (or upload a `.txt`) and get one zip with audio and subtitles for each. Built for a week of reels at once. |
| ⚡ **Live** | Streaming: audio starts after the first sentence. **Speak into the mic → hear it in the cloned voice**, optionally translated to English. Spoken Urdu stays in Urdu script (Whisper's habit of writing Urdu as Hindi/Devanagari is corrected). |
| 🧠 **Train** | Fine-tune on **15–60 min** of one speaker: auto-cut and auto-transcribe, **fix transcripts in a table**, LoRA training with a live loss chart, presets plus advanced controls (rank, alpha, dropout, target layers, LR, batch, warm-up, resume). The **best checkpoint is picked automatically** by scoring every checkpoint against the base model. |
| 🎯 **Accuracy helpers** | Reference-clip quality meter (length, noise, clipping), automatic pick of the clearest 5–10 s with its exact transcript, Urdu numbers → words (`2026` → دو ہزار چھبیس, `45%` → پینتالیس فیصد, lakh/crore), pronunciation dictionary, Arabic-keyboard letter normalisation. |
| 🧰 **Also** | Voice library, render history, base-vs-trained A/B compare, a one-minute **self-test**, command line, mobile-friendly UI with right-to-left Urdu text boxes. |

---

## 🚀 Quick start — Google Colab (free, nothing to install)

1. Click **[Open in Colab](https://colab.research.google.com/github/qasimrazamalik-coder/voice-cloner/blob/main/notebooks/Voice_Cloner_Colab.ipynb)**.
2. **Runtime → Change runtime type → T4 GPU** → Save.
3. Run the three cells. Allow Google Drive access so your voices are kept.
4. Open the **`https://….gradio.live`** link that appears. It also works on your phone.
5. In the app: **🩺 Diagnostics → Run self-test**, then upload a 5–10 s clip in **🎤 Voice** and go to **🎬 Studio**.

Everything you create is saved to `MyDrive/VoiceCloner/` (voices, adapters, datasets, history).

## 💻 Run on your own computer

Needs an NVIDIA GPU (about 8 GB VRAM or more recommended) and Python 3.10+.

```bash
git clone https://github.com/qasimrazamalik-coder/voice-cloner
cd voice-cloner
pip install -r requirements.txt        # install the PyTorch build for your CUDA first if needed
python -m voice_cloner app             # opens http://127.0.0.1:7860
python -m voice_cloner app --share     # + a public link
```

Command line, no browser:

```bash
python -m voice_cloner say --voice my_voice --file examples/reel_scripts_urdu.txt --out reel.wav
python -m voice_cloner say --ref clip.wav --text "Hello from my cloned voice." --lang en --quality Draft
python -m voice_cloner selftest
```

Storage defaults to `~/VoiceCloner`. Change it with `--home DIR` or `VOICE_CLONER_HOME`.

---

## 🎤 Getting the best clone

| Do | Avoid |
|---|---|
| 5–10 s of natural speech, one speaker | music, echo, other voices |
| quiet room, phone or mic 15–20 cm away | shouting or whispering |
| clip in the **same language** you'll generate (Urdu clip → Urdu output) | a reference in another language (the clone inherits its accent) |
| type the exact words of the clip | wrong transcripts |

The **Clip quality** meter under the upload box tells you right away if a clip is too short, noisy or distorted.

## 🧠 Training your own voice

1. **🧠 Train → ① Dataset:** upload 15–60 min of recordings (long files are fine) → **Prepare dataset**. Skim the transcript table, fix wrong words (leave a cell empty to drop that clip), then **Save transcript fixes**.
2. **② Train:** pick a preset (Quick / Standard / Thorough) → **Start training**. Watch the loss go down. On a free T4 expect roughly an hour or more; A100/L4 is much faster. Out of memory? Lower *Batch tokens* in Advanced.
3. **③ Pick best checkpoint:** each checkpoint speaks held-out sentences and is scored against the base model. The winner is saved as your adapter.
4. In the top **🎤 Voice** panel choose your voice **and** your trained adapter.

Training uses OmniVoice's official LoRA recipe (`omnivoice.cli.train`) with single-GPU settings and automatic fp16 on GPUs without bf16 (such as the T4).

---

## 🗂️ Project layout

```
voice_cloner/
  engine.py     models, text normalisation, audio, judging, streaming, training (no UI)
  app.py        Gradio interface — build_app() / launch()
  __main__.py   command line: app | say | selftest
notebooks/      Voice_Cloner_Colab.ipynb
tests/          CPU test suite (heavy ML libraries are stubbed)
examples/       sample Urdu + English reel scripts
```

## 🧪 Tests

```bash
pip install pytest numpy pandas
pytest -q
```

The suite runs the whole pipeline (studio, retake, batch, live streaming, voice library, history, dataset tools) on a CPU using stand-ins for the models. It also checks that every button in the app is wired to a handler that accepts its inputs. CI runs it on Python 3.10–3.12.

## ⚠️ Known limitations

- "Real-time" here means **sentence-by-sentence streaming**. The first audio arrives after about the first sentence, so this is not a zero-latency live voice changer.
- Urdu quality depends on OmniVoice's Urdu data. Rare words may need a pronunciation rule.
- Free Colab sessions time out. Your Drive files stay; just run the cells again.
- Full-model fine-tuning isn't exposed (LoRA up to rank 128 is).

## 🛡️ Responsible use

Clone **only your own voice or voices you have clear permission to use.** Don't use this to impersonate people, deceive, or bypass voice-based security. Many countries regulate synthetic media; label AI-generated audio where appropriate. The author is not responsible for misuse.

## 🙏 Credits

- [OmniVoice](https://github.com/k2-fsa/OmniVoice) by k2-fsa (Apache-2.0): the speech model and LoRA training recipe
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): transcription and accuracy checking
- [WavLM speaker verification](https://huggingface.co/microsoft/wavlm-base-plus-sv) by Microsoft: voice-similarity scoring
- [Gradio](https://gradio.app): the web interface

## 📄 License

[MIT](LICENSE) © 2026 Qasim Raza Malik. Model weights keep their own licenses (OmniVoice: Apache-2.0).

**Contact:** iamqasimrazamalik@gmail.com · Issues and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
