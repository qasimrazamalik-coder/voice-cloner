"""Voice Cloner engine — Urdu + English voice cloning, streaming and LoRA fine-tuning on OmniVoice.

Everything that touches models, audio, text and training lives here. The Gradio UI is in ``app.py``
and the command line in ``__main__.py``. Call :func:`configure` once to choose where voices,
adapters, datasets and history are stored (defaults to ``~/VoiceCloner`` or ``$VOICE_CLONER_HOME``).
"""
import re, gc, os, sys, json, time, glob, shutil, random, zipfile, subprocess, tempfile, unicodedata
import numpy as np, torch, librosa, soundfile as sf

__all__ = []  # filled at the bottom: every public name

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
SR = 24000
BASE_ONLY = "None (base model)"
UPLOAD = "— use the clip I upload —"
LANG = {"Urdu": "ur", "English": "en"}
MODEL_ID = os.environ.get("VOICE_CLONER_MODEL", "k2-fsa/OmniVoice")

# ═════════════ storage ═════════════
HOME = VOICES = ADAPTERS = DATASETS = HISTORY = RUNS = TOKS = PRON_FILE = None

def configure(home=None, work_dir=None):
    """Set storage folders. ``home`` keeps everything you want to keep (put it on Google Drive in Colab);
    ``work_dir`` holds large temporary training files (fast local disk is best)."""
    global HOME, VOICES, ADAPTERS, DATASETS, HISTORY, RUNS, TOKS, PRON_FILE
    HOME = os.path.abspath(os.path.expanduser(home or os.environ.get("VOICE_CLONER_HOME") or "~/VoiceCloner"))
    work = os.path.abspath(os.path.expanduser(work_dir or os.environ.get("VOICE_CLONER_WORK") or os.path.join(HOME, "_work")))
    VOICES, ADAPTERS, DATASETS, HISTORY = (os.path.join(HOME, d) for d in ("voices", "adapters", "datasets", "history"))
    RUNS, TOKS = os.path.join(work, "runs"), os.path.join(work, "tokens")
    PRON_FILE = os.path.join(HOME, "pronunciations.json")
    for d in (VOICES, ADAPTERS, DATASETS, HISTORY, RUNS, TOKS): os.makedirs(d, exist_ok=True)
    return HOME

configure()

# ═════════════ models: loaded lazily, freed before training ═════════════
M = {"tts": None, "adapter": BASE_ONLY, "asr": None, "asr_failed": False, "spk": None, "spk_fe": None, "spk_failed": False}

def free_models(keep_judges=False):
    M["tts"] = None; M["adapter"] = BASE_ONLY
    if not keep_judges: M["asr"] = None; M["spk"] = M["spk_fe"] = None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def get_tts(adapter=BASE_ONLY, adapter_path=None):
    # adapter = name of a saved adapter, or adapter_path = any LoRA checkpoint directory
    from omnivoice import OmniVoice
    key = adapter_path or adapter
    if M["tts"] is not None and M["adapter"] == key: return M["tts"]
    M["tts"] = None; gc.collect(); torch.cuda.empty_cache()
    m = OmniVoice.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=DTYPE)
    path = adapter_path or (f"{ADAPTERS}/{adapter}/final" if adapter != BASE_ONLY else None)
    if path:
        from omnivoice.utils.lora import load_lora_adapter
        m = load_lora_adapter(m, path, merge=True)
    M["tts"], M["adapter"] = m, key
    global SR
    SR = int(getattr(m, "sampling_rate", None) or getattr(m, "sample_rate", None) or 24000)
    return m

def get_asr():
    if M["asr"] is not None or M["asr_failed"]: return M["asr"]
    try:
        from faster_whisper import WhisperModel
        cuda = DEVICE.startswith("cuda")
        for name in ("large-v3-turbo", "large-v3", "medium"):
            try:
                M["asr"] = WhisperModel(name, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
                print("✅ Whisper:", name); break
            except Exception as e:
                print("  whisper", name, "unavailable:", str(e)[:80])
    except Exception as e:
        print("⚠️ Whisper off:", str(e)[:120])
    M["asr_failed"] = M["asr"] is None
    return M["asr"]

def get_spk():
    if M["spk"] is not None or M["spk_failed"]: return M["spk"]
    try:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        M["spk_fe"] = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        M["spk"] = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEVICE).eval()
        print("✅ Speaker judge: WavLM-SV")
    except Exception as e:
        print("⚠️ Speaker judge off:", str(e)[:120])
    M["spk_failed"] = M["spk"] is None
    return M["spk"]

def speaker_embedding(wav, sr):
    spk = get_spk()
    if spk is None: return None
    w16 = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000)
    inp = M["spk_fe"](w16, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        e = spk(**inp).embeddings
    return torch.nn.functional.normalize(e, dim=-1)[0]

def transcribe_segments(wav, sr, lang=None, task="transcribe", vad=True):
    asr = get_asr()
    if asr is None: return [], lang
    w16 = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000)
    segs, info = asr.transcribe(w16, language=lang, task=task, beam_size=5, vad_filter=vad,
                                condition_on_previous_text=False)
    out = [(s.start, s.end, s.text.strip(), s.avg_logprob, s.no_speech_prob) for s in segs if s.text.strip()]
    return out, getattr(info, "language", lang)

def transcribe(wav, sr, lang):
    segs, _ = transcribe_segments(wav, sr, lang, vad=False)
    return " ".join(s[2] for s in segs).strip()

def gpu_status():
    if not torch.cuda.is_available(): return "⚠️ **No GPU** — Runtime → Change runtime type → GPU"
    used = torch.cuda.memory_allocated() / 2**30; total = torch.cuda.get_device_properties(0).total_memory / 2**30
    loaded = [k for k in ("tts", "asr", "spk") if M[k] is not None]
    ad = M["adapter"] if M["tts"] is not None else "—"
    return (f"🟢 **{torch.cuda.get_device_name(0)}** · VRAM {used:.1f}/{total:.0f} GB · "
            f"models: {', '.join(loaded) or 'none yet (load on first use)'} · adapter: {os.path.basename(str(ad))}")

# ═════════════ text: numbers, pronunciation, scoring ═════════════
UR_0_99 = ("صفر ایک دو تین چار پانچ چھ سات آٹھ نو دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ انیس "
           "بیس اکیس بائیس تئیس چوبیس پچیس چھبیس ستائیس اٹھائیس انتیس تیس اکتیس بتیس تینتیس چونتیس "
           "پینتیس چھتیس سینتیس اڑتیس انتالیس چالیس اکتالیس بیالیس تینتالیس چوالیس پینتالیس چھیالیس "
           "سینتالیس اڑتالیس انچاس پچاس اکیاون باون ترپن چوون پچپن چھپن ستاون اٹھاون انسٹھ ساٹھ "
           "اکسٹھ باسٹھ تریسٹھ چونسٹھ پینسٹھ چھیاسٹھ سڑسٹھ اڑسٹھ انہتر ستر اکہتر بہتر تہتر چوہتر "
           "پچہتر چھہتر ستتر اٹھہتر اناسی اسی اکاسی بیاسی تراسی چوراسی پچاسی چھیاسی ستاسی اٹھاسی "
           "نواسی نوے اکانوے بانوے ترانوے چورانوے پچانوے چھیانوے ستانوے اٹھانوے ننانوے").split()
assert len(UR_0_99) == 100

def ur_int(n):
    if n < 100: return UR_0_99[n]
    for val, word in ((10**9, "ارب"), (10**7, "کروڑ"), (10**5, "لاکھ"), (1000, "ہزار"), (100, "سو")):
        if n >= val:
            q, r = divmod(n, val)
            return f"{ur_int(q)} {word}" + (f" {ur_int(r)}" if r else "")

def en_number(num_str, is_int):
    try:
        from num2words import num2words
        if is_int:
            n = int(num_str)
            return num2words(n, to="year") if 1100 <= n <= 2099 and len(num_str) == 4 else num2words(n)
        return num2words(float(num_str))
    except Exception:
        return num_str

def numbers_to_words(text, lang):
    text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    def repl(m):
        s, pct = m.group(1), m.group(2)
        if "." in s:
            a, b = s.split(".", 1)
            words = (ur_int(int(a)) + " اعشاریہ " + " ".join(UR_0_99[int(d)] for d in b)) if lang == "ur" else en_number(s, False)
        else:
            words = ur_int(int(s)) if lang == "ur" else en_number(s, True)
        if pct: words += " فیصد" if lang == "ur" else " percent"
        return f" {words} "
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(%)?", repl, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\s+([,.!?;:،۔؟])", r"\1", text).strip()

def clean_script_chars(text):
    # Arabic-keyboard variants → standard Urdu letters; strip invisible marks that confuse tokenizers
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ە", "ہ")
    return re.sub(r"[​-‏‪-‮﻿]", "", text)

def load_pron():
    try: return json.load(open(PRON_FILE, encoding="utf-8"))
    except Exception: return {}

def pron_to_text(d): return "\n".join(f"{k} = {v}" for k, v in d.items())

def save_pron(text):
    d = {}
    for line in (text or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip() and v.strip(): d[k.strip()] = v.strip()
    json.dump(d, open(PRON_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return f"✅ Saved {len(d)} rule(s)"

def apply_pron(text, d):
    for k in sorted(d, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", d[k], text, flags=re.IGNORECASE)
    return text

def prepare_text(text, lang, pron=None):
    return numbers_to_words(apply_pron(clean_script_chars(text), pron if pron is not None else load_pron()), lang)

def norm_text(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[ً-ٰٟ]", "", s)
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ه", "ہ")
    return re.sub(r"[^\w]", "", s)

def cer(ref, hyp):
    r, h = norm_text(ref), norm_text(hyp)
    if not r: return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (rc != hc))
        prev = cur
    return min(prev[-1] / len(r), 1.0)

PAUSE = {"،": 0.20, ",": 0.20, ";": 0.28, ":": 0.28, "۔": 0.45, ".": 0.45, "!": 0.48, "?": 0.48, "؟": 0.48}
PARA_PAUSE = 0.80

def split_script(text, max_chars=200, pack=True):
    # pack=True joins short sentences (smoother prosody); pack=False = one sentence per part (fastest streaming)
    out = []
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", (text or "").strip()) if p.strip()]
    for pi, para in enumerate(paras):
        sents = [s.strip() for s in re.split(r'(?<=[.!?۔؟])\s+', para) if s.strip()]
        chunks, cur = [], ""
        for s in sents:
            while len(s) > max_chars:
                cut = max(s.rfind("،", 0, max_chars), s.rfind(",", 0, max_chars), s.rfind(" ", 0, max_chars))
                cut = cut if cut > 40 else max_chars
                if cur: chunks.append(cur); cur = ""
                chunks.append(s[:cut + 1].strip()); s = s[cut + 1:].strip()
            if not s: continue
            if pack and len(cur) + len(s) + 1 <= max_chars: cur = (cur + " " + s).strip()
            else:
                if cur: chunks.append(cur)
                cur = s
        if cur: chunks.append(cur)
        for ci, ch in enumerate(chunks):
            pause = PAUSE.get(ch[-1], 0.22)
            if ci == len(chunks) - 1 and pi < len(paras) - 1: pause = PARA_PAUSE
            out.append((ch, pause))
    return out

def merge_segments(segs, min_len=3.0, max_len=15.0, max_gap=0.6):
    # segs: (start, end, text, avg_logprob, ...) → windows on natural pauses, min..max seconds
    out, cur = [], None
    for seg in segs:
        s, e, t, lp = seg[:4]
        if cur and (e - cur[0] <= max_len) and (s - cur[1] <= max_gap):
            cur = [cur[0], e, cur[2] + " " + t, min(cur[3], lp)]
        else:
            if cur and cur[1] - cur[0] >= min_len: out.append(cur)
            cur = [s, e, t, lp]
    if cur and cur[1] - cur[0] >= min_len: out.append(cur)
    return [c for c in out if c[1] - c[0] <= max_len]

# ═════════════ audio ═════════════
def denoise_wav(y, sr, amount=0.8):
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=amount).astype(np.float32)
    except Exception as e:
        print("denoise skipped:", e); return y

def peak_norm(y, peak=0.95): return (peak * y / (np.max(np.abs(y)) or 1.0)).astype(np.float32)

def pick_reference(path, lang, denoise=True):
    """Clean the clip; if it is long, pick the clearest 5–10 s stretch that starts/ends on pauses
    and return its exact transcript (a matching transcript improves cloning accuracy)."""
    y, sr = librosa.load(path, sr=24000, mono=True)
    if len(y) < sr * 1.5: raise ValueError("Reference clip is too short — use at least 3 seconds (5–10 s is best).")
    if denoise: y = denoise_wav(y, sr)
    y, _ = librosa.effects.trim(y, top_db=30)
    text = None
    if len(y) > sr * 12 and get_asr() is not None:
        segs, _ = transcribe_segments(y, sr, lang)
        wins = merge_segments([s for s in segs if s[4] < 0.5], min_len=5.0, max_len=10.0, max_gap=0.8)
        if wins:
            s, e, text, _ = max(wins, key=lambda w: w[3])
            y = y[max(0, int((s - 0.1) * sr)): int((e + 0.15) * sr)]
    if text is None: y = y[: sr * 10]
    y = peak_norm(y)
    out = os.path.join(tempfile.gettempdir(), f"ref_{int(time.time() * 1000)}.wav")
    sf.write(out, y, sr)
    return out, y, sr, text

def tidy_chunk(w, sr, fade_ms=12):
    w, _ = librosa.effects.trim(np.asarray(w, dtype=np.float32), top_db=40)
    if len(w) < int(sr * 0.05): w = np.zeros(int(sr * 0.05), dtype=np.float32)
    n = min(int(sr * fade_ms / 1000), len(w) // 2)
    if n > 0:
        w = w.copy(); ramp = np.linspace(0, 1, n, dtype=np.float32)
        w[:n] *= ramp; w[-n:] *= ramp[::-1]
    return w

def assemble(wavs, pauses, sr, pause_scale=1.0):
    rms = [max(float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0, 1e-6) for w in wavs]
    voiced = [r for r in rms if r > 1e-3]
    target = float(np.median(voiced)) if voiced else 1.0
    parts, spans, t = [], [], 0
    for i, (w, p) in enumerate(zip(wavs, pauses)):
        w = w * float(np.clip(target / rms[i], 0.6, 1.7))
        spans.append((t / sr, (t + len(w)) / sr)); parts.append(w); t += len(w)
        if i < len(wavs) - 1:
            gap = np.zeros(int(sr * p * pause_scale), dtype=np.float32); parts.append(gap); t += len(gap)
    return np.concatenate(parts).astype(np.float32), spans

def master(audio, sr, target_lufs=-14.0):
    try:
        import pyloudnorm as pyln
        loud = pyln.Meter(sr).integrated_loudness(audio)
        if np.isfinite(loud): audio = pyln.normalize.loudness(audio, loud, target_lufs)
    except Exception as e:
        print("loudness step skipped:", e)
    peak = np.max(np.abs(audio)) or 1.0
    if peak > 0.98: audio = audio * (0.98 / peak)
    return audio.astype(np.float32)

def new_stamp():
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"

def write_mp3(audio, sr, path):
    """MP3 copy for phones / social uploads; returns None if this soundfile build lacks MP3 support."""
    try:
        sf.write(path, audio, sr, format="MP3")
        return path
    except Exception as e:
        print("mp3 export skipped:", str(e)[:80])
        return None

def to_int16(w): return (np.clip(w, -1, 1) * 32767).astype(np.int16)

def srt_time(t):
    h, rem = divmod(int(round(t * 1000)), 3600000); m, rem = divmod(rem, 60000); s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def make_srt(texts, spans, max_words=6, max_chars=38):
    cues = []
    for text, (a, b) in zip(texts, spans):
        groups, cur = [], []
        for w in text.split():
            if cur and (len(cur) >= max_words or len(" ".join(cur + [w])) > max_chars): groups.append(cur); cur = []
            cur.append(w)
        if cur: groups.append(cur)
        total = sum(len(" ".join(g)) for g in groups) or 1; t = a
        for g in groups:
            d = (b - a) * len(" ".join(g)) / total; cues.append((t, t + d, " ".join(g))); t += d
    return "\n".join(f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{x}\n" for i, (s, e, x) in enumerate(cues, 1))

# ═════════════ voices ═════════════
def list_voices(): return sorted(d for d in os.listdir(VOICES) if os.path.exists(f"{VOICES}/{d}/ref.wav"))
def list_adapters():
    return [BASE_ONLY] + sorted(d for d in os.listdir(ADAPTERS) if os.path.exists(f"{ADAPTERS}/{d}/final/adapter_config.json"))
def list_history(): return sorted((f[:-5] for f in os.listdir(HISTORY) if f.endswith(".json")), reverse=True)
def safe_name(s): return re.sub(r"[^\w\-]+", "_", (s or "").strip()).strip("_")

def _voice_prompt_cls():
    try:
        from omnivoice import VoiceClonePrompt
    except ImportError:
        from omnivoice.models.omnivoice import VoiceClonePrompt
    return VoiceClonePrompt

def save_voice(name, ref_audio, ref_text, language, denoise):
    name = safe_name(name)
    if not name: raise ValueError("Give the voice a name.")
    if not ref_audio: raise ValueError("Upload or record a reference clip first.")
    lang = LANG[language]
    path, y, sr, auto_text = pick_reference(ref_audio, lang, denoise)
    text = (ref_text or "").strip() or auto_text or transcribe(y, sr, lang)
    d = f"{VOICES}/{name}"; os.makedirs(d, exist_ok=True)
    shutil.copy(path, f"{d}/ref.wav")
    open(f"{d}/ref.txt", "w", encoding="utf-8").write(text)
    try: (M["tts"] or get_tts(BASE_ONLY)).create_voice_clone_prompt(ref_audio=f"{d}/ref.wav", ref_text=text or None).save(f"{d}/prompt.pt")
    except Exception as e: print("prompt cache skipped:", e)
    return f"✅ Saved **{name}** · {len(y)/sr:.1f}s · “{text[:70]}”"

def resolve_voice(voice_choice, ref_audio, ref_text, lang, denoise):
    tts = M["tts"]
    if voice_choice and voice_choice != UPLOAD:
        d = f"{VOICES}/{voice_choice}"
        y, sr = librosa.load(f"{d}/ref.wav", sr=24000, mono=True)
        txt = open(f"{d}/ref.txt", encoding="utf-8").read().strip() if os.path.exists(f"{d}/ref.txt") else None
        if os.path.exists(f"{d}/prompt.pt"):
            try: return {"voice_clone_prompt": _voice_prompt_cls().load(f"{d}/prompt.pt")}, y, sr
            except Exception as e: print("saved prompt unreadable, rebuilding:", e)
        return {"voice_clone_prompt": tts.create_voice_clone_prompt(ref_audio=f"{d}/ref.wav", ref_text=txt or None)}, y, sr
    if not ref_audio: raise ValueError("Pick a saved voice or upload a reference clip.")
    path, y, sr, auto_text = pick_reference(ref_audio, lang, denoise)
    txt = (ref_text or "").strip() or auto_text or transcribe(y, sr, lang) or None
    try: return {"voice_clone_prompt": tts.create_voice_clone_prompt(ref_audio=path, ref_text=txt)}, y, sr
    except Exception as e:
        print("prompt build fell back to direct reference:", e)
        return {"ref_audio": path, "ref_text": txt}, y, sr

# ═════════════ generation ═════════════
QUALITY = {"Draft — fast preview": (1, 16), "Final — best quality": (3, 32), "Max — slowest, cleanest": (5, 48)}
STYLE = {"Natural": (1.00, 1.00), "Reel narration": (1.05, 0.80), "Calm explainer": (0.95, 1.20), "Energetic": (1.10, 0.70)}
TAKE_SPEEDS = (1.00, 0.97, 1.03, 0.95, 1.05)

def _to_np(w):
    if torch.is_tensor(w): w = w.detach().float().cpu().numpy()
    return np.asarray(w, dtype=np.float32).squeeze()

def gen_takes(spoken, voice, lang, steps, speed, n):
    tts = M["tts"]
    speeds = [round(speed * f, 3) for f in TAKE_SPEEDS[:n]]
    kw = {"num_step": int(steps)}
    try:
        outs = tts.generate(text=[spoken] * n, language=lang, speed=speeds, **voice, **kw)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        outs = [tts.generate(text=spoken, language=lang, speed=s, **voice, **kw)[0] for s in speeds]
    return [_to_np(w) for w in outs]

def judge(wav, spoken, lang, ref_emb):
    emb = speaker_embedding(wav, SR) if ref_emb is not None else None
    sim = float(torch.dot(emb, ref_emb)) if emb is not None else None
    err = cer(spoken, transcribe(wav, SR, lang)) if get_asr() is not None else None
    return (sim or 0.0) - 1.5 * (err or 0.0), sim, err

def best_take(spoken, voice, lang, steps, speed, n, ref_emb, judged=True):
    takes = gen_takes(spoken, voice, lang, steps, speed, n)
    if not judged: return takes[0], None, None
    scored = [(judge(w, spoken, lang, ref_emb), w) for w in takes]
    (_, sim, err), w = max(scored, key=lambda x: x[0][0])
    return w, sim, err

def badge(sim, err):
    if sim is None and err is None: return "⚪"
    if (err is not None and err > 0.15) or (sim is not None and sim < 0.70): return "🔴"
    if (err is None or err <= 0.05) and (sim is None or sim >= 0.80): return "🟢"
    return "🟡"

# ═════════════ studio session ═════════════
S = {}

def report_md():
    rows = []
    for i, c in enumerate(S["chunks"]):
        s = f"{c['sim']:.2f}" if c["sim"] is not None else "–"
        e = f"{c['err']*100:.0f}%" if c["err"] is not None else "–"
        rows.append(f"| {badge(c['sim'], c['err'])} {i+1} | {s} | {e} | {c['text'][:44]}{'…' if len(c['text']) > 44 else ''} |")
    return ("| part | voice | word err | text |\n|---|---|---|---|\n" + "\n".join(rows) +
            f"\n\n{len(S['chunks'])} part(s) · {S['takes']} take(s) each · 🟢 great · 🟡 ok · 🔴 retake it")

def render():
    audio, spans = assemble([c["wav"] for c in S["chunks"]], [c["pause"] for c in S["chunks"]], SR, S["pause_scale"])
    audio = master(audio, SR)
    stamp = S["stamp"]
    wav_path, srt_path = f"{HISTORY}/{stamp}.wav", f"{HISTORY}/{stamp}.srt"
    sf.write(wav_path, audio, SR, subtype="PCM_24")
    open(srt_path, "w", encoding="utf-8").write(make_srt([c["text"] for c in S["chunks"]], spans))
    mp3_path = write_mp3(audio, SR, f"{HISTORY}/{stamp}.mp3")
    json.dump({"title": S["title"], "language": S["lang"], "adapter": S["adapter"], "voice": S["voice_name"],
               "duration": round(len(audio) / SR, 1), "sentences": [c["text"] for c in S["chunks"]]},
              open(f"{HISTORY}/{stamp}.json", "w", encoding="utf-8"), ensure_ascii=False)
    return wav_path, [p for p in (mp3_path, srt_path) if p]

def studio_generate(voice_choice, ref_audio, ref_text, text, language, adapter, quality, denoise, speed, pause_scale, progress=None):
    if not text or not text.strip(): raise ValueError("Type a script first.")
    parts = split_script(text)
    if not parts: raise ValueError("The script has no speakable text.")
    lang = LANG[language]; takes, steps = QUALITY[quality]
    if progress: progress(0.02, desc="Loading models…")
    get_tts(adapter)
    voice, ref_wav, ref_sr = resolve_voice(voice_choice, ref_audio, ref_text, lang, denoise)
    ref_emb = speaker_embedding(ref_wav, ref_sr)
    pron = load_pron()
    S.clear(); S.update(voice=voice, ref_emb=ref_emb, lang=lang, steps=steps, speed=speed, takes=takes, adapter=adapter,
                        pause_scale=pause_scale, pron=pron, chunks=[], stamp=new_stamp(),
                        title=re.sub(r"\s+", " ", text.strip())[:40], voice_name=voice_choice if voice_choice != UPLOAD else "uploaded clip")
    for i, (chunk, pause) in enumerate(parts):
        if progress: progress((i + 0.5) / len(parts), desc=f"Sentence {i+1}/{len(parts)} · {takes} take(s)")
        spoken = prepare_text(chunk, lang, pron)
        wav, sim, err = best_take(spoken, voice, lang, steps, speed, takes, ref_emb)
        S["chunks"].append({"text": chunk, "spoken": spoken, "pause": pause, "wav": tidy_chunk(wav, SR), "sim": sim, "err": err})
    wav_path, downloads = render()
    return wav_path, downloads, report_md(), [str(i + 1) for i in range(len(S["chunks"]))]

def studio_retake(idx, new_text, progress=None):
    if not S.get("chunks"): raise ValueError("Generate something first.")
    if not idx: raise ValueError("Choose a sentence number.")
    c = S["chunks"][int(idx) - 1]
    if new_text and new_text.strip(): c["text"] = new_text.strip()
    get_tts(S["adapter"])
    c["spoken"] = prepare_text(c["text"], S["lang"], S["pron"])
    if progress: progress(0.3, desc=f"Retaking sentence {idx}…")
    wav, sim, err = best_take(c["spoken"], S["voice"], S["lang"], S["steps"], S["speed"], max(S["takes"], 3), S["ref_emb"])
    c.update(wav=tidy_chunk(wav, SR), sim=sim, err=err)
    wav_path, downloads = render()
    return wav_path, downloads, report_md(), (SR, to_int16(c["wav"]))

def sentence_preview(idx):
    if not S.get("chunks") or not idx: return None, ""
    c = S["chunks"][int(idx) - 1]
    return (SR, to_int16(c["wav"])), c["text"]

# ═════════════ real-time: streaming + speak-to-clone ═════════════
def _stream_text(text, lang, voice, fast, speed=1.0):
    pron = load_pron(); target = 0.08
    for chunk, pause in split_script(text, max_chars=160, pack=False):
        wav = gen_takes(prepare_text(chunk, lang, pron), voice, lang, 16 if fast else 32, speed, 1)[0]
        wav = tidy_chunk(wav, SR)
        rms = float(np.sqrt(np.mean(wav ** 2))) or 1e-6
        wav = np.clip(wav * min(target / rms, 3.0), -0.98, 0.98)
        yield (SR, to_int16(np.concatenate([wav, np.zeros(int(SR * pause), np.float32)])))

def live_tts(voice_choice, ref_audio, ref_text, text, language, adapter, denoise, fast):
    if not text or not text.strip(): raise ValueError("Type something to speak.")
    get_tts(adapter); lang = LANG[language]
    voice, _, _ = resolve_voice(voice_choice, ref_audio, ref_text, lang, denoise)
    yield from _stream_text(text, lang, voice, fast)

def speak_to_clone(mic, voice_choice, ref_audio, ref_text, in_lang, out_mode, adapter, denoise, fast):
    if not mic: raise ValueError("Record yourself first (press the mic, speak, stop).")
    if get_asr() is None: raise ValueError("Speech recognition is unavailable in this session.")
    y, sr = librosa.load(mic, sr=16000, mono=True)
    translate = out_mode.startswith("English")
    if translate:
        segs, _ = transcribe_segments(y, sr, None if in_lang == "Auto" else LANG[in_lang], task="translate")
        out_lang = "en"
    else:
        src = None if in_lang == "Auto" else LANG[in_lang]
        segs, detected = transcribe_segments(y, sr, src)
        # Whisper often labels spoken Urdu as Hindi and writes Devanagari — anything that is not English
        # is re-transcribed as Urdu so the text (and the cloned voice) stays in Urdu script.
        out_lang = "en" if (src or detected) == "en" else "ur"
        if src is None and out_lang == "ur" and detected != "ur":
            segs, _ = transcribe_segments(y, sr, "ur")
    said = " ".join(s[2] for s in segs).strip()
    if not said: raise ValueError("I couldn't hear any speech — try again a little closer to the mic.")
    yield f"🗣️ **Heard:** {said}", None
    get_tts(adapter)
    voice, _, _ = resolve_voice(voice_choice, ref_audio, ref_text, out_lang, denoise)
    for chunk in _stream_text(said, out_lang, voice, fast):
        yield f"🗣️ **Heard:** {said}", chunk

# ═════════════ reference quality meter ═════════════
def reference_stats(y, sr):
    """Duration, speech level, noise floor, SNR and clipping of a mono clip (pure numpy)."""
    y = np.asarray(y, dtype=np.float32)
    dur = len(y) / sr
    frame, hop = 2048, 512
    if len(y) < frame:
        return {"duration": dur, "speech_db": -120.0, "noise_db": -120.0, "snr_db": 0.0, "clip_pct": 0.0, "peak_db": -120.0}
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    speech, noise = float(np.percentile(db, 90)), float(np.percentile(db, 10))
    peak = float(np.max(np.abs(y))) + 1e-9
    return {"duration": dur, "speech_db": speech, "noise_db": noise, "snr_db": speech - noise,
            "clip_pct": float(np.mean(np.abs(y) >= 0.999) * 100), "peak_db": 20 * np.log10(peak)}

def analyze_reference(path):
    """Plain-language quality report for a reference clip, shown as soon as one is uploaded."""
    if not path: return ""
    y, sr = librosa.load(path, sr=24000, mono=True)
    st = reference_stats(y, sr)
    checks = []
    d = st["duration"]
    checks.append(("❌", f"Only {d:.1f}s — record at least 5 s") if d < 3 else
                  ("⚠️", f"{d:.1f}s — usable; 5–10 s clones better") if d < 5 else
                  ("✅", f"{d:.1f}s long") if d <= 15 else
                  ("✅", f"{d:.0f}s long — the clearest 5–10 s is picked automatically"))
    snr = st["snr_db"]
    checks.append(("✅", f"Clean background (≈{snr:.0f} dB SNR)") if snr >= 25 else
                  ("⚠️", f"Some background noise (≈{snr:.0f} dB) — keep “Clean background noise” on") if snr >= 15 else
                  ("❌", f"Noisy (≈{snr:.0f} dB) — re-record somewhere quieter for a closer clone"))
    if st["clip_pct"] > 0.1: checks.append(("⚠️", f"Distorted / clipped ({st['clip_pct']:.1f}% of samples) — record a little further from the mic"))
    if st["speech_db"] < -35: checks.append(("⚠️", "Very quiet — speak closer to the mic"))
    grade = "❌ Poor" if any(c[0] == "❌" for c in checks) else ("⚠️ OK" if any(c[0] == "⚠️" for c in checks) else "✅ Great")
    return f"**Clip quality: {grade}**  \n" + "  \n".join(f"{i} {t}" for i, t in checks)

def voice_details(name):
    if not name or name == UPLOAD: return None, ""
    d = f"{VOICES}/{name}"
    txt = open(f"{d}/ref.txt", encoding="utf-8").read().strip() if os.path.exists(f"{d}/ref.txt") else ""
    q = analyze_reference(f"{d}/ref.wav") if os.path.exists(f"{d}/ref.wav") else ""
    return f"{d}/ref.wav", (f"“{txt}”\n\n" if txt else "") + q

# ═════════════ batch rendering ═════════════
def split_batch(text):
    """Scripts separated by a line containing only --- (or ===)."""
    parts = re.split(r"^\s*(?:-{3,}|={3,})\s*$", (text or "").strip(), flags=re.M)
    return [p.strip() for p in parts if p.strip()]

def batch_render(scripts_text, scripts_file, voice_choice, ref_audio, ref_text, language, adapter, quality, denoise, speed, pause_scale, progress=None):
    text = scripts_text or ""
    if scripts_file:
        path = scripts_file if isinstance(scripts_file, str) else getattr(scripts_file, "name", "")
        text = (text + "\n---\n" if text.strip() else "") + open(path, encoding="utf-8").read()
    scripts = split_batch(text)
    if not scripts: raise ValueError("Add scripts — separate them with a line containing only ---")
    stamp = new_stamp(); zpath = f"{HISTORY}/batch-{stamp}.zip"; rows = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for i, script in enumerate(scripts, 1):
            if progress: progress((i - 1) / len(scripts), desc=f"Script {i}/{len(scripts)}")
            wav, downloads, _, _ = studio_generate(voice_choice, ref_audio, ref_text, script, language, adapter,
                                                   quality, denoise, speed, pause_scale, progress=None)
            base = f"{i:02d}_" + (safe_name(script[:24]) or "script")
            for p in [wav] + downloads: z.write(p, base + os.path.splitext(p)[1])
            worst = max((("🟢", "🟡", "🔴", "⚪").index(badge(c["sim"], c["err"])) for c in S["chunks"]), default=3)
            one_line = re.sub(r"\s+", " ", script)[:50]
            rows.append(f"| {i} | {('🟢', '🟡', '🔴', '⚪')[worst]} | {len(S['chunks'])} | {one_line} |")
    return zpath, "| # | worst part | parts | script |\n|---|---|---|---|\n" + "\n".join(rows) + f"\n\n✅ {len(scripts)} script(s) → one zip (WAV + MP3 + SRT each)"

# ═════════════ compare / history / diagnostics ═════════════
def compare(voice_choice, ref_audio, ref_text, sentence, language, adapter, denoise, progress=None):
    if adapter == BASE_ONLY: raise ValueError("Pick a trained adapter to compare against the base model.")
    if not sentence or not sentence.strip(): raise ValueError("Type a test sentence.")
    lang = LANG[language]; spoken = prepare_text(sentence, lang); results = []
    for k, ad in enumerate((BASE_ONLY, adapter)):
        if progress: progress(0.1 + 0.45 * k, desc=f"Rendering with {ad}…")
        get_tts(ad)
        voice, ref_wav, ref_sr = resolve_voice(voice_choice, ref_audio, ref_text, lang, denoise)
        wav, sim, err = best_take(spoken, voice, lang, 32, 1.0, 3, speaker_embedding(ref_wav, ref_sr))
        results.append(((SR, to_int16(master(tidy_chunk(wav, SR), SR))), sim, err))
    fmt = lambda s, e: (f"{badge(s, e)} voice {s:.2f} · word err {e*100:.0f}%" if s is not None and e is not None else "scores unavailable")
    return results[0][0], fmt(*results[0][1:]), results[1][0], fmt(*results[1][1:])

def open_history(stamp):
    if not stamp: return None, None, ""
    meta = json.load(open(f"{HISTORY}/{stamp}.json", encoding="utf-8"))
    files = [f"{HISTORY}/{stamp}.{ext}" for ext in ("mp3", "srt") if os.path.exists(f"{HISTORY}/{stamp}.{ext}")]
    return (f"{HISTORY}/{stamp}.wav", files,
            f"**{meta['title']}** · {meta['language']} · {meta.get('duration', '?')}s · voice: {meta['voice']} · adapter: {meta['adapter']}")

def self_test():
    lines = []
    def check(name, fn):
        t = time.time()
        try:
            r = fn()
            lines.append(f"✅ {name} ({time.time()-t:.1f}s)" + (f" — {r}" if r else ""))
            return True
        except Exception as e:
            lines.append(f"❌ {name}: {type(e).__name__}: {str(e)[:160]}")
            return False
    def need(cond, msg):
        if not cond: raise RuntimeError(msg)
    check("GPU", lambda: (need(torch.cuda.is_available(), "no GPU — switch the runtime to a GPU"), torch.cuda.get_device_name(0))[1])
    if check("Load OmniVoice", lambda: (get_tts(BASE_ONLY), "loaded")[1]):
        judges_ok = check("Whisper judge", lambda: (need(get_asr() is not None, "could not load"), "loaded")[1])
        spk_ok = check("Speaker judge", lambda: (need(get_spk() is not None, "could not load"), "loaded")[1])
        for lang, txt in (("ur", "یہ ایک آزمائشی جملہ ہے۔"), ("en", "This is a quick test sentence.")):
            box = {}
            def synth():
                box["w"] = _to_np(M["tts"].generate(text=prepare_text(txt, lang), language=lang, num_step=16)[0])
                need(len(box["w"]) > SR * 0.3, "audio came out empty")
                return f"{len(box['w'])/SR:.1f}s audio"
            if check(f"Synthesize ({lang})", synth) and judges_ok:
                def acc():
                    heard = transcribe(box["w"], SR, lang)
                    return f"word error {cer(txt, heard)*100:.0f}% · heard: {heard[:50]}"
                check(f"Round-trip accuracy ({lang})", acc)
            if "w" in box and spk_ok:
                check(f"Speaker embedding ({lang})", lambda: f"dim {speaker_embedding(box['w'], SR).shape[0]}")
    lines.append(gpu_status())
    return "\n\n".join(lines)

# ═════════════ training: dataset → tokens → LoRA → pick best checkpoint ═════════════
TRAIN = {"proc": None}

def ds_dir(name): return f"{DATASETS}/{name}"

def write_splits(name, entries):
    rnd = random.Random(0); items = entries[:]; rnd.shuffle(items)
    n_dev = max(2, len(items) // 20) if len(items) >= 10 else 1
    dev, train = items[:n_dev], items[n_dev:]
    for split, rows in (("train", train), ("dev", dev)):
        with open(f"{ds_dir(name)}/{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({k: r[k] for k in ("id", "audio_path", "text", "language_id")}, ensure_ascii=False) + "\n")
    open(f"{ds_dir(name)}/.changed", "w").write(str(time.time()))
    return len(train), len(dev)

def load_entries(name):
    p = f"{ds_dir(name)}/all.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []

def save_entries(name, entries):
    with open(f"{ds_dir(name)}/all.jsonl", "w", encoding="utf-8") as fh:
        for e in entries: fh.write(json.dumps(e, ensure_ascii=False) + "\n")

def list_datasets(): return sorted(d for d in os.listdir(DATASETS) if os.path.exists(f"{DATASETS}/{d}/all.jsonl"))

def prepare_dataset(name, language, files, denoise, min_len, max_len):
    name = safe_name(name); lang = LANG[language]
    if not name: raise ValueError("Give the voice a name.")
    raw = f"{ds_dir(name)}/raw"; clips = f"{ds_dir(name)}/clips"
    os.makedirs(raw, exist_ok=True); os.makedirs(clips, exist_ok=True)
    for f in files or []:
        src = f if isinstance(f, str) else getattr(f, "name", None)
        if src: shutil.copy(src, f"{raw}/{os.path.basename(src)}")
    audio_files = sorted(f for f in glob.glob(f"{raw}/*") if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".webm")))
    if not audio_files: raise ValueError("No recordings found — upload audio files first.")
    log = [f"📁 {len(audio_files)} recording(s) · language {lang}"]; yield "\n".join(log), "", []
    free_models(keep_judges=True)
    if get_asr() is None: raise ValueError("Whisper could not load — needed to transcribe the recordings.")
    entries, n = [], 0
    for f in audio_files:
        y, sr = librosa.load(f, sr=24000, mono=True)
        if denoise: y = denoise_wav(y, sr, 0.75)
        segs, _ = transcribe_segments(y, sr, lang)
        segs = [s for s in segs if s[4] < 0.5 and s[3] > -1.0]
        for s, e, t, lp in merge_segments(segs, float(min_len), float(max_len)):
            a, b = max(0, int((s - 0.1) * sr)), min(len(y), int((e + 0.15) * sr))
            cid = f"{name}_{n:05d}"; n += 1; p = f"{clips}/{cid}.wav"
            sf.write(p, peak_norm(y[a:b]), sr)
            entries.append({"id": cid, "audio_path": p, "text": t, "language_id": lang, "dur": round((b - a) / sr, 2), "lp": round(lp, 3)})
        log.append(f"  ✓ {os.path.basename(f)} → {n} clips so far"); yield "\n".join(log), "", []
    if len(entries) < 5: raise ValueError(f"Only {len(entries)} usable clips — need more clean speech (aim for 15+ minutes).")
    save_entries(name, entries); ntr, ndv = write_splits(name, entries)
    best = max([x for x in entries if 5 <= x["dur"] <= 10] or entries, key=lambda x: x["lp"])
    vd = f"{VOICES}/{name}"; os.makedirs(vd, exist_ok=True)
    shutil.copy(best["audio_path"], f"{vd}/ref.wav"); open(f"{vd}/ref.txt", "w", encoding="utf-8").write(best["text"])
    mins = sum(x["dur"] for x in entries) / 60
    tip = "⚠️ under 10 min — add more recordings for a real improvement" if mins < 10 else ("👍 good amount" if mins < 30 else "💪 great amount")
    summary = f"### ✅ {len(entries)} clips · {mins:.1f} min of speech · {ntr} train / {ndv} test\n{tip}. Fix any wrong transcripts in the table, then **Save fixes**."
    log.append("Done. Voice reference saved to your Voice Library as " + name)
    yield "\n".join(log), summary, [[e["id"], e["dur"], e["text"]] for e in entries]

def save_transcript_fixes(name, table):
    name = safe_name(name); entries = load_entries(name)
    if not entries: raise ValueError("Prepare the dataset first.")
    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    fixes = {str(r[0]): str(r[2]).strip() for r in rows if len(r) >= 3}
    changed, keep = 0, []
    for e in entries:
        t = fixes.get(e["id"], e["text"])
        if t != e["text"]: changed += 1
        if t: e["text"] = t; keep.append(e)          # empty text = drop that clip
    save_entries(name, keep); write_splits(name, keep)
    return f"✅ Saved — {changed} transcript(s) fixed, {len(entries) - len(keep)} clip(s) removed, {len(keep)} kept."

def load_dataset_table(name):
    entries = load_entries(safe_name(name))
    mins = sum(e.get("dur", 0) for e in entries) / 60
    return ([[e["id"], e.get("dur", 0), e["text"]] for e in entries],
            f"### {len(entries)} clips · {mins:.1f} min" if entries else "No dataset with that name yet.")

def _run_stream(cmd, env=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    TRAIN["proc"] = p
    for line in p.stdout: yield line.rstrip("\n")
    p.wait(); TRAIN["proc"] = None
    if p.returncode not in (0, None) and p.returncode != -15:
        raise RuntimeError(f"command failed (exit {p.returncode}) — see the log above")

def stop_training():
    p = TRAIN.get("proc")
    if p and p.poll() is None:
        p.terminate(); return "⏹️ Stopping… checkpoints saved so far are kept."
    return "Nothing is running."

def tokenize_if_needed(name):
    tok = f"{TOKS}/{name}"; stamp_src = f"{ds_dir(name)}/.changed"; stamp_tok = f"{tok}/.from"
    src = open(stamp_src).read() if os.path.exists(stamp_src) else "x"
    if os.path.exists(f"{tok}/train/data.lst") and os.path.exists(stamp_tok) and open(stamp_tok).read() == src:
        yield "✓ audio already tokenized"; return
    shutil.rmtree(tok, ignore_errors=True)
    for split in ("train", "dev"):
        yield f"▶ tokenizing {split}…"
        cmd = [sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
               "--input_jsonl", f"{ds_dir(name)}/{split}.jsonl",
               "--tar_output_pattern", f"{tok}/{split}/audios/shard-%06d.tar",
               "--jsonl_output_pattern", f"{tok}/{split}/txts/shard-%06d.jsonl",
               "--nj_per_gpu", "1", "--loader_workers", "2", "--min_num_shards", "1",
               "--min_length", "1.0", "--max_length", "20.0", "--skip_errors"]
        for line in _run_stream(cmd):
            if line.strip(): yield "   " + line[-160:]
        if not os.path.exists(f"{tok}/{split}/data.lst"): raise RuntimeError(f"tokenizing {split} produced no data.lst")
    os.makedirs(tok, exist_ok=True); open(stamp_tok, "w").write(src)
    yield "✓ tokenized"

TRAIN_PRESETS = {  # steps scale with data; LoRA rank/targets for capacity
    "Quick — ~15 min data": dict(steps=800, lora_r=16, lr=1e-4),
    "Standard — 20–40 min": dict(steps=1500, lora_r=32, lr=1e-4),
    "Thorough — 40+ min": dict(steps=3000, lora_r=64, lr=8e-5),
}

def train_voice(name, steps, lora_r, lora_alpha, lora_dropout, targets, lr, batch_tokens, grad_acc, warmup, save_every, resume):
    name = safe_name(name)
    if not load_entries(name): raise ValueError("Prepare a dataset with this name first (step 1).")
    free_models()
    log, pts = [], []
    def out(status):
        import pandas as pd
        df = pd.DataFrame(pts, columns=["step", "loss", "series"]) if pts else pd.DataFrame({"step": [], "loss": [], "series": []})
        return "\n".join(log[-40:]), df, status
    for line in tokenize_if_needed(name):
        log.append(line); yield out("Tokenizing audio…")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    mlp = ["gate_proj", "up_proj", "down_proj"]
    cfg = {
        "llm_name_or_path": "Qwen/Qwen3-0.6B", "audio_vocab_size": 1025, "audio_mask_id": 1024,
        "num_audio_codebook": 8, "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
        "drop_cond_ratio": 0.1, "prompt_ratio_range": [0.0, 0.3], "mask_ratio_range": [0.0, 1.0],
        "language_ratio": 0.8, "use_pinyin_ratio": 0.0, "instruct_ratio": 0.0, "only_instruct_ratio": 0.0,
        "resume_from_checkpoint": None, "init_from_checkpoint": MODEL_ID,
        "use_lora": True, "lora_r": int(lora_r), "lora_alpha": int(lora_alpha), "lora_dropout": float(lora_dropout), "lora_bias": "none",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"] + (mlp if targets.startswith("Attention + MLP") else []),
        "lora_modules_to_save": ["audio_embeddings", "audio_heads"],
        "learning_rate": float(lr), "weight_decay": 0.01, "max_grad_norm": 1.0,
        "steps": int(steps), "seed": 42, "warmup_type": "ratio", "warmup_ratio": float(warmup), "warmup_steps": 0,
        "attn_implementation": "sdpa", "max_sample_tokens": 2000, "min_sample_tokens": 50,
        "max_batch_size": 16, "batch_tokens": int(batch_tokens), "gradient_accumulation_steps": int(grad_acc), "num_workers": 2,
        "mixed_precision": "bf16" if bf16 else "fp16", "allow_tf32": bool(bf16),
        "logging_steps": 10, "eval_steps": int(save_every), "save_steps": int(save_every), "keep_last_n_checkpoints": 4,
    }
    run = f"{RUNS}/{name}"; exp = f"{run}/exp"
    if resume:
        ck = sorted(glob.glob(f"{exp}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[-1]))
        if ck: cfg["resume_from_checkpoint"] = ck[-1]; log.append(f"↻ resuming from {os.path.basename(ck[-1])}")
    else:
        shutil.rmtree(exp, ignore_errors=True)
    os.makedirs(run, exist_ok=True)
    json.dump(cfg, open(f"{run}/train_config.json", "w"), indent=2)
    json.dump({"train": [{"manifest_path": [f"{TOKS}/{name}/train/data.lst"]}],
               "dev": [{"manifest_path": [f"{TOKS}/{name}/dev/data.lst"]}]}, open(f"{run}/data_config.json", "w"), indent=2)
    log.append(f"▶ LoRA r={lora_r} α={lora_alpha} · lr {lr} · {steps} steps · {cfg['mixed_precision']}"); yield out("Starting training…")
    launcher = shutil.which("accelerate")
    head = [launcher, "launch"] if launcher else [sys.executable, "-m", "accelerate.commands.launch"]
    cmd = head + ["--num_processes", "1", "--mixed_precision", cfg["mixed_precision"],
                  "-m", "omnivoice.cli.train", "--train_config", f"{run}/train_config.json",
                  "--data_config", f"{run}/data_config.json", "--output_dir", exp]
    bar_re = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
    step_re, loss_re, eval_re = re.compile(r"[Ss]tep\D{0,3}(\d+)"), re.compile(r"train/loss:\s*([0-9.eE+-]+)"), re.compile(r"Eval Loss:\s*([0-9.eE+-]+)")
    last_step = 0; t0 = time.time()
    for line in _run_stream(cmd):
        m = step_re.search(line) or bar_re.search(line)
        if m: last_step = max(last_step, int(m.group(1)))
        m = loss_re.search(line)
        if m: pts.append([last_step, float(m.group(1)), "train"])
        m = eval_re.search(line)
        if m: pts.append([last_step, float(m.group(1)), "test"])
        if line.strip() and ("%|" not in line or len(log) == 0 or "%|" not in log[-1]): log.append(line[-200:])
        elif "%|" in line: log[-1] = line[-200:]
        el = time.time() - t0
        eta = f" · ~{(el / last_step) * (int(steps) - last_step) / 60:.0f} min left" if last_step > 5 else ""
        yield out(f"Training… step {last_step}/{steps}{eta}")
    log.append("✅ Training finished — now run **Pick best checkpoint**.")
    yield out("Done ✅")

def list_checkpoints(name):
    exp = f"{RUNS}/{safe_name(name)}/exp"
    ck = sorted(glob.glob(f"{exp}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[-1]))
    return [c for c in ck if os.path.exists(f"{c}/adapter_config.json")]

def export_adapter(name, ckpt):
    dst = f"{ADAPTERS}/{safe_name(name)}/final"
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    for f in os.listdir(ckpt):
        if f.startswith(("adapter_", "tokenizer", "special_tokens", "vocab", "merges", "added_tokens")):
            shutil.copy(f"{ckpt}/{f}", dst)
    json.dump({"from": ckpt, "exported": time.strftime("%Y-%m-%d %H:%M")}, open(f"{dst}/source.json", "w"))
    return dst

def evaluate_checkpoints(name, n_sent):
    """Score base model + every checkpoint on held-out test sentences (word error + voice match);
    export the best one. This picks the checkpoint that actually sounds best, not just the last one."""
    name = safe_name(name); cks = list_checkpoints(name)
    if not cks: raise ValueError("No checkpoints yet — train first.")
    dev = [json.loads(l) for l in open(f"{ds_dir(name)}/dev.jsonl", encoding="utf-8")][: int(n_sent)]
    if not dev: raise ValueError("No test sentences in the dataset.")
    lang = dev[0]["language_id"]; ref = f"{VOICES}/{name}/ref.wav"
    ref_txt = open(f"{VOICES}/{name}/ref.txt", encoding="utf-8").read().strip()
    rows = []; ref_emb = None; scores = {}
    def table(): return "| model | voice match | word error | score |\n|---|---|---|---|\n" + "\n".join(rows)
    yield table() + "\n\nLoading…"
    for label, path in [("base model", None)] + [(os.path.basename(c), c) for c in cks]:
        free_models(keep_judges=True)
        tts = get_tts(adapter_path=path) if path else get_tts(BASE_ONLY)
        if ref_emb is None:
            y, sr = librosa.load(ref, sr=24000, mono=True); ref_emb = speaker_embedding(y, sr)
        prompt = tts.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_txt or None)
        sims, errs = [], []
        for d in dev:
            w = gen_takes(prepare_text(d["text"], lang), {"voice_clone_prompt": prompt}, lang, 32, 1.0, 1)[0]
            _, s, e = judge(w, d["text"], lang, ref_emb)
            if s is not None: sims.append(s)
            if e is not None: errs.append(e)
        sim = float(np.mean(sims)) if sims else 0.0; err = float(np.mean(errs)) if errs else 0.0
        rows.append(f"| {label} | {sim:.3f} | {err*100:.1f}% | {sim - 1.5*err:.3f} |")
        scores[label] = (sim - 1.5 * err, path)
        yield table() + f"\n\nScored {label}…"
    base_score = scores.pop("base model", (None, None))[0]
    best_label, (best_score, best_path) = max(scores.items(), key=lambda kv: kv[1][0])
    export_adapter(name, best_path); free_models(keep_judges=True)
    gain = f" (base model scored {base_score:.3f})" if base_score is not None else ""
    yield table() + f"\n\n### 🏆 Best: **{best_label}** · score {best_score:.3f}{gain}\nExported to your Library → pick adapter **{name}** in the Studio."

__all__ = [n for n in list(globals()) if not n.startswith("__") and n not in ("re", "gc", "os", "sys", "json", "time", "glob",
           "shutil", "random", "zipfile", "subprocess", "tempfile", "unicodedata", "np", "torch", "librosa", "sf")]
