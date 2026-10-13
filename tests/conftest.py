"""Test setup: lightweight stand-ins for torch, librosa, soundfile, gradio and omnivoice so the
whole pipeline (text → takes → judging → assembly → files, plus every UI handler) runs on any CPU
in seconds. Real numpy and pandas are used."""
import os, sys, tempfile, types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["VOICE_CLONER_HOME"] = tempfile.mkdtemp(prefix="vc_home_")
os.environ["VOICE_CLONER_WORK"] = tempfile.mkdtemp(prefix="vc_work_")

# ---------- torch ----------
torch = types.ModuleType("torch")
class _Cuda:
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    @staticmethod
    def is_available(): return False
    @staticmethod
    def empty_cache(): pass
    @staticmethod
    def is_bf16_supported(): return False
class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): return False
torch.cuda = _Cuda()
torch.float16, torch.float32 = "float16", "float32"
torch.is_tensor = lambda x: False
torch.dot = lambda a, b: float(np.dot(a, b))
torch.no_grad = _NoGrad
torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(normalize=lambda e, dim=-1: e))
sys.modules["torch"] = torch

# ---------- librosa + soundfile (in-memory audio store) ----------
AUDIO = {}
def _tone(seconds, sr=24000, amp=0.3):
    return (np.sin(np.linspace(0, 2 * np.pi * 220 * seconds, int(sr * seconds))) * amp).astype(np.float32)
def _trim(y, top_db=40):
    if len(y) == 0: return y, (0, 0)
    thr = np.max(np.abs(y)) * 10 ** (-top_db / 20); idx = np.where(np.abs(y) > thr)[0]
    return (y[idx[0]: idx[-1] + 1], (idx[0], idx[-1])) if len(idx) else (y[:0], (0, 0))
def _load(path, sr=24000, mono=True):
    if path in AUDIO: return np.asarray(AUDIO[path][0], dtype=np.float32), sr
    return _tone(8, sr), sr
librosa = types.ModuleType("librosa")
librosa.effects = types.SimpleNamespace(trim=_trim)
librosa.load = _load
librosa.resample = lambda y, orig_sr, target_sr: np.asarray(y)[:: max(1, orig_sr // target_sr)]
sys.modules["librosa"] = librosa
soundfile = types.ModuleType("soundfile")
def _write(path, y, sr, subtype=None, format=None):
    AUDIO[path] = (np.asarray(y), sr)
    with open(path, "wb") as fh: fh.write(b"RIFF")
soundfile.write = _write
sys.modules["soundfile"] = soundfile

# ---------- omnivoice ----------
class FakePrompt:
    def __init__(self, ref_text=None): self.ref_text = ref_text
    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh: fh.write(self.ref_text or "")
    @classmethod
    def load(cls, path): return cls(open(path, encoding="utf-8").read())
class FakeOmniVoice:
    sampling_rate = 24000
    calls = []
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def create_voice_clone_prompt(self, ref_audio, ref_text=None, **k): return FakePrompt(ref_text)
    def generate(self, text, language=None, speed=None, voice_clone_prompt=None, ref_audio=None, ref_text=None, **kw):
        texts = text if isinstance(text, list) else [text]
        speeds = speed if isinstance(speed, list) else [speed] * len(texts)
        assert len(speeds) == len(texts), "per-take speed list must match the batch"
        FakeOmniVoice.calls.append({"n": len(texts), "steps": kw.get("num_step"), "language": language})
        return [_tone(0.05 * max(len(t), 6)) for t in texts]
omnivoice = types.ModuleType("omnivoice")
omnivoice.OmniVoice, omnivoice.VoiceClonePrompt = FakeOmniVoice, FakePrompt
sys.modules["omnivoice"] = omnivoice
sys.modules["omnivoice.utils"] = types.ModuleType("omnivoice.utils")
lora = types.ModuleType("omnivoice.utils.lora"); lora.load_lora_adapter = lambda m, p, merge=True: m
sys.modules["omnivoice.utils.lora"] = lora

# ---------- gradio (records components and event wiring) ----------
EVENTS = []
class _Comp:
    def __init__(self, *a, **k): self.args, self.kw = a, k
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def _bind(self, name):
        def bind(fn, inputs=None, outputs=None, **k):
            ins = [] if inputs is None else (inputs if isinstance(inputs, list) else [inputs])
            outs = [] if outputs is None else (outputs if isinstance(outputs, list) else [outputs])
            ev = types.SimpleNamespace(fn=fn, inputs=ins, outputs=outs, name=name)
            EVENTS.append(ev)
            ev.then = lambda f2, i2=None, o2=None, **kk: bind(f2, i2, o2)
            return ev
        return bind
    def __getattr__(self, n):
        if n in ("click", "change", "submit", "then", "upload"): return self._bind(n)
        raise AttributeError(n)
gradio = types.ModuleType("gradio")
for _name in ["Blocks", "Row", "Column", "Tabs", "Tab", "Accordion", "Group", "Markdown", "Button", "Dropdown", "Radio",
              "Audio", "Textbox", "Checkbox", "Slider", "File", "Dataframe", "Number", "LinePlot"]:
    setattr(gradio, _name, type(_name, (_Comp,), {}))
gradio.update = lambda **k: ("update", k)
gradio.Progress = lambda *a, **k: (lambda *a, **k: None)
gradio.WARNINGS = []
gradio.Warning = lambda msg: gradio.WARNINGS.append(msg)
gradio.themes = types.SimpleNamespace(Soft=lambda **k: None)
gradio.Blocks.queue = lambda self, **k: self
gradio.Blocks.launch = lambda self, **k: k
sys.modules["gradio"] = gradio


@pytest.fixture(scope="session")
def engine():
    from voice_cloner import engine as e
    e.M["asr_failed"] = True     # no Whisper / WavLM on CI: judging degrades gracefully
    e.M["spk_failed"] = True
    return e


@pytest.fixture(scope="session")
def ref_clip(engine):
    path = os.path.join(engine.HOME, "ref.wav")
    AUDIO[path] = (_tone(8), 24000)
    return path


@pytest.fixture
def audio_store(): return AUDIO


@pytest.fixture
def tone(): return _tone


@pytest.fixture(scope="session")
def built_app(engine):
    EVENTS.clear()
    from voice_cloner import app
    return app, app.build_app(), list(EVENTS)


@pytest.fixture
def fake_model(): return FakeOmniVoice
