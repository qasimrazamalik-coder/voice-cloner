"""End-to-end flows with a fake model: studio, retake, batch, live streaming, voices, history, datasets."""
import os, zipfile
import pandas as pd


def test_studio_generate_and_retake(engine, ref_clip, fake_model):
    wav, files, report, ids = engine.studio_generate(
        engine.UPLOAD, ref_clip, "میرا نام علی ہے۔", "پہلا جملہ 2026 میں۔ دوسرا، ٹھیک ہے؟\n\nنیا پیراگراف!",
        "Urdu", engine.BASE_ONLY, "Final — best quality", True, 1.0, 1.0)
    assert os.path.exists(wav) and ids == ["1", "2"]
    assert any(f.endswith(".srt") for f in files) and any(f.endswith(".mp3") for f in files)
    assert fake_model.calls[-1]["n"] == 3 and fake_model.calls[-1]["steps"] == 32   # 3 takes batched in one call
    assert fake_model.calls[-1]["language"] == "ur"
    assert "پہلا جملہ" in report
    assert "دو ہزار چھبیس" in engine.S["chunks"][0]["spoken"]
    _, _, _, part = engine.studio_retake("2", "دوسرا جملہ نیا۔")
    assert engine.S["chunks"][1]["text"] == "دوسرا جملہ نیا۔" and part[1].dtype.name == "int16"


def test_draft_quality_is_single_fast_take(engine, ref_clip, fake_model):
    engine.studio_generate(engine.UPLOAD, ref_clip, "", "Quick test.", "English", engine.BASE_ONLY,
                           "Draft — fast preview", False, 1.0, 1.0)
    assert fake_model.calls[-1] == {"n": 1, "steps": 16, "language": "en"}


def test_empty_script_is_a_clear_error(engine, ref_clip):
    import pytest
    with pytest.raises(ValueError, match="script"):
        engine.studio_generate(engine.UPLOAD, ref_clip, "", "  ", "Urdu", engine.BASE_ONLY, "Final — best quality", True, 1.0, 1.0)


def test_short_reference_is_rejected(engine, audio_store, tone, tmp_path):
    import pytest
    p = str(tmp_path / "tiny.wav"); audio_store[p] = (tone(1), 24000)
    with pytest.raises(ValueError, match="too short"):
        engine.pick_reference(p, "ur")


def test_batch_render_zip(engine, ref_clip):
    zpath, summary = engine.batch_render("پہلا اسکرپٹ۔\n---\nدوسرا اسکرپٹ۔", None, engine.UPLOAD, ref_clip, "", "Urdu",
                                         engine.BASE_ONLY, "Draft — fast preview", True, 1.0, 1.0)
    names = zipfile.ZipFile(zpath).namelist()
    assert len([n for n in names if n.endswith(".wav")]) == 2 and any(n.endswith(".srt") for n in names)
    assert "2 script(s)" in summary


def test_live_streams_one_chunk_per_sentence(engine, ref_clip):
    chunks = list(engine.live_tts(engine.UPLOAD, ref_clip, "", "پہلا۔ دوسرا۔ تیسرا۔", "Urdu", engine.BASE_ONLY, True, True))
    assert len(chunks) == 3 and all(c[0] == engine.SR and c[1].dtype.name == "int16" for c in chunks)


def test_voice_library_roundtrip(engine, ref_clip):
    msg = engine.save_voice("my voice!", ref_clip, "میرا نام علی ہے۔", "Urdu", True)
    assert "my_voice" in msg and "my_voice" in engine.list_voices()
    path, info = engine.voice_details("my_voice")
    assert path.endswith("ref.wav") and "میرا نام علی ہے" in info
    voice, _, _ = engine.resolve_voice("my_voice", None, "", "ur", True)
    assert "voice_clone_prompt" in voice


def test_history(engine, ref_clip):
    engine.studio_generate(engine.UPLOAD, ref_clip, "", "ہسٹری ٹیسٹ۔", "Urdu", engine.BASE_ONLY, "Draft — fast preview", True, 1.0, 1.0)
    stamp = engine.list_history()[0]
    wav, files, info = engine.open_history(stamp)
    assert os.path.exists(wav) and files and "ur" in info


def test_dataset_splits_and_transcript_fixes(engine):
    name = "speaker"; os.makedirs(engine.ds_dir(name), exist_ok=True)
    entries = [{"id": f"speaker_{i:05d}", "audio_path": f"/x/{i}.wav", "text": f"جملہ {i}", "language_id": "ur", "dur": 6.0, "lp": -0.2}
               for i in range(30)]
    engine.save_entries(name, entries)
    assert engine.write_splits(name, entries) == (28, 2)
    table = pd.DataFrame([[e["id"], e["dur"], e["text"]] for e in entries])
    table.iloc[0, 2] = "درست جملہ"; table.iloc[1, 2] = ""
    msg = engine.save_transcript_fixes(name, table)
    assert "1 transcript(s) fixed" in msg or "2 transcript(s) fixed" in msg
    assert "1 clip(s) removed" in msg and len(engine.load_entries(name)) == 29
    assert name in engine.list_datasets()


def test_stop_training_when_idle(engine):
    assert engine.stop_training() == "Nothing is running."


def test_speech_to_speech_keeps_urdu_in_urdu_script(engine, ref_clip, audio_store, tone, tmp_path, fake_model):
    """Whisper often labels spoken Urdu as Hindi (Devanagari). Auto mode must re-transcribe as Urdu."""
    mic = str(tmp_path / "mic.wav"); audio_store[mic] = (tone(3), 16000)
    calls = []
    def fake_transcribe(wav, sr, lang=None, task="transcribe", vad=True):
        calls.append((lang, task))
        if lang is None: return [(0, 1, "आज मौसम अच्छा है", -0.2, 0.0)], "hi"
        return [(0, 1, "آج موسم اچھا ہے", -0.2, 0.0)], lang
    saved = (engine.transcribe_segments, engine.M["asr"], engine.M["asr_failed"])
    engine.transcribe_segments, engine.M["asr"], engine.M["asr_failed"] = fake_transcribe, object(), False
    try:
        out = list(engine.speak_to_clone(mic, engine.UPLOAD, ref_clip, "", "Auto", "Same language", engine.BASE_ONLY, True, True))
    finally:
        engine.transcribe_segments, engine.M["asr"], engine.M["asr_failed"] = saved
    assert calls[:2] == [(None, "transcribe"), ("ur", "transcribe")]   # detect, then re-transcribe as Urdu
    assert "آج موسم اچھا ہے" in out[0][0] and "आज" not in out[-1][0]
    assert fake_model.calls[-1]["language"] == "ur"
