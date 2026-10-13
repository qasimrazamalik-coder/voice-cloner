"""Audio assembly, loudness matching, subtitles, reference quality meter, dataset segmentation."""
import numpy as np


def test_assemble_matches_loudness_and_inserts_pauses(engine, tone):
    sr = 24000
    a = engine.tidy_chunk(tone(1, amp=0.5), sr)
    b = engine.tidy_chunk(tone(1, amp=0.3), sr)
    silent = engine.tidy_chunk(np.zeros(10, np.float32), sr)
    audio, spans = engine.assemble([a, b, silent], [0.45, 0.2, 0], sr)
    rms = [float(np.sqrt(np.mean(audio[int(s * sr): int(e * sr)] ** 2))) for s, e in spans[:2]]
    assert abs(rms[0] - rms[1]) / rms[0] < 0.05
    assert abs((spans[1][0] - spans[0][1]) - 0.45) < 0.01
    assert np.isfinite(audio).all()


def test_tidy_chunk_never_empty(engine):
    assert len(engine.tidy_chunk(np.zeros(0, np.float32), 24000)) > 0


def test_srt_format_and_timing(engine):
    srt = engine.make_srt(["اگر چاند آج رات غائب ہو جائے تو کیا ہوگا اور سمندر کی لہریں", "دوسرا"], [(0.0, 3.0), (3.5, 4.5)])
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert "00:00:03,500 --> 00:00:04,500\nدوسرا" in srt
    assert engine.srt_time(3723.25) == "01:02:03,250"


def test_reference_quality_meter(engine, tone, audio_store, tmp_path):
    rng = np.random.default_rng(0)
    clean = np.concatenate([tone(4), np.zeros(24000, np.float32), tone(4)])
    noisy = clean + rng.normal(0, 0.08, len(clean)).astype(np.float32)
    short = tone(2)
    for name, y in (("clean", clean), ("noisy", noisy), ("short", short)):
        audio_store[str(tmp_path / f"{name}.wav")] = (y, 24000)
    assert engine.reference_stats(clean, 24000)["snr_db"] > 25
    assert "✅ Great" in engine.analyze_reference(str(tmp_path / "clean.wav"))
    assert "Noisy" in engine.analyze_reference(str(tmp_path / "noisy.wav")) or "noise" in engine.analyze_reference(str(tmp_path / "noisy.wav"))
    assert "❌" in engine.analyze_reference(str(tmp_path / "short.wav"))
    assert engine.analyze_reference(None) == ""


def test_merge_segments_cuts_on_pauses(engine):
    segs = [(0, 2, "a", -.2, 0), (2.3, 5, "b", -.3, 0), (5.2, 9, "c", -.1, 0),
            (10.5, 12, "d", -.2, 0), (12.1, 14, "e", -.4, 0), (20, 21, "x", -.1, 0)]
    assert [c[:3] for c in engine.merge_segments(segs)] == [[0, 9, "a b c"], [10.5, 14, "d e"]]
    long = [(i * 1.0, i * 1.0 + 0.9, "w", -.1, 0) for i in range(40)]
    assert all(3 <= c[1] - c[0] <= 15 for c in engine.merge_segments(long))
