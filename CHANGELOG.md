# Changelog

## 1.0.0 — 2026-10-13

First public release.

- **Studio** — script → mastered audio; best-of-N takes rendered in one GPU batch and judged by Whisper (words) + WavLM speaker verification (voice); 🟢🟡🔴 score per part; edit-and-retake one part; Draft / Final / Max quality; style presets; −14 LUFS mastering; WAV + MP3 + synced SRT.
- **Batch** — many scripts (separated by `---`) → one zip.
- **Live** — sentence-by-sentence streaming; speak into the mic and hear it back in the cloned voice, optionally translated to English. Urdu speech always stays in Urdu script (Whisper's Hindi/Devanagari mislabel is corrected).
- **Train** — dataset prep (auto-cut, auto-transcribe, editable transcripts), LoRA fine-tuning with live loss chart, presets + advanced controls, resume, automatic best-checkpoint selection against the base model.
- **Accuracy helpers** — reference-clip quality meter, clearest 5–10 s auto-picked with its transcript, Urdu numbers → words, pronunciation dictionary, Arabic-keyboard letter normalisation.
- Voice library, history, A/B compare, one-minute self-test, CLI (`voice-cloner app | say | selftest`), Colab notebook, CPU test suite + CI.
