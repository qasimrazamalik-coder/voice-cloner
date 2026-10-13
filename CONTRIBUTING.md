# Contributing

Thanks for helping! Bug reports, Urdu pronunciation fixes and new features are all welcome.

## Reporting a bug
Open an issue with:
1. What you did (tab, settings, a short script if relevant)
2. What happened — paste the **🩺 Diagnostics → Run self-test** output and any red error message
3. Where you ran it (Colab T4 / A100, or local GPU + OS)

Please **don't upload voice recordings of other people** to issues.

## Development
```bash
git clone https://github.com/qasimrazamalik-coder/voice-cloner && cd voice-cloner
pip install pytest numpy pandas      # tests stub the heavy ML libraries, no GPU needed
pytest -q
```
- `voice_cloner/engine.py` — models, text, audio, training (no UI code)
- `voice_cloner/app.py` — Gradio interface only; every handler is wrapped so errors become toasts
- `tests/` — add a test for any bug you fix; `tests/test_app.py` checks every button is wired to a handler with the right inputs

Keep pull requests focused, and describe how you tested on a real GPU if the change touches generation or training.
