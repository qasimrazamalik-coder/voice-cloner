"""Command line.

    python -m voice_cloner app   [--home DIR] [--share] [--port 7860]
    python -m voice_cloner say   --voice NAME --text "..." [--lang ur|en] [--out out.wav]
    python -m voice_cloner say   --ref clip.wav --file script.txt --quality Draft
    python -m voice_cloner selftest
"""
import argparse, os, shutil, sys


def _quality(name):
    from .engine import QUALITY
    for q in QUALITY:
        if q.lower().startswith(name.lower()): return q
    raise SystemExit(f"--quality must be one of: {', '.join(q.split(' ')[0] for q in QUALITY)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="voice-cloner", description="Urdu + English voice cloning (OmniVoice)")
    p.add_argument("--home", help="storage folder for voices / adapters / history (default ~/VoiceCloner or $VOICE_CLONER_HOME)")
    p.add_argument("--work-dir", help="folder for large temporary training files")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("app", help="start the web app")
    a.add_argument("--share", action="store_true", help="create a public gradio.live link")
    a.add_argument("--host", default=None, help="e.g. 0.0.0.0 to allow other devices on your network")
    a.add_argument("--port", type=int, default=None)

    s = sub.add_parser("say", help="render a script to an audio file")
    who = s.add_mutually_exclusive_group(required=True)
    who.add_argument("--voice", help="name of a saved voice")
    who.add_argument("--ref", help="path to a reference clip")
    what = s.add_mutually_exclusive_group(required=True)
    what.add_argument("--text"); what.add_argument("--file", help="UTF-8 text file with the script")
    s.add_argument("--ref-text", default="", help="exact words spoken in --ref (optional)")
    s.add_argument("--lang", choices=["ur", "en"], default="ur")
    s.add_argument("--adapter", default=None, help="name of a trained adapter")
    s.add_argument("--quality", default="Final", help="Draft | Final | Max")
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--out", default="voice_cloner_output.wav")

    sub.add_parser("selftest", help="check GPU, models and accuracy end to end")
    args = p.parse_args(argv)

    from . import engine
    engine.configure(args.home, args.work_dir)

    if args.cmd in (None, "app"):
        from .app import launch
        launch(share=getattr(args, "share", False), server_name=getattr(args, "host", None), server_port=getattr(args, "port", None))
    elif args.cmd == "say":
        text = args.text if args.text is not None else open(args.file, encoding="utf-8").read()
        language = {v: k for k, v in engine.LANG.items()}[args.lang]
        wav, files, report, _ = engine.studio_generate(
            args.voice or engine.UPLOAD, args.ref, args.ref_text, text, language,
            args.adapter or engine.BASE_ONLY, _quality(args.quality), True, args.speed, 1.0,
            progress=lambda frac, desc="": print(f"  {desc}", file=sys.stderr))
        shutil.copy(wav, args.out)
        for f in files: shutil.copy(f, os.path.splitext(args.out)[0] + os.path.splitext(f)[1])
        print(report); print(f"\n✅ saved {args.out}")
    elif args.cmd == "selftest":
        print(engine.self_test().replace("\n\n", "\n"))


if __name__ == "__main__":
    main()
