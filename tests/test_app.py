"""The web app builds, and every button/dropdown is wired to a handler with the right number of inputs."""
import inspect


def test_app_builds_and_launches(built_app):
    app_mod, blocks, events = built_app
    assert len(events) >= 25
    assert app_mod.launch(share=True)["share"] is True


def test_every_handler_accepts_its_inputs(built_app):
    _, _, events = built_app
    problems = []
    for ev in events:
        try:
            sig = inspect.signature(ev.fn)
        except (TypeError, ValueError):
            continue
        params = [p for p in sig.parameters.values() if p.name != "progress"]
        if any(p.kind == p.VAR_POSITIONAL for p in params):
            continue
        required = [p for p in params if p.default is p.empty]
        if not (len(required) <= len(ev.inputs) <= len(params)):
            problems.append(f"{getattr(ev.fn, '__name__', ev.fn)}: {len(ev.inputs)} inputs vs {len(params)} params")
    assert not problems, problems


def test_errors_become_toasts_not_crashes(built_app, engine):
    import gradio as gr
    app_mod, _, _ = built_app
    run = app_mod.safe(app_mod.studio_run, 5)
    out = run(engine.UPLOAD, None, "", "", "Urdu", engine.BASE_ONLY, "Final — best quality", True, 1.0, 1.0)
    assert len(out) == 5 and any("script" in w for w in gr.WARNINGS)


def test_cli_parses(engine):
    from voice_cloner.__main__ import _quality
    assert _quality("draft").startswith("Draft") and _quality("Max").startswith("Max")
