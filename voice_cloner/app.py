"""Voice Cloner web app (Gradio). ``build_app()`` returns the Blocks; ``launch()`` serves it."""
import gradio as gr

from .engine import *  # noqa: F401,F403  (engine functions read their storage paths at call time)
from . import __version__

# ---------- safety wrappers: every error becomes a friendly toast, never a crash ----------
def _keep(n): return tuple(gr.update() for _ in range(n)) if n > 1 else gr.update()

def safe(fn, n_out):
    def run(*args, progress=gr.Progress()):
        try:
            return fn(*args, progress=progress) if "progress" in fn.__code__.co_varnames else fn(*args)
        except Exception as e:
            gr.Warning(f"{type(e).__name__}: {e}"); print("⚠️", repr(e)); return _keep(n_out)
    run.__name__ = getattr(fn, "__name__", "handler")
    return run

def safe_gen(fn, n_out):
    def run(*args):
        try:
            for item in fn(*args):
                yield item
        except Exception as e:
            gr.Warning(f"{type(e).__name__}: {e}"); print("⚠️", repr(e)); yield _keep(n_out)
    run.__name__ = getattr(fn, "__name__", "handler")
    return run

def voices_update(): return gr.update(choices=[UPLOAD] + list_voices())
def adapters_update(): return gr.update(choices=list_adapters())

def studio_run(*a, progress=None):
    wav, files, rep, ids = studio_generate(*a, progress=progress)
    return wav, files, rep, gr.update(choices=ids, value=ids[0] if ids else None), gr.update(choices=list_history())

def batch_run(*a, progress=None):
    zpath, summary = batch_render(*a, progress=progress)
    return zpath, summary, gr.update(choices=list_history())

def prep_run(*a):
    for log, summary, rows in prepare_dataset(*a):
        yield log, summary, (rows if rows else gr.update())

def speak_run(*a):
    for md, chunk in speak_to_clone(*a):
        yield md, (chunk if chunk is not None else gr.update())

def save_voice_run(name, ref, ref_text, language, denoise):
    return save_voice(name, ref, ref_text, language, denoise), voices_update(), voices_update()

def preset_fill(p):
    v = TRAIN_PRESETS[p]
    return gr.update(value=v["steps"]), gr.update(value=v["lora_r"]), gr.update(value=v["lora_r"] * 2), gr.update(value=v["lr"])

def style_fill(s): return gr.update(value=STYLE[s][0]), gr.update(value=STYLE[s][1])

def refresh_all():
    v, a = voices_update(), adapters_update()
    return v, a, v, a, gr.update(choices=list_history()), gr.update(choices=list_datasets()), gpu_status()

def after_prep(): return gr.update(choices=list_datasets()), voices_update(), gpu_status()
def after_eval(): return adapters_update(), adapters_update(), gpu_status()
def open_dataset(n): return (n, *load_dataset_table(n))
def pron_preview(t, l): return f"→ {prepare_text(t, LANG[l])}" if t else ""

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto}
.rtl textarea, .rtl input {direction: rtl; text-align: right; font-size: 1.12em; line-height: 1.9}
#hero h1 {margin-bottom: 0} #hero p {opacity: .75; margin-top: .2em}
#status {font-size: .88em; opacity: .85}
.big-btn button, button.big-btn {min-height: 52px; font-size: 1.05em}
@media (max-width: 640px) { .gradio-container {padding: 6px !important} h1 {font-size: 1.4em !important} }
"""

SAMPLE_UR = "اگر چاند آج رات غائب ہو جائے تو کیا ہوگا؟"

def build_app():
    with gr.Blocks(title="Voice Cloner", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet"), css=CSS) as app:
        with gr.Row(equal_height=True):
            gr.Markdown(f"# 🎙️ Voice Cloner\nUrdu + English · real-time · train your own voice · free & open source · v{__version__}", elem_id="hero")
            refresh_btn = gr.Button("↻ Refresh lists", scale=0, min_width=130)
        status = gr.Markdown(gpu_status(), elem_id="status")

        # shared voice controls (Studio, Batch, Live, Compare)
        with gr.Accordion("🎤 Voice — pick a saved voice or upload a clip", open=True):
            with gr.Row():
                voice_choice = gr.Dropdown([UPLOAD] + list_voices(), value=UPLOAD, label="Voice", min_width=200)
                adapter = gr.Dropdown(list_adapters(), value=BASE_ONLY, label="Trained adapter", min_width=200)
                language = gr.Radio(list(LANG), value="Urdu", label="Language", min_width=200)
            with gr.Row():
                with gr.Column(min_width=280):
                    ref = gr.Audio(label="Reference clip (any length — the clearest 5–10 s is picked automatically)",
                                   type="filepath", sources=["upload", "microphone"])
                    ref_quality = gr.Markdown()
                with gr.Column(min_width=280):
                    ref_text = gr.Textbox(label="Exact words in the clip (optional, improves accuracy)", lines=3, elem_classes="rtl")
                    denoise = gr.Checkbox(True, label="Clean background noise from the clip")
                    with gr.Row():
                        vname = gr.Textbox(label="Save as voice", placeholder="my_voice_urdu", scale=3)
                        save_btn = gr.Button("💾 Save", scale=1)
                    save_msg = gr.Markdown()

        with gr.Tabs():
            # ═══════════ Studio ═══════════
            with gr.Tab("🎬 Studio"):
                with gr.Row():
                    with gr.Column(scale=5, min_width=320):
                        text = gr.Textbox(label="Script — blank line = paragraph pause", lines=10, elem_classes="rtl", placeholder=SAMPLE_UR)
                        with gr.Row():
                            quality = gr.Dropdown(list(QUALITY), value="Final — best quality", label="Quality", min_width=160)
                            style = gr.Dropdown(list(STYLE), value="Natural", label="Style", min_width=160)
                        with gr.Accordion("Fine controls", open=False):
                            speed = gr.Slider(0.7, 1.3, 1.0, step=0.01, label="Speed")
                            pause_scale = gr.Slider(0.4, 2.0, 1.0, step=0.05, label="Pause length")
                        go = gr.Button("🔊 Generate", variant="primary", elem_classes="big-btn")
                    with gr.Column(scale=4, min_width=320):
                        out = gr.Audio(label="Mastered result (−14 LUFS)", type="filepath")
                        files = gr.File(label="Downloads — MP3 + subtitles (.srt)", file_count="multiple")
                        report = gr.Markdown()
                        with gr.Group():
                            gr.Markdown("**✂️ Fix one part** — pick it, edit if needed, retake")
                            with gr.Row():
                                sent_idx = gr.Dropdown([], label="Part #", min_width=100)
                                retake_btn = gr.Button("🔁 Retake", min_width=100)
                            sent_text = gr.Textbox(label="Part text", lines=2, elem_classes="rtl")
                            sent_audio = gr.Audio(label="This part", type="numpy")

            # ═══════════ Batch ═══════════
            with gr.Tab("📦 Batch"):
                gr.Markdown("Render many scripts in one go (e.g. a week of reels). Separate scripts with a line containing only `---`. "
                            "Uses the voice, language and Studio quality/style settings. You get one zip with WAV + MP3 + SRT per script.")
                with gr.Row():
                    with gr.Column(scale=5, min_width=320):
                        b_text = gr.Textbox(label="Scripts", lines=12, elem_classes="rtl",
                                            placeholder=f"{SAMPLE_UR}\n---\nاگر زمین گھومنا بند کر دے تو؟")
                        b_file = gr.File(label="…or a .txt file", file_types=[".txt"], type="filepath")
                        b_btn = gr.Button("📦 Render all", variant="primary", elem_classes="big-btn")
                    with gr.Column(scale=4, min_width=320):
                        b_zip = gr.File(label="Zip")
                        b_report = gr.Markdown()

            # ═══════════ Live ═══════════
            with gr.Tab("⚡ Live"):
                gr.Markdown("Audio starts playing after the **first sentence** while the rest renders. Uses the voice chosen above.")
                with gr.Row():
                    with gr.Column(min_width=320):
                        gr.Markdown("### ⌨️ Type → hear")
                        live_text = gr.Textbox(label="Text", lines=5, elem_classes="rtl", placeholder="جو بھی لکھیں، فوراً سنیں…")
                        live_fast = gr.Checkbox(True, label="Fastest (lower quality steps)")
                        live_btn = gr.Button("▶️ Speak now", variant="primary", elem_classes="big-btn")
                        live_audio = gr.Audio(label="Live output", streaming=True, autoplay=True)
                    with gr.Column(min_width=320):
                        gr.Markdown("### 🎙️ Speak → hear it in the cloned voice")
                        mic = gr.Audio(label="Record yourself", sources=["microphone"], type="filepath")
                        with gr.Row():
                            in_lang = gr.Radio(["Auto", "Urdu", "English"], value="Auto", label="You speak")
                            out_mode = gr.Radio(["Same language", "English (translate)"], value="Same language", label="Cloned voice says")
                        s2s_fast = gr.Checkbox(True, label="Fastest")
                        s2s_btn = gr.Button("🔁 Clone my speech", variant="primary", elem_classes="big-btn")
                        s2s_text = gr.Markdown()
                        s2s_audio = gr.Audio(label="Cloned speech", streaming=True, autoplay=True)

            # ═══════════ Train ═══════════
            with gr.Tab("🧠 Train"):
                gr.Markdown("Teach the model one voice from **15–60 min** of recordings. One speaker · quiet room · no music. "
                            "Training frees the GPU first; the Studio reloads models automatically afterwards.")
                with gr.Accordion("① Dataset — upload, auto-cut, auto-transcribe, fix transcripts", open=True):
                    with gr.Row():
                        t_name = gr.Textbox(label="Voice name", placeholder="my_voice", min_width=160)
                        t_lang = gr.Radio(list(LANG), value="Urdu", label="Recording language", min_width=160)
                        t_existing = gr.Dropdown(list_datasets(), label="…or open an existing dataset", min_width=160)
                    t_files = gr.File(label="Recordings (wav / mp3 / m4a / flac / ogg)", file_count="multiple", type="filepath")
                    with gr.Row():
                        t_denoise = gr.Checkbox(True, label="Denoise recordings")
                        t_min = gr.Slider(2, 6, 3, step=0.5, label="Min clip (s)")
                        t_max = gr.Slider(8, 20, 15, step=1, label="Max clip (s)")
                    prep_btn = gr.Button("✂️ Prepare dataset", variant="primary")
                    prep_log = gr.Textbox(label="Log", lines=6, max_lines=12, autoscroll=True)
                    prep_summary = gr.Markdown()
                    t_table = gr.Dataframe(headers=["id", "sec", "transcript (edit to fix · empty = drop clip)"],
                                           datatype=["str", "number", "str"], interactive=True, wrap=True, col_count=(3, "fixed"))
                    fix_btn = gr.Button("💾 Save transcript fixes"); fix_msg = gr.Markdown()
                with gr.Accordion("② Train", open=True):
                    t_preset = gr.Dropdown(list(TRAIN_PRESETS), value="Standard — 20–40 min", label="Preset")
                    with gr.Accordion("Advanced fine-tuning", open=False):
                        with gr.Row():
                            t_steps = gr.Number(1500, label="Steps", precision=0, minimum=50)
                            t_lr = gr.Number(1e-4, label="Learning rate")
                            t_save = gr.Number(250, label="Checkpoint every N steps", precision=0, minimum=50)
                        with gr.Row():
                            t_r = gr.Slider(4, 128, 32, step=4, label="LoRA rank (capacity)")
                            t_alpha = gr.Slider(4, 256, 64, step=4, label="LoRA alpha (strength, usually 2× rank)")
                            t_drop = gr.Slider(0.0, 0.3, 0.05, step=0.01, label="LoRA dropout")
                        with gr.Row():
                            t_targets = gr.Radio(["Attention + MLP (stronger)", "Attention only (lighter)"], value="Attention + MLP (stronger)", label="Layers adapted")
                            t_bt = gr.Slider(1024, 8192, 4096, step=512, label="Batch tokens (lower if out of memory)")
                            t_acc = gr.Slider(1, 8, 2, step=1, label="Gradient accumulation")
                            t_warm = gr.Slider(0.0, 0.1, 0.02, step=0.01, label="Warm-up ratio")
                        t_resume = gr.Checkbox(False, label="Resume from the last checkpoint (same session)")
                    with gr.Row():
                        train_btn = gr.Button("🚀 Start training", variant="primary", elem_classes="big-btn")
                        stop_btn = gr.Button("⏹️ Stop", variant="stop", elem_classes="big-btn")
                    t_status = gr.Markdown()
                    t_plot = gr.LinePlot(x="step", y="loss", color="series", title="Loss (lower = learning)", height=260)
                    t_log = gr.Textbox(label="Training log", lines=8, max_lines=14, autoscroll=True)
                with gr.Accordion("③ Pick the best checkpoint & export", open=True):
                    gr.Markdown("Each checkpoint speaks held-out sentences; Whisper + speaker verification score them against the base model. The winner is saved as your adapter.")
                    with gr.Row():
                        t_nsent = gr.Slider(2, 10, 4, step=1, label="Test sentences per checkpoint")
                        eval_btn = gr.Button("🏆 Pick best checkpoint", variant="primary")
                    eval_out = gr.Markdown()

            # ═══════════ Compare ═══════════
            with gr.Tab("⚖️ Compare"):
                gr.Markdown("Same sentence · base model vs your trained adapter · uses the voice chosen above.")
                cmp_text = gr.Textbox(label="Test sentence", lines=2, elem_classes="rtl", value="یہ میری آواز کا ایک تجربہ ہے، آپ کو کون سی بہتر لگی؟")
                cmp_adapter = gr.Dropdown(list_adapters(), value=list_adapters()[-1], label="Adapter to test")
                cmp_btn = gr.Button("⚖️ Compare", variant="primary")
                with gr.Row():
                    with gr.Column(min_width=280):
                        a_audio = gr.Audio(label="Base model", type="numpy"); a_score = gr.Markdown()
                    with gr.Column(min_width=280):
                        b_audio = gr.Audio(label="Your adapter", type="numpy"); b_score = gr.Markdown()

            # ═══════════ Voices ═══════════
            with gr.Tab("🗂️ Voices"):
                voices_list = gr.Dropdown([UPLOAD] + list_voices(), label="Saved voices")
                voice_preview = gr.Audio(label="Reference clip", type="filepath"); voice_info = gr.Markdown()

            # ═══════════ Pronunciation ═══════════
            with gr.Tab("🗣️ Pronunciation"):
                gr.Markdown("One rule per line: `written = how to say it`. Numbers are spoken as words automatically (2026 → دو ہزار چھبیس).")
                pron_box = gr.Textbox(value=pron_to_text(load_pron()), lines=12, label="Rules", placeholder="Primeura = پرائم یورا\nAI = اے آئی")
                pron_btn = gr.Button("💾 Save rules", variant="primary"); pron_msg = gr.Markdown()
                pron_test = gr.Textbox(label="Try it — type text to see what will be spoken", elem_classes="rtl"); pron_out = gr.Markdown()

            # ═══════════ History ═══════════
            with gr.Tab("🕘 History"):
                hist = gr.Dropdown(list_history(), label="Past renders (newest first)")
                hist_info = gr.Markdown()
                hist_audio = gr.Audio(label="Audio", type="filepath"); hist_files = gr.File(label="MP3 + subtitles", file_count="multiple")

            # ═══════════ Diagnostics ═══════════
            with gr.Tab("🩺 Diagnostics"):
                gr.Markdown("Runs a 1-minute end-to-end check (GPU, models, Urdu + English synthesis, accuracy judges). Run it first in a new session.")
                diag_btn = gr.Button("🩺 Run self-test", variant="primary"); diag_out = gr.Markdown()

        # ═══════════ wiring ═══════════
        voice_inputs = [voice_choice, ref, ref_text]
        render_settings = [language, adapter, quality, denoise, speed, pause_scale]
        refresh_btn.click(refresh_all, None, [voice_choice, adapter, voices_list, cmp_adapter, hist, t_existing, status])
        ref.change(safe(analyze_reference, 1), ref, ref_quality)
        save_btn.click(safe(save_voice_run, 3), [vname, ref, ref_text, language, denoise], [save_msg, voice_choice, voices_list])

        style.change(style_fill, style, [speed, pause_scale])
        go.click(safe(studio_run, 5), voice_inputs + [text] + render_settings,
                 [out, files, report, sent_idx, hist]).then(gpu_status, None, status)
        sent_idx.change(safe(sentence_preview, 2), sent_idx, [sent_audio, sent_text])
        retake_btn.click(safe(studio_retake, 4), [sent_idx, sent_text], [out, files, report, sent_audio])

        b_btn.click(safe(batch_run, 3), [b_text, b_file] + voice_inputs + render_settings, [b_zip, b_report, hist])

        live_btn.click(safe_gen(live_tts, 1), voice_inputs + [live_text, language, adapter, denoise, live_fast], live_audio)
        s2s_btn.click(safe_gen(speak_run, 2), [mic] + voice_inputs + [in_lang, out_mode, adapter, denoise, s2s_fast], [s2s_text, s2s_audio])

        t_existing.change(open_dataset, t_existing, [t_name, t_table, prep_summary])
        prep_btn.click(safe_gen(prep_run, 3), [t_name, t_lang, t_files, t_denoise, t_min, t_max], [prep_log, prep_summary, t_table]) \
            .then(after_prep, None, [t_existing, voice_choice, status])
        fix_btn.click(safe(save_transcript_fixes, 1), [t_name, t_table], fix_msg)
        t_preset.change(preset_fill, t_preset, [t_steps, t_r, t_alpha, t_lr])
        train_evt = train_btn.click(safe_gen(train_voice, 3),
                                    [t_name, t_steps, t_r, t_alpha, t_drop, t_targets, t_lr, t_bt, t_acc, t_warm, t_save, t_resume],
                                    [t_log, t_plot, t_status])
        stop_btn.click(stop_training, None, t_status, cancels=[train_evt])
        eval_btn.click(safe_gen(evaluate_checkpoints, 1), [t_name, t_nsent], eval_out).then(after_eval, None, [adapter, cmp_adapter, status])

        cmp_btn.click(safe(compare, 4), voice_inputs + [cmp_text, language, cmp_adapter, denoise], [a_audio, a_score, b_audio, b_score])
        voices_list.change(safe(voice_details, 2), voices_list, [voice_preview, voice_info])
        pron_btn.click(save_pron, pron_box, pron_msg)
        pron_test.change(pron_preview, [pron_test, language], pron_out)
        hist.change(safe(open_history, 3), hist, [hist_audio, hist_files, hist_info])
        diag_btn.click(self_test, None, diag_out)
    return app


def launch(share=False, server_name=None, server_port=None, **kwargs):
    """Build and serve the app. One GPU job at a time keeps memory predictable."""
    app = build_app()
    app.queue(default_concurrency_limit=1)
    return app.launch(share=share, server_name=server_name, server_port=server_port, show_error=True, **kwargs)
