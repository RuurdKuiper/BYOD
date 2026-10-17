"""A small Gradio playground for adapters trained in the notebook."""
from pathlib import Path
import time

from .inference import denoise_stream, find_adapters, load_session, release_session
from .notebook import saved_checkpoint


class InferenceDemo:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.session = None
        self.default_checkpoint = saved_checkpoint(self.run_dir).name
        self.app = self._build()

    def generate(self, checkpoint, question, max_new_tokens=128, num_steps=64,
                 temperature=.7, top_k=20, seed=42, block_length=0, noise_level=1.,
                 retention="Lock accepted tokens", confidence_guided=True,
                 delay_endings=True, repetition_penalty=1., system_prompt="",
                 early_stopping=False, pause=0.):
        """Yield each denoising iteration directly to the browser."""
        if not question.strip():
            raise ValueError("Enter a question first.")
        modes = {"Refine every token": (False, True),
                 "Keep token positions": (True, False), "Lock accepted tokens": (True, True)}
        permanent, freeze = modes[retention]
        yield "Loading adapter…", "", ""
        if self.session is None or self.session.adapter_path != self.run_dir / checkpoint:
            release_session(self.session)
            self.session = None
            self.session = load_session(checkpoint, outputs_dir=self.run_dir)
        for text, status, html in denoise_stream(
            self.session, question=question, system_prompt=system_prompt,
            max_new_tokens=int(max_new_tokens), num_steps=int(num_steps),
            noise_level=float(noise_level), temperature=float(temperature), top_k=int(top_k), seed=int(seed),
            permanent_unmask=permanent, confidence_guided=bool(confidence_guided),
            proportional_unmask=False, freeze_retained_tokens=freeze,
            confidence_eos_eot_inf=bool(delay_endings), repetition_penalty=float(repetition_penalty),
            block_length=int(block_length) or None, early_stopping=bool(early_stopping),
        ):
            yield status, html, text
            if pause:
                time.sleep(float(pause))

    def _build(self):
        import gradio as gr
        with gr.Blocks(title="Ask your DLM", analytics_enabled=False) as app:
            gr.Markdown("# Ask your DLM\nWatch the answer take shape at each denoising iteration.")
            checkpoint = gr.Dropdown(find_adapters(self.run_dir), value=self.default_checkpoint, label="Saved adapter")
            question = gr.Textbox(label="Question", lines=3, placeholder="What would you like to ask?")
            with gr.Row():
                tokens = gr.Slider(8, 512, value=128, step=8, label="Answer tokens")
                steps = gr.Slider(1, 256, value=64, step=1, label="Denoising steps")
                temperature = gr.Slider(0, 2, value=.7, step=.05, label="Temperature")
            with gr.Accordion("More settings", open=False):
                with gr.Row():
                    top_k = gr.Slider(1, 100, value=20, step=1, label="Top-k")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    block = gr.Slider(0, 512, value=0, step=8, label="Block length (0 = full answer)")
                with gr.Row():
                    noise = gr.Slider(0, 1, value=1, step=.05, label="Re-mask probability")
                    retention = gr.Dropdown(["Refine every token", "Keep token positions", "Lock accepted tokens"],
                                            value="Lock accepted tokens", label="Token retention")
                    repetition = gr.Slider(1, 2, value=1, step=.05, label="Repetition penalty")
                system = gr.Textbox(label="System prompt", value="")
                guided = gr.Checkbox(value=True, label="Confidence-guided retention")
                delay = gr.Checkbox(value=True, label="Delay ending tokens")
                early = gr.Checkbox(value=False, label="Stop after two identical predictions")
                pause = gr.Slider(0, 1, value=0, step=.05, label="Pause between iterations (seconds)")
            with gr.Row():
                generate = gr.Button("Generate", variant="primary")
                stop = gr.Button("Stop")
            status = gr.Markdown("The adapter loads when you click Generate.")
            tokens_view = gr.HTML()
            answer = gr.Textbox(label="Current answer", lines=5, interactive=False)
            event = generate.click(
                self.generate,
                inputs=[checkpoint, question, tokens, steps, temperature, top_k, seed, block, noise,
                        retention, guided, delay, repetition, system, early, pause],
                outputs=[status, tokens_view, answer], concurrency_limit=1, stream_every=0,
            )
            stop.click(fn=None, cancels=[event], queue=False)
            gr.Markdown("Interactive refinement uses the original demo sampler. The link stays active while this Colab runtime runs.")
        return app

    def launch(self, **kwargs):
        return self.app.queue(default_concurrency_limit=1).launch(show_error=True, ssr_mode=False, **kwargs)

    def close(self):
        self.app.close()
        release_session(self.session)
        self.session = None
