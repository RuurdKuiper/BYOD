from pathlib import Path
from types import SimpleNamespace

from diffusion_lm import demo as demo_module


def test_interactive_demo_forwards_controls_streams_each_step_and_switches_adapters(tmp_path, monkeypatch):
    for name in ("best", "interrupted"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "adapter_config.json").write_text("{}")
        (folder / "adapter_model.safetensors").write_bytes(b"saved")
    loaded, released, settings = [], [], []
    def load(selection, outputs_dir):
        session = SimpleNamespace(adapter_path=Path(outputs_dir) / selection)
        loaded.append(session)
        return session
    def stream(session, **kwargs):
        settings.append(kwargs)
        for step in range(kwargs["num_steps"]):
            yield f"answer {step}", f"step {step + 1}", f"<span>{step}</span>"
    monkeypatch.setattr(demo_module, "load_session", load)
    monkeypatch.setattr(demo_module, "release_session", released.append)
    monkeypatch.setattr(demo_module, "denoise_stream", stream)
    demo = demo_module.InferenceDemo(tmp_path)
    assert demo.default_checkpoint == "interrupted"
    states = list(demo.generate("best", "Question", num_steps=3, temperature=.2,
        block_length=16, retention="Keep token positions", seed=17, delay_endings=False))
    assert len(states) == 4
    assert states[-1] == ("step 3", "<span>2</span>", "answer 2")
    assert settings[0]["temperature"] == .2 and settings[0]["seed"] == 17
    assert settings[0]["block_length"] == 16
    assert settings[0]["permanent_unmask"] is True and settings[0]["freeze_retained_tokens"] is False
    assert settings[0]["confidence_eos_eot_inf"] is False
    list(demo.generate("best", "Again", num_steps=1))
    assert len(loaded) == 1
    list(demo.generate("interrupted", "Again", num_steps=1))
    assert len(loaded) == 2 and loaded[0] in released
    event = next(fn for fn in demo.app.fns.values() if fn.fn == demo.generate)
    assert event.stream_every == 0 and event.concurrency_limit == 1
    demo.close()
    assert demo.session is None and loaded[-1] in released
