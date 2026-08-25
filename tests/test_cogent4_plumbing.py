"""CPU-only pins for the Cogent4 gate option at the public pipeline boundary."""

from types import SimpleNamespace

from PIL import Image


def _anima_model():
    return SimpleNamespace(spec=SimpleNamespace(architecture="anima", image_size=64))


def test_text_to_image_forwards_gate_reduce_to_anima(monkeypatch):
    import diffucore.pipelines.text_to_image as mod

    seen = {}
    sentinel = object()

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(mod, "anima_text_to_image", fake)
    out = mod.TextToImage(_anima_model())("x", gate_reduce="per_channel")
    assert out is sentinel
    assert seen["gate_reduce"] == "per_channel"


def test_image_to_image_forwards_gate_reduce_to_anima(monkeypatch):
    import diffucore.pipelines._anima as anima
    import diffucore.pipelines.image_to_image as mod

    seen = {}
    sentinel = object()

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(anima, "anima_img2img", fake)
    out = mod.ImageToImage(_anima_model())(
        "x", Image.new("RGB", (64, 64)), gate_reduce="per_channel",
    )
    assert out is sentinel
    assert seen["gate_reduce"] == "per_channel"


def test_inpaint_forwards_gate_reduce_to_anima(monkeypatch):
    import diffucore.pipelines._anima as anima
    import diffucore.pipelines.inpaint as mod

    seen = {}
    sentinel = object()

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(anima, "anima_img2img", fake)
    out = mod.Inpaint(_anima_model())(
        "x", Image.new("RGB", (64, 64)), Image.new("L", (64, 64)),
        gate_reduce="per_channel",
    )
    assert out is sentinel
    assert seen["gate_reduce"] == "per_channel"
