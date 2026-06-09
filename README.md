# Diffucore

**A clean, from-scratch diffusion inference engine in PyTorch.**

Point it at a checkpoint, give it a prompt, get an image. Diffucore owns the
entire generation path — checkpoint loading, text conditioning, the
sampling/denoising loop, and VAE decoding — in one small, readable library you
can `import` or embed behind your own UI.

![status](https://img.shields.io/badge/status-alpha-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

```python
import torch
from diffucore import load_checkpoint, TextToImage

model = load_checkpoint("models/sdxl.safetensors", device="cuda", dtype=torch.float16)
image = TextToImage(model)(
    prompt="a watercolor fox in a misty forest",
    negative_prompt="blurry, low quality",
    steps=25, cfg_scale=6.0, width=1024, height=1024, seed=0,
)
image.save("fox.png")
```

Supports SD 1.5, SDXL, **Anima**, and the **FLUX** family — txt2img, img2img, and
inpaint, the same way across all of them. The full feature list, every model's
examples, and performance numbers live in
**[docs/USAGE.md](docs/USAGE.md)**.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
# then install the CUDA build of torch for your GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Runtime dependencies are small: `torch`, `numpy`, `safetensors`, `tokenizers`,
`Pillow`, `einops`, `tqdm`.

## Documentation

- **[Usage & examples](docs/USAGE.md)** — features, per-model code, samplers /
  schedulers, supported models, performance
- [Architecture & rationale](docs/ARCHITECTURE.md)
- [Roadmap & verified milestones](docs/ROADMAP.md)
- [Runtime / VRAM (offload + tiled VAE)](docs/RUNTIME_SPEC.md)
- [Implementation notes](docs/IMPLEMENTATION_SPEC.md) · [build log](docs/HANDOFF.md)

## Status

Diffucore is in **alpha**. The engine is end-to-end working and seed-reproducible
on SD1.5 / SDXL / Anima, with the sampling core and checkpoint detection
unit-tested and every shipped backbone verified against reference
implementations. APIs may still shift before 1.0. CPU works for testing; real
generation targets CUDA.

**FLUX runs on real weights.** FLUX.1 (schnell) and FLUX.2 (Klein-4B) pass a
strict no-missing-keys load and produce coherent, prompt-faithful, deterministic
images on a 24 GB GPU (RTX 4090, bf16, `offload="stream"`). Bit-exact parity
against the Black Forest Labs / ComfyUI reference is still pending; the FLUX.1-dev
and FLUX.2-Dev (Mistral-3) paths are implemented to spec but not yet
hardware-verified (the Dev path's Tekken tokenizer isn't vendored — pass
`mistral_tokenizer_path`). See [`docs/ROADMAP.md`](docs/ROADMAP.md) for
per-component status.

**For ComfyUI users:** samplers and schedulers follow ComfyUI's k-diffusion
conventions, but the SDE samplers re-inject seeded Gaussian noise instead of
Brownian-tree noise — results are coherent and reproducible, but not bit-identical
to a ComfyUI render.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Diffucore is an
independent implementation; model architectures and sampling algorithms are
implemented from their original research publications.
