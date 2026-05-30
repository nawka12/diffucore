# Diffucore

A clean, from-scratch diffusion **inference engine** in PyTorch.

Diffucore turns a model checkpoint plus a prompt into an image. It owns the full
generation path — checkpoint loading, text conditioning, the sampling/denoising
loop, and VAE decoding — and exposes a small Python API that a separate UI can
drive. It is **not** a node-graph editor; there is no workflow JSON and no node
system.

> **Status: pre-alpha.** The sampling/denoising core and checkpoint detection are
> implemented and tested (32 CPU tests). The model components (CLIP, VAE, UNet)
> and pipeline are specced with skeletons in place — see
> [`docs/ROADMAP.md`](docs/ROADMAP.md), [`docs/IMPLEMENTATION_SPEC.md`](docs/IMPLEMENTATION_SPEC.md),
> and [`docs/HANDOFF.md`](docs/HANDOFF.md).

## Design at a glance

```python
from diffucore import load_checkpoint, TextToImage   # planned public API

model = load_checkpoint("models/sd15.safetensors")
pipe = TextToImage(model)
image = pipe(prompt="a photo of an astronaut riding a horse",
             steps=20, cfg_scale=7.0, seed=0)
image.save("out.png")
```

The architecture and the rationale behind it live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why another engine

Existing engines are excellent but tend to be node-graph-first and carry large
surface areas. Diffucore is a focused library: a small, readable core that does
one thing — diffusion inference — and is easy to embed behind a custom UI.

## Development

The first target is **Stable Diffusion 1.5** text-to-image. CPU is supported for
testing; real generation targets CUDA (developed against an RTX 2060 12 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add torch from the appropriate index for your platform
pytest                            # runs the CPU-only test suite
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Diffucore is an independent implementation. Model architectures and sampling
algorithms are implemented from their original research publications.
