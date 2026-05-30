# Diffucore

A clean, from-scratch diffusion **inference engine** in PyTorch.

Diffucore turns a model checkpoint plus a prompt into an image. It owns the full
generation path — checkpoint loading, text conditioning, the sampling/denoising
loop, and VAE decoding — and exposes a small Python API that a separate UI can
drive. It is **not** a node-graph editor; there is no workflow JSON and no node
system.

> **Status: pre-alpha, but end-to-end working.** Stable Diffusion 1.5 (512²) and
> SDXL (1024²) text-to-image both run: `load_checkpoint` → `TextToImage` produces
> a coherent, seed-reproducible image. The sampling core + checkpoint detection
> are unit tested (CPU); the model components (CLIP, OpenCLIP bigG, VAE, UNet) are
> verified on an RTX 2060 against HF `transformers`/`diffusers` as numerical
> oracles (text encoders and UNet match bit-for-bit in fp32; VAE round-trip 35 dB
> PSNR). See [`docs/ROADMAP.md`](docs/ROADMAP.md),
> [`docs/IMPLEMENTATION_SPEC.md`](docs/IMPLEMENTATION_SPEC.md),
> and [`docs/HANDOFF.md`](docs/HANDOFF.md).

## Design at a glance

```python
import torch
from diffucore import load_checkpoint, TextToImage

model = load_checkpoint("models/v1-5-pruned-emaonly.safetensors",
                        device="cuda", dtype=torch.float16)
pipe = TextToImage(model)
image = pipe(prompt="a photo of an astronaut riding a horse",
             negative_prompt="blurry, low quality",
             steps=20, cfg_scale=7.5, seed=0)
image.save("out.png")
```

The architecture and the rationale behind it live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why another engine

Existing engines are excellent but tend to be node-graph-first and carry large
surface areas. Diffucore is a focused library: a small, readable core that does
one thing — diffusion inference — and is easy to embed behind a custom UI.

## Development

The first targets are **Stable Diffusion 1.5** and **SDXL** text-to-image. CPU is
supported for testing; real generation targets CUDA (developed against an RTX
2060 12 GB — SDXL at 1024² needs ~11 GB resident).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # numpy/safetensors/tokenizers/Pillow/einops/pytest
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build for your GPU
pytest                            # runs the CPU-only test suite (32 tests)
```

For real generation, fetch an SD1.5 checkpoint (e.g. `v1-5-pruned-emaonly.safetensors`)
into `models/` (gitignored). Numerical verification additionally uses HF
`transformers`/`diffusers` as oracles — they are dev-only and not runtime deps.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Diffucore is an independent implementation. Model architectures and sampling
algorithms are implemented from their original research publications.
