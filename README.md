# Diffucore

A clean, from-scratch diffusion **inference engine** in PyTorch.

Diffucore turns a model checkpoint plus a prompt into an image. It owns the full
generation path — checkpoint loading, text conditioning, the sampling/denoising
loop, and VAE decoding — and exposes a small Python API that a separate UI can
drive. It is **not** a node-graph editor; there is no workflow JSON and no node
system.

> **Status: alpha.** Stable Diffusion 1.5 (512²) and SDXL (1024²) run
> text-to-image, image-to-image, and inpainting — `load_checkpoint` →
> `TextToImage` / `ImageToImage` / `Inpaint` produces a coherent,
> seed-reproducible image. Both **epsilon- and v-prediction** checkpoints are
> supported (auto-detected), including **zero-terminal-SNR** schedules with CFG
> rescale, and **LoRA / LoKr** adapters fuse in at load time. **Anima**
> (CircleStone Labs' 2 B DiT built on Cosmos-Predict2 with Qwen3-0.6B +
> Qwen-Image VAE) is integrated as the first DiT family —
> `load_anima_checkpoint(dit, vae, te)` → `TextToImage` generates a coherent
> 1024² image (~46 s / 20 steps on an RTX 2060, ~8.6 GB peak; flow-matching with
> shift=3, CFG, seed-reproducible). Ten samplers and several schedulers are
> available, with the DPM++ / ER-SDE family made flow-aware so they drive Anima
> too (see [Samplers & schedulers](#samplers--schedulers)). The sampling core +
> checkpoint detection are unit tested (CPU); the model components (CLIP,
> OpenCLIP bigG, VAE, UNet, Qwen3) are verified on an RTX 2060 against HF
> `transformers`/`diffusers` as numerical oracles (text encoders and UNet match
> bit-for-bit in fp32; VAE round-trip 35 dB PSNR on SD, 49 dB on Qwen-Image).
> See [`docs/ROADMAP.md`](docs/ROADMAP.md),
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

## Samplers & schedulers

Pick them per call via `sampler=` and `scheduler=`:

```python
image = pipe(prompt, negative_prompt,
             steps=32, cfg_scale=4.5, sampler="dpmpp_2m", scheduler="sgm_uniform")
```

- **Samplers** — `euler`, `heun`, `euler_ancestral`, `dpm_2`, `dpm_2_ancestral`,
  `dpmpp_2m`, `dpmpp_sde`, `dpmpp_2m_sde`, `dpmpp_3m_sde`, `er_sde`. The DPM++ and
  ER-SDE family are flow-aware (half-logSNR mapping), so they drive Anima as well
  as SD/SDXL.
- **Schedulers** — SD/SDXL: `karras`, `exponential`, `polyexponential`,
  `sgm_uniform`, `simple`. Anima (flow): `flow` (the default rectified-flow
  schedule), `sgm_uniform`, `simple`.

Samplers and schedulers follow the conventions of (and are cross-checked
against) ComfyUI's k-diffusion implementations. The SDE samplers re-inject
seeded Gaussian noise rather than ComfyUI's Brownian-tree noise — output is
coherent and seed-reproducible but not bit-identical to a ComfyUI render.

## Why another engine

Existing engines are excellent but tend to be node-graph-first and carry large
surface areas. Diffucore is a focused library: a small, readable core that does
one thing — diffusion inference — and is easy to embed behind a custom UI.

## Development

The first targets are **Stable Diffusion 1.5** and **SDXL** text-to-image;
**Anima** is the first DiT family on top of the same engine. CPU is supported
for testing; real generation targets CUDA (developed against an RTX 2060 12 GB).
SDXL at 1024² needs ~10 GB resident, or ~6.6 GB with the opt-in
`DevicePolicy(offload=True)` sequential CPU offload + tiled VAE decode (see
[`docs/RUNTIME_SPEC.md`](docs/RUNTIME_SPEC.md)); Anima at 1024² currently runs
~8.6 GB resident (offload-aware for the Anima path is a follow-up).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # numpy/safetensors/tokenizers/Pillow/einops/pytest
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build for your GPU
pytest                            # runs the CPU-only test suite
```

For real generation, fetch an SD1.5 checkpoint (e.g. `v1-5-pruned-emaonly.safetensors`)
into `models/` (gitignored). Numerical verification additionally uses HF
`transformers`/`diffusers` as oracles — they are dev-only and not runtime deps.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Diffucore is an independent implementation. Model architectures and sampling
algorithms are implemented from their original research publications.
