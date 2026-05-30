# Diffucore

**A clean, from-scratch diffusion inference engine in PyTorch.**

Point it at a checkpoint, give it a prompt, get an image. Diffucore owns the
entire generation path — checkpoint loading, text conditioning, the
sampling/denoising loop, and VAE decoding — in one small, readable library you
can `import` or embed behind your own UI.

![status](https://img.shields.io/badge/status-alpha-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

It's a focused library, not a node-graph app: no workflow JSON, no node system,
and no runtime dependency on `diffusers` or ComfyUI.

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

## Highlights

- **Three model families, one API** — Stable Diffusion 1.5, SDXL, and **Anima**
  (a 2 B DiT built on Cosmos-Predict2 with Qwen3-0.6B + Qwen-Image VAE). Load any
  of them, drive them all the same way.
- **Text-to-image, image-to-image, and inpainting** out of the box.
- **Checkpoint types auto-detected** — epsilon and v-prediction, with
  zero-terminal-SNR (ZTSNR) + CFG-rescale handled for you.
- **LoRA & LoKr** adapters fuse into the weights at load time (kohya/A1111,
  PEFT, and Anima naming conventions).
- **10 samplers, multiple schedulers** — Euler, Heun, ancestral, DPM2, the full
  DPM++ family, and ER-SDE; Karras / exponential / sgm_uniform / simple / flow
  schedules. The DPM++ and ER-SDE samplers are flow-aware, so they drive Anima too.
- **Runs on modest GPUs** — optional sequential CPU offload + tiled VAE decode
  fit SDXL into ~6.6 GB.
- **Seed-reproducible** — same seed, same image, every run.
- **From scratch** — every backbone is implemented from the original papers and
  verified against reference implementations (the text encoders and SD1.5 UNet
  match bit-for-bit in fp32). No `diffusers`/ComfyUI at runtime.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
# then install the CUDA build of torch for your GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Runtime dependencies are small: `torch`, `numpy`, `safetensors`, `tokenizers`,
`Pillow`, `einops`, `tqdm`.

## Usage

### Choosing samplers & schedulers

```python
image = TextToImage(model)(
    prompt, negative_prompt,
    steps=30, cfg_scale=5.0,
    sampler="dpmpp_2m", scheduler="sgm_uniform", seed=0,
)
```

- **Samplers** — `euler`, `heun`, `euler_ancestral`, `dpm_2`, `dpm_2_ancestral`,
  `dpmpp_2m`, `dpmpp_sde`, `dpmpp_2m_sde`, `dpmpp_3m_sde`, `er_sde`.
- **Schedulers** — `karras`, `exponential`, `polyexponential`, `sgm_uniform`,
  `simple` (SD/SDXL); `flow` (default), `sgm_uniform`, `simple` (Anima).

### Anima (DiT)

```python
from diffucore import load_anima_checkpoint, TextToImage

model = load_anima_checkpoint(
    "anima/dit.safetensors", "anima/qwen_image_vae.safetensors", "anima/qwen3_te.safetensors",
    device="cuda", dtype=torch.float16,
)
image = TextToImage(model)(
    prompt, negative_prompt,
    steps=32, cfg_scale=4.5, width=832, height=1216,
    sampler="er_sde", shift=3.0, seed=0,
)
```

### Image-to-image & inpainting

```python
from PIL import Image
from diffucore import ImageToImage, Inpaint

edit = ImageToImage(model)(prompt, Image.open("input.png"), strength=0.6, seed=0)
fill = Inpaint(model)(prompt, Image.open("input.png"), Image.open("mask.png"), seed=0)
```

### LoRA / LoKr

```python
from diffucore import apply_lora

report = apply_lora(model, "loras/style.safetensors", multiplier=0.8)
print(report)  # matched / unmatched module counts
```

### Fitting bigger models on smaller GPUs

```python
import torch
from diffucore import load_checkpoint
from diffucore.runtime import DevicePolicy

policy = DevicePolicy(device=torch.device("cuda"), offload=True, vae_tile=True)
model = load_checkpoint("models/sdxl.safetensors", policy=policy)
```

## Supported models

| Family | Native res | Modes | Notes |
|---|---|---|---|
| Stable Diffusion 1.5 | 512² | t2i · img2img · inpaint | eps + v-pred, ZTSNR |
| SDXL | 1024² | t2i · img2img · inpaint | dual text encoders, eps + v-pred, ZTSNR |
| Anima (Cosmos-Predict2 2 B DiT) | 1024² | t2i | flow-matching, Qwen3 + Qwen-Image VAE |

LoRA / LoKr adapters are supported on all three.

## Performance

Measured on an RTX 2060 (12 GB), fp16, 20 steps:

| Model | Resolution | Time | Peak VRAM |
|---|---|---|---|
| SD 1.5 | 512² | ~3.6 s | ~3.2 GB |
| SDXL | 1024² | ~19 s | ~10.7 GB (≈6.6 GB with offload) |
| Anima | 1024² | ~46 s | ~8.6 GB |

## Status

Diffucore is in **alpha**. The engine is end-to-end working and seed-reproducible
across all three model families, with the sampling core and checkpoint detection
unit-tested and every backbone verified against reference implementations. APIs
may still shift before 1.0. CPU works for testing; real generation targets CUDA.

**For ComfyUI users:** samplers and schedulers follow ComfyUI's k-diffusion
conventions, but the SDE samplers re-inject seeded Gaussian noise instead of
Brownian-tree noise — results are coherent and reproducible, but not bit-identical
to a ComfyUI render.

## Documentation

- [Architecture & rationale](docs/ARCHITECTURE.md)
- [Roadmap & verified milestones](docs/ROADMAP.md)
- [Runtime / VRAM (offload + tiled VAE)](docs/RUNTIME_SPEC.md)
- [Implementation notes](docs/IMPLEMENTATION_SPEC.md) · [build log](docs/HANDOFF.md)

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Diffucore is an
independent implementation; model architectures and sampling algorithms are
implemented from their original research publications.
