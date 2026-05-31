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

## Highlights

- **Three model families, one API** — Stable Diffusion 1.5, SDXL, and **Anima**
  (a 2 B DiT built on Cosmos-Predict2 with Qwen3-0.6B + Qwen-Image VAE). Load any
  of them, drive them all the same way.
- **Text-to-image, image-to-image, and inpainting** out of the box.
- **Long prompt weighting (LPW) on SDXL** — A1111-style attention syntax
  (`(word:1.3)`, `(word)`, `[word]`) and prompts beyond CLIP's 77-token limit.
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
from diffucore import apply_lora, remove_lora, clear_loras

report = apply_lora(model, "loras/style.safetensors", multiplier=0.8)
print(report)  # matched / unmatched module counts

# LoRAs stack; remove one (or all) without reloading the checkpoint
apply_lora(model, "loras/character.safetensors", multiplier=0.6)
remove_lora(model, "loras/style.safetensors")  # re-fuses the rest, exact
clear_loras(model)                             # back to base weights
```

### Fitting bigger models on smaller GPUs

```python
import torch
from diffucore import load_checkpoint
from diffucore.runtime import DevicePolicy

policy = DevicePolicy(device=torch.device("cuda"), offload=True, vae_tile=True)
model = load_checkpoint("models/sdxl.safetensors", policy=policy)
```

### Going faster

Five perf flags on `DevicePolicy`. `cudnn_benchmark` defaults **on** (bit-exact,
3-17 % free win); the rest default off and opt in only when you accept their
tradeoffs:

```python
policy = DevicePolicy(
    device=torch.device("cuda"),
    # cudnn_benchmark=True,  # default on; set False to disable
    tf32=True,              # TF32 for fp32 matmul/cuDNN on Ampere+
    channels_last=True,     # NHWC for the SD/SDXL UNet + AutoencoderKL
    compile=True,           # torch.compile the backbone (~30-60s warmup)
    cuda_graphs=True,       # capture per-step graph via reduce-overhead mode
)
```

`compile=True` is incompatible with offload modes that move the backbone
(`offload=True` / `"full"`) — pair it with `offload=False` or `"encoders"`.
`cuda_graphs=True` requires `compile=True` and assumes stable input shapes
(same resolution + LPW chunk count); each new shape triggers one re-record.
**On Anima above 1024² (e.g. 1024×1536), skip `cuda_graphs` on 12 GB cards** —
Anima's CFG runs cond/uncond as two different-shape forwards, so each step
captures two graph pools, and DiT self-attention activations scale as
O(tokens²). Above 1024² the two resident pools blow the VRAM budget. Use
`compile=True` alone instead (eager activations are transient and still fit).
`channels_last` and `compile` are most effective on Ampere+ (RTX 30/40-series);
on Turing (RTX 20-series) the cuDNN NCHW path is already near-optimal and
`channels_last` may regress. See [Performance](#performance).

## Supported models

| Family | Native res | Modes | Notes |
|---|---|---|---|
| Stable Diffusion 1.5 | 512² | t2i · img2img · inpaint | eps + v-pred, ZTSNR |
| SDXL | 1024² | t2i · img2img · inpaint | dual text encoders, eps + v-pred, ZTSNR |
| Anima (Cosmos-Predict2 2 B DiT) | 1024² | t2i | flow-matching, Qwen3 + Qwen-Image VAE |

LoRA / LoKr adapters are supported on all three.

## Performance

Measured on an RTX 2060 (12 GB, Turing sm_75), fp16, 20 steps.

**Default behavior** (`cudnn_benchmark=True`, the rest off):

| Model | Resolution | Time | Peak VRAM |
|---|---|---|---|
| SD 1.5 | 512² | ~2.83 s | ~3.2 GB |
| SDXL | 1024² | ~16.89 s | ~10.7 GB (≈6.6 GB with offload) |
| Anima | 1024² | ~46.33 s | ~8.6 GB |

**Pre-PR-A baseline** (all flags off, for comparison only):

| Model | Time | Default speedup |
|---|---|---|
| SD 1.5 | ~3.01 s | 1.06× |
| SDXL | ~19.84 s | 1.17× |
| Anima | ~52.20 s | 1.13× |

**With additional opt-in flags** stacked on top (same card, same prompt/seed):

| Model | Best time | Speedup vs pre-PR-A | Winning flags (over the default) |
|---|---|---|---|
| SD 1.5 | ~2.78 s | 1.08× | `+ channels_last` |
| SDXL | ~16.89 s | 1.17× | (none — default is the winner) |
| Anima | ~35.55 s | **1.47×** | `+ compile + cuda_graphs` |

Findings:

- `cudnn_benchmark=True` is a free, bit-exact win on every model (3-17 %).
  **It's the default** as of PR-A; the pre-PR-A numbers above are baselines you
  no longer have to opt in to. Set `cudnn_benchmark=False` to restore the old
  path (e.g. for diffusers byte-equivalence comparisons).
- `compile=True` is the headline win on Anima (~33-41 % on Turing; expect more
  on Ampere+). Its one-time warmup is ~30-180 s depending on model size, paid
  at load.
- `cuda_graphs=True` (requires `compile=True`) adds a small extra speedup on
  top of compile (~5 % on Anima Turing) and produces *more deterministic*
  output (PSNR 54.7 dB vs 29.0 dB for compile-alone), because
  `mode="reduce-overhead", dynamic=False` selects kernels consistently.
- `channels_last=True` and `compile=True` are hardware-sensitive: on Turing
  they help SD1.5 marginally and slightly regress on SDXL. On Ampere+ both
  typically help more.
- `tf32=True` is a no-op for diffucore's fp16 backbones; only the fp32 VAE
  benefits, and that's one call per image.

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
