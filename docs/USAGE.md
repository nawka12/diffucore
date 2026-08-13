# Diffucore — Usage & examples

The full feature list, code examples, the complete sampler/scheduler set,
supported models, and performance numbers. For install and status, see the
[README](../README.md).

## Highlights

- **One API across architectures** — Stable Diffusion 1.5, SDXL, **Anima**
  (a 2 B DiT built on Cosmos-Predict2 with Qwen3-0.6B + Qwen-Image VAE), and the
  **FLUX** family (FLUX.1 dev/schnell, FLUX.2 Klein/Dev). Load any of them, drive
  them all the same way. (FLUX.1 schnell and FLUX.2 Klein-4B run on real weights;
  the dev / Dev variants are build-to-spec — see [Status](../README.md#status).)
- **Text-to-image, image-to-image, and inpainting** — all four architectures do
  all three. Anima and FLUX use soft, latent-mask inpainting (no dedicated
  inpaint model), so masked edits follow the base model's denoise.
- **Long prompt weighting (LPW) on SDXL** — A1111-style attention syntax
  (`(word:1.3)`, `(word)`, `[word]`) and prompts beyond CLIP's 77-token limit.
- **Checkpoint types auto-detected** — epsilon and v-prediction, with
  zero-terminal-SNR (ZTSNR) + CFG-rescale handled for you.
- **LoRA & LoKr** adapters fuse into the weights at load time (kohya/A1111,
  PEFT, and Anima naming conventions).
- **11 samplers, multiple schedulers** — Euler/Heun, the DPM++ family, ER-SDE,
  SECANT, and more (full list under [Usage](#choosing-samplers--schedulers)). The
  DPM++, ER-SDE, and SECANT samplers are flow-aware, so they drive Anima too.
- **Runs on modest GPUs** — sequential CPU offload + tiled VAE fit SDXL into
  ~6.6 GB; block streaming (`offload="stream"`) drops it further so SDXL or Anima
  fit a ~4 GB card, and FLUX.1's ~23 GB transformer fits a 24 GB card.
- **Seed-reproducible** — same seed, same image, every run.
- **From scratch** — every backbone is implemented from the original papers and
  verified against reference implementations (text encoders and the SD1.5 UNet
  match bit-for-bit in fp32). No `diffusers`/ComfyUI at runtime.

## Usage

### Choosing samplers & schedulers

```python
image = TextToImage(model)(
    prompt, negative_prompt,
    steps=30, cfg_scale=5.0,
    sampler="dpmpp_2m", scheduler="sgm_uniform", seed=0,
)
```

- **Samplers** — `euler`, `euler_ancestral`, `heun`, `heunpp2`, `dpm_2`,
  `dpm_2_ancestral`, `dpmpp_2s_ancestral`, `dpmpp_2m`, `dpmpp_2m_sde`,
  `dpmpp_2m_sde_heun`, `dpmpp_sde`, `dpmpp_3m_sde`, `ipndm`, `ipndm_v`,
  `res_multistep`, `res_multistep_ancestral`, `gradient_estimation`, `lms`,
  `er_sde`, `lcm`, `secant`, plus `ddpm` (SD/SDXL only — a VP/VE sampler) and
  `secant_anneal` (Anima only — σ-annealed ancestral burn-in at high σ handing
  off to `secant`'s 2nd-order x0 refinement as σ→0; spans `euler_ancestral_anneal`
  at `curvature=0` and deterministic `secant` at `eta_max=0`), and
  `dpmpp_2m_anneal` (Anima only — the "good and fast" sibling: `euler_ancestral_anneal`'s
  same σ-annealed burn-in (`eta = eta_max·σ`) but with the DPM++(2M) flow multistep
  as the deterministic core instead of plain Euler / the secant. The 2M core stays
  genuinely 2nd-order at low step counts — where the secant self-gates back to Euler —
  so it reaches the same quality in fewer steps; `eta_max=0` is the deterministic
  2M flow solver. Pair with `beta`/`flow` like its siblings), and
  `cogent` (all families — the same σ-annealed burn-in and DPM++(2M) exponential
  core as `dpmpp_2m_anneal`, but the 2nd-order correction is scaled by a *measured*
  weight `psi = max((1 + 2·rho)/3, 1 − e^−h)` instead of a hardcoded σ heuristic:
  `rho` is the coherence of consecutive x0 differences, giving a Wiener shrinkage
  that damps itself on an imperfect model and stays undamped on a clean one, and
  `1 − e^−h` is the integrator's own phi-weight as a step-size floor. `eta_max=0`
  is deterministic; `psi ≡ 1` is exactly `dpmpp_2m_anneal`. Prefer 24+ steps),
  and `cogent3` (all families — cogent's measured 2nd-order gate plus a second
  gate `psi_2 = (2 + 3·rho_2)/5` on the 3rd-order DPM-Solver++(3M) term, where
  `rho_2` is the coherence of consecutive *second* differences of the x0
  history; with both gates pinned to 1 and `eta_max=0` it is bit-for-bit the
  deterministic 3M core, and under a rough/merged model the 3rd-order term
  damps itself back to cogent's 2nd-order behaviour instead of amplifying the
  error. `psi_2` has no floor — the term is never load-bearing. Prefer 24+
  steps).
  Ancestral samplers are rectified-flow-aware on Anima/FLUX.
- **Schedulers** — `karras`, `exponential`, `polyexponential`, `kl_optimal`,
  `sgm_uniform`, `simple`, `normal`, `ddim_uniform`, `linear_quadratic`
  (SD/SDXL); `flow` (default), `flow_dyn`, `oss`, `sgm_uniform`, `simple`,
  `normal`, `kl_optimal`, `linear_quadratic`, `smoothstep`, `beta`,
  `beta_mix` (Anima); `flux` (default), `flow`,
  `sgm_uniform`, `simple`, `normal`, `kl_optimal`, `linear_quadratic` (FLUX).
  `ddim_uniform` is SD/SDXL-only (it starts below σ_max, which the flow
  pipelines' σ_max = 1 init assumes).
  `flow_dyn` is `flow` with a Flux-style resolution-aware shift (auto-derived
  from the image's token count; the `shift` value is ignored — and the UI has
  no shift control, so plain `flow` always runs shift=3.0). `oss` is a
  pre-calibrated optimal-stepsize schedule — calibrate it from the UI's OSS
  panel (or headless via `calibrate_oss.py`), once per steps/resolution/shift.
  `smoothstep` (Anima) is U-shaped — dense near σ = 1 *and* σ = 0 — designed
  to pair with `euler_ancestral_anneal` on rectified-flow merges.
  `beta` (Anima) places timesteps at Beta(0.6, 0.6) quantiles before the flow
  shift map — a tunable U-shape (the ComfyUI `beta` schedule, pure-torch) that
  the community favors with DPM++-family samplers.
  `beta_mix` (Anima) generalizes `beta` to a two-Beta mixture
  `w·Beta(α₁,β₁) + (1−w)·Beta(α₂,β₂)` so the two endpoint peaks can differ in
  shape — the symmetric `beta` forces them to match. Defaults `(w, α₁, β₁, α₂,
  β₂) = (0.5, 0.8, 2.0, 3.0, 0.7)` follow Lee et al. (2024, arXiv:2407.12173
  Fig. 2d)'s finding that LDMs want an *asymmetric* curve — more steps at the
  high-freq detail (low-σ) end than at the high-noise end — but are tuned for
  Anima's flow shift map rather than transcribed literally: the paper's raw
  `Beta(0.5,2.0)+Beta(3.0,0.5)` come out near-*symmetric* once mapped through
  σ(t), over-pack the pure-noise σ≈1 end, and collide steps at the σ floor for
  step counts ≳40. Pair with a 2nd-order solver (`dpmpp_2m`, `heunpp2`) for
  best effect.

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

### FLUX (DiT)

FLUX is a guidance-distilled rectified-flow MMDiT, so there's no negative-prompt
CFG pass — `cfg_scale` is the *distilled guidance* scale. Load an all-in-one
checkpoint (`load_checkpoint` auto-detects it) or the official split files
(`load_flux_checkpoint`):

```python
from diffucore import load_flux_checkpoint, TextToImage

# FLUX.1: transformer + VAE + T5-XXL + CLIP-L
model = load_flux_checkpoint(
    transformer_path="flux/flux1-dev.safetensors",
    vae_path="flux/ae.safetensors",
    t5_path="flux/t5xxl.safetensors",
    clip_path="flux/clip_l.safetensors",
    device="cuda", dtype=torch.bfloat16,
)
image = TextToImage(model)(
    "a watercolor fox", steps=20, cfg_scale=3.5, width=1024, height=1024, seed=0,
)

# FLUX.2 Klein: transformer + VAE + a single Qwen3 (4B/8B) text encoder
model = load_flux_checkpoint(
    transformer_path="flux2/flux2-klein.safetensors",
    vae_path="flux2/ae.safetensors",
    mistral_path="flux2/qwen3_4b.safetensors",   # Qwen3 or Mistral-3; auto-detected
    device="cuda", dtype=torch.bfloat16,
)
```

For FLUX.2, use the **ComfyUI-format single files** for the VAE and text encoder
(e.g. `flux2-vae.safetensors`, `qwen_3_4b.safetensors`). The official BFL repo
ships those two in diffusers layout, which this loader's single-file path doesn't
consume; the transformer single-file loads as-is.

**Fitting a big backbone on a small GPU (`offload="stream"`).** When the backbone
can't share the GPU with its own activations, stream its blocks: the small
modules stay resident and each block is shuttled onto the GPU just for its own
forward, the ComfyUI `--lowvram` analog. Works for every backbone —
- **FLUX.1** (~23 GB transformer): peak VRAM drops to ~22 GB, so a 1024² schnell
  image fits a 24 GB card (whole-module staging OOMs).
- **SD/SDXL** (UNet, `("input_blocks", "middle_block", "output_blocks")`) and
  **Anima** (DiT, `("blocks",)`): fit a ~4 GB card where `offload="full"`
  (whole-UNet staging) OOMs once 1024² activations land on top.

It's the slowest mode (weights cross PCIe each step), so reserve it for cards
that can't fit the backbone otherwise. The example below is FLUX.1; SD/SDXL and
Anima take the same `offload="stream"` policy through `load_checkpoint` /
`load_anima_checkpoint`.

```python
from diffucore import load_flux_checkpoint, TextToImage
from diffucore.runtime import DevicePolicy
import torch

policy = DevicePolicy(device=torch.device("cuda"), offload="stream", compute_dtype=torch.bfloat16)
model = load_flux_checkpoint(
    transformer_path="flux/flux1-schnell.safetensors", vae_path="flux/ae.safetensors",
    t5_path="flux/t5xxl_fp16.safetensors", clip_path="flux/clip_l.safetensors",
    dtype=torch.bfloat16, policy=policy,
)
image = TextToImage(model)("a watercolor fox", steps=4, width=1024, height=1024, seed=0)
```

### Image-to-image & inpainting

```python
from PIL import Image
from diffucore import ImageToImage, Inpaint

edit = ImageToImage(model)(prompt, Image.open("input.png"), strength=0.6, seed=0)
fill = Inpaint(model)(prompt, Image.open("input.png"), Image.open("mask.png"), seed=0)
```

### Progress and runtime info

Pipelines return a `PIL.Image` by default. UIs can opt into step callbacks and
structured runtime info without changing the default API:

```python
from diffucore import PipelineInfo, TextToImage

steps_seen = []
image, info = TextToImage(model)(
    prompt,
    steps=25,
    progress_callback=lambda step, total: steps_seen.append((step, total)),
    return_info=True,
)
assert isinstance(info, PipelineInfo)
print(info.vae_decode_mode)  # "tiled" or "untiled"
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
3-17 % free win); the rest default off:

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

Constraints: `compile=True` is incompatible with backbone-moving offload
(`offload=True` / `"full"`) — pair it with `offload=False` or `"encoders"`.
`cuda_graphs=True` requires `compile=True` and stable input shapes (each new
resolution / LPW chunk count re-records once). **On Anima above 1024² on 12 GB
cards, skip `cuda_graphs`** — CFG runs cond/uncond as two different-shape
forwards, so each step captures two graph pools and DiT attention scales
O(tokens²), blowing the budget; use `compile=True` alone. `channels_last` /
`compile` help most on Ampere+; on Turing the cuDNN NCHW path is already
near-optimal. See [Performance](#performance).

**TeaCache (Anima, opt-in).** Pass `teacache_thresh > 0` to any Anima pipeline
(`TextToImage` / `ImageToImage` / `Inpaint`) to skip the 28-block transformer on
steps whose timestep-modulated input drifts little from the last computed step,
reusing the cached block residual (Liu et al., 2024). The threshold directly
bounds the accumulated per-step change between recomputes — higher = more skipping
= faster but lower fidelity; `0` (default) is off and bit-exact to a plain run.
The safe value depends on sampler and step count: high step counts with
single-step / secant-family samplers stay near-lossless to ~0.3–0.5, while
few-step multistep samplers (`dpmpp_2m`) need ≤0.01. Tune by eye — start low.

## Supported models

| Family | Native res | Modes | Notes |
|---|---|---|---|
| Stable Diffusion 1.5 | 512² | t2i · img2img · inpaint | eps + v-pred, ZTSNR |
| SDXL | 1024² | t2i · img2img · inpaint | dual text encoders, eps + v-pred, ZTSNR |
| Anima (Cosmos-Predict2 2 B DiT) | 1024² | t2i · img2img · inpaint ‡ | flow-matching, Qwen3 + Qwen-Image VAE |
| FLUX.1 (dev / schnell) | 1024² | t2i · img2img · inpaint ‡ | flow-matching MMDiT, T5-XXL + CLIP-L † |
| FLUX.2 (Klein / Dev) | 1024² | t2i · img2img · inpaint ‡ | global-mod MMDiT, Qwen3 (Klein) / Mistral-3 (Dev) † |

‡ Soft, latent-mask inpaint (no dedicated inpaint model).

LoRA / LoKr adapters are supported on SD1.5 / SDXL / Anima.

† **FLUX.1 (schnell) and FLUX.2 (Klein-4B) load and run on the official real
weights** — coherent, prompt-faithful, deterministic. Bit-exact parity against
the reference is still pending, and the FLUX.1-dev / FLUX.2-Dev (Mistral-3) paths
are implemented to spec but not yet hardware-verified. See [Status](../README.md#status).

## Performance

Measured on an RTX 2060 (12 GB, Turing sm_75), fp16, 20 steps.

**Default behavior** (`cudnn_benchmark=True`, the rest off):

| Model | Resolution | Time | Peak VRAM |
|---|---|---|---|
| SD 1.5 | 512² | ~2.83 s | ~3.2 GB |
| SDXL | 1024² | ~16.89 s | ~10.7 GB (≈6.6 GB with offload) |
| Anima | 1024² | ~46.33 s | ~8.6 GB |

**With additional opt-in flags** stacked on top (same card, same prompt/seed;
speedup is vs all flags off):

| Model | Best time | Speedup | Winning flags (over the default) |
|---|---|---|---|
| SD 1.5 | ~2.78 s | 1.08× | `+ channels_last` |
| SDXL | ~16.89 s | 1.17× | (none — default is the winner) |
| Anima | ~35.55 s | **1.47×** | `+ compile + cuda_graphs` |

Findings:

- `cudnn_benchmark=True` is a free, bit-exact win on every model (3-17 %) and is
  the default. Set `cudnn_benchmark=False` to restore the old path (e.g. for
  diffusers byte-equivalence comparisons).
- `compile=True` is the headline win on Anima (~33-41 % on Turing, more on
  Ampere+), for a one-time ~30-180 s warmup paid at load.
- `cuda_graphs=True` (requires `compile=True`) adds a small extra speedup and
  *more deterministic* output, but skip it on Anima above 1024² on 12 GB cards
  (two shape-specific graph pools blow the VRAM budget).
- `channels_last` / `compile` are hardware-sensitive (help more on Ampere+, can
  regress on Turing SDXL); `tf32` only touches the fp32 VAE — one call per image.
