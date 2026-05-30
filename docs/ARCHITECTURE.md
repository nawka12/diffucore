# Diffucore Architecture

This document is the design spec for Diffucore's inference engine. It describes
*what* the engine does and *how* the pieces fit, independent of any particular
model. The first concrete target is Stable Diffusion 1.5 text-to-image; the
abstractions are chosen so that later architectures (SDXL, and DiT-style models)
slot in without reshaping the core.

## 1. Scope and non-goals

**In scope:** the path from `(checkpoint, prompt, settings) -> image`.

- Checkpoint loading and architecture detection.
- Text conditioning (tokenize -> text encoder -> embeddings).
- The sampling/denoising loop in sigma space, with classifier-free guidance.
- Latent <-> pixel conversion via the VAE.
- Device, dtype, and VRAM management.

**Out of scope (by design):** node graphs, a workflow file format, a node API,
training, a built-in web server, and a plugin/custom-node ecosystem. A separate
UI consumes the Python API described in §5.

## 2. The generation pipeline

A text-to-image request flows through five stages:

```
  prompt ─▶ [1] Conditioner ─▶ cond / uncond embeddings
                                        │
  seed  ─▶ [2] initial latent (noise)   │
                                        ▼
                       ┌─────────────────────────────┐
                       │ [3] Sampling loop            │
                       │   for sigma in schedule:     │
                       │     denoised = Denoiser(x,σ) │  ◀── wraps the diffusion
                       │     x = Sampler.step(...)    │       backbone + CFG
                       └─────────────────────────────┘
                                        │ final latent
                                        ▼
                            [4] VAE.decode ─▶ pixels ─▶ [5] image
```

Each stage is a plain object with a narrow interface; stages do not know about
each other beyond the data they pass.

## 3. Module layout

```
src/diffucore/
  __init__.py            Public API re-exports.
  sampling/              Noise schedules, parameterizations, samplers, the loop.
    schedules.py         σ schedules: karras, exponential, polyexponential, …
    parameterization.py  betas -> σ table; σ<->t; eps / v prediction scalings.
    samplers.py          Euler, Heun, ancestral — pure σ-space steppers.
    denoiser.py          wraps a backbone: applies scalings + CFG.
  models/                nn.Module backbones, implemented from papers.
    clip_text.py         CLIP ViT-L/14 text encoder (SD1.5 + SDXL encoder 1).
    open_clip_text.py    OpenCLIP ViT-bigG/14 text encoder (SDXL encoder 2).
    unet.py              config-driven eps UNet (SD1.5 and SDXL).
    vae.py               AutoencoderKL encode/decode.
  conditioning/          tokenizer(s) + text-encoder orchestration (incl. SDXL dual).
  loading/               safetensors IO, arch detection.
  runtime/               device/dtype selection (offload, tiling: planned).
  pipelines/             TextToImage — user-facing glue.
```

The full SD1.5 (512²) and SDXL (1024²) text-to-image paths are implemented.
`runtime/` offload + tiled VAE remain the main planned items (not needed for
512² SD1.5 on 12 GB; would let SDXL run on smaller cards); see `ROADMAP.md`.

## 4. Core abstractions

- **ModelBundle** — the result of loading a checkpoint: the diffusion backbone,
  the VAE, the text encoder(s), and a small `ModelSpec` (architecture id,
  prediction type, latent channels/scale, training schedule). Detection fills
  the spec from the checkpoint's tensor keys and shapes.

- **DiscreteSchedule** (`parameterization.py`) — derives the per-timestep σ table
  from the model's training betas and converts between σ and the continuous
  timestep the backbone expects. This is the bridge between "model time" and
  "sampler time."

- **Scaling** (`parameterization.py`) — the prediction parameterization. Given σ
  it yields `(c_skip, c_out, c_in)` so that
  `denoised = c_skip·x + c_out·model(c_in·x, t(σ))`. `EpsScaling` covers SD1.5;
  `VScaling` covers v-prediction models.

- **Schedule** (`schedules.py`) — a sampling-time function
  `(steps, σ_min, σ_max) -> σ[0..steps]` (descending, trailing 0).

- **Denoiser** — composes a backbone + `Scaling` + `DiscreteSchedule`
  into the single callable the loop wants: `x, σ -> denoised`. CFG is applied
  here (`CFGDenoiser`) by evaluating cond/uncond.

- **Sampler** — a pure function of σ-space: consumes `Denoiser`, an
  initial latent, and a σ schedule; returns the final latent. Knows nothing
  about text, models, or VAEs.

## 5. Public API (target)

```python
from diffucore import load_checkpoint, TextToImage

model = load_checkpoint("models/sd15.safetensors")     # -> ModelBundle
pipe  = TextToImage(model)
image = pipe(
    prompt="a watercolor fox",
    negative_prompt="blurry",
    steps=20, cfg_scale=7.0,
    width=512, height=512,
    sampler="euler", scheduler="karras",
    seed=0,
)                                                       # -> PIL.Image
```

Lower layers stay usable on their own (e.g. build a `Denoiser` and call a
`Sampler` directly) so the engine is composable, not just a single black box.

## 6. Numerical conventions

- Sampling happens in **sigma space** (Karras et al., 2022). Discrete models are
  lifted into σ via `σ(t) = sqrt((1 - ᾱ_t) / ᾱ_t)` from their training betas
  (Ho et al., 2020). This keeps one sampler implementation working across
  prediction types and schedules.
- `denoised` is always an estimate of the clean latent `x₀`; samplers operate on
  `x₀` estimates, never on raw model outputs.
- Default working dtype is **fp16** on CUDA (the 2060 has no usable bf16 path);
  σ math and the VAE run in fp32 for stability.

## 7. Device and memory strategy

Targeting a 12 GB RTX 2060 shapes the defaults:

- fp16 weights; per-submodule load-on-use with optional offload to CPU RAM so
  the text encoder, UNet, and VAE need not be resident simultaneously.
- Optional **tiled VAE** decode for large resolutions.
- A single device/dtype policy object in `runtime/` decides placement; modules
  never hardcode `.cuda()`.

## 8. Extensibility

New work plugs in at the seams, without touching the loop:

- A new **sampler** = one σ-space stepper function in `samplers.py`.
- A new **schedule** = one function in `schedules.py`.
- A new **architecture** = a backbone module in `models/` + a detector entry +
  a `ModelSpec`. The loop and samplers are unaffected.

This is deliberately minimal: no plugin registry or config DSL until a second
real architecture proves the seams.

## 9. Provenance and licensing

Diffucore is licensed Apache-2.0. Algorithms and architectures are implemented
from their original publications (DDPM, EDM/Karras schedules, DPM-Solver, the
LDM/Stable Diffusion and CLIP papers). Permissively licensed libraries (PyTorch,
safetensors, HF `tokenizers`, einops, Pillow) are used as dependencies, not
vendored.
