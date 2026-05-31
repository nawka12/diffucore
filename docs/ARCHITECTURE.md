# Diffucore Architecture

This document is the design spec for Diffucore's inference engine. It describes
*what* the engine does and *how* the pieces fit, independent of any particular
model. The first concrete target is Stable Diffusion 1.5 text-to-image; the
abstractions are chosen so that later architectures (SDXL, and DiT-style models)
slot in without reshaping the core. **Anima** (Cosmos-Predict2-family DiT) was
the first DiT integration and is now end-to-end working at 1024² — see
[`ROADMAP.md`](ROADMAP.md) for the per-component status.

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
    schedules.py         σ schedules: karras, exponential, polyexponential,
                         sgm_uniform, simple, flow-matching.
    parameterization.py  betas -> σ table; σ<->t; eps / v prediction scalings.
    samplers.py          Euler/Heun/ancestral, DPM2(+ancestral), DPM++ (2M, SDE,
                         2M-SDE, 3M-SDE), ER-SDE — pure σ-space steppers, the
                         DPM++/ER-SDE family flow-aware (half-logSNR).
    denoiser.py          wraps a backbone: applies scalings + CFG.
  models/                nn.Module backbones, implemented from papers.
    clip_text.py         CLIP ViT-L/14 text encoder (SD1.5 + SDXL encoder 1).
    open_clip_text.py    OpenCLIP ViT-bigG/14 text encoder (SDXL encoder 2).
    unet.py              config-driven eps UNet (SD1.5 and SDXL).
    vae.py               AutoencoderKL encode/decode.
  conditioning/          tokenizer(s) + text-encoder orchestration (incl. SDXL dual).
  loading/               safetensors IO, arch detection.
  runtime/               device/dtype policy + CPU offload + tiled VAE decode.
  pipelines/             TextToImage — user-facing glue.
  lora.py                apply_lora / remove_lora / clear_loras: fuse and unfuse LoRA/LoKr deltas.
```

The full SD1.5 (512²) and SDXL (1024²) text-to-image paths are implemented, as
are `runtime/` sequential CPU offload and tiled VAE decode (opt-in; they let
SDXL run on smaller cards — see §7 and `RUNTIME_SPEC.md`).

## 4. Core abstractions

- **ModelBundle** — the result of loading a checkpoint: the diffusion backbone,
  the VAE, the text encoder(s), and a small `ModelSpec` (architecture id,
  prediction type, latent channels/scale, training schedule). Detection fills
  the spec from the checkpoint's tensor keys and shapes — including the
  `prediction` type, which is `"v"` when the checkpoint carries the bare `v_pred`
  marker tensor (the eps/v weights are otherwise identical) and `"eps"` otherwise.

- **DiscreteSchedule** (`parameterization.py`) — derives the per-timestep σ table
  from the model's training betas and converts between σ and the continuous
  timestep the backbone expects. This is the bridge between "model time" and
  "sampler time."

- **Scaling** (`parameterization.py`) — the prediction parameterization. Given σ
  it yields `(c_skip, c_out, c_in)` so that
  `denoised = c_skip·x + c_out·model(c_in·x, t(σ))`. `EpsScaling` covers SD1.5;
  `VScaling` covers v-prediction models; `FlowMatchingConstScaling` covers
  rectified-flow models (Anima / Cosmos-Predict2 / Flux / SD3 CONST convention)
  where the model receives the raw noisy latent and predicts a velocity
  `v = ε − x0`. For flow, σ plays the role of the rectified-flow time
  ``t ∈ (0, 1]``; the same σ-space samplers integrate the ODE/SDE. Euler is
  exact for the rectified-flow ODE; the DPM++ / ER-SDE samplers switch to the
  flow half-logSNR mapping (`model_type="flow"`) so they apply too.

- **Schedule** (`schedules.py`) — a sampling-time function
  `(steps, σ_min, σ_max) -> σ[0..steps]` (descending, trailing 0).

- **Denoiser** — composes a backbone + `Scaling` + `DiscreteSchedule`
  into the single callable the loop wants: `x, σ -> denoised`. The pipeline picks
  the `Scaling` (`EpsScaling`/`VScaling`) from `spec.prediction`. CFG is applied
  here (`CFGDenoiser`) by evaluating cond/uncond — batched into a single backbone
  forward when the cond/uncond kwargs are equal-length tensors, else two forwards;
  inpainting wraps it with a `MaskedDenoiser` that pins the keep region to the
  original latent.

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

Anima ships as three separate files (DiT + Qwen-Image VAE + Qwen3-0.6B TE) so
the entry point takes the trio:

```python
from diffucore import load_anima_checkpoint, TextToImage

model = load_anima_checkpoint(
    dit_path="models/anima-base-v1.0.safetensors",
    vae_path="models/qwen_image_vae.safetensors",
    te_path="models/qwen_3_06b_base.safetensors",
    device="cuda", dtype=torch.float16,
)
image = TextToImage(model)("a watercolor fox", steps=20, cfg_scale=4.0,
                           width=1024, height=1024, shift=3.0, seed=0)
```

`TextToImage` dispatches by `model.spec.architecture`; the SD/SDXL and Anima
paths share the bundle/conditioning/decode contract but diverge in the
sampling-loop internals (different parameterization, schedule, and the DiT's
extra `t5xxl_ids` kwarg routed through its built-in LLM-Adapter).

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

By default placement is "everything resident": `load_checkpoint` eagerly moves
all modules to the device. Opting into `DevicePolicy(offload=True)` instead loads
the modules on CPU and the pipeline shuttles each onto the GPU around its stage
(`on_device`), and `tiled_vae_decode` bounds decode memory by tiling (auto ≥768 px).
On the RTX 2060 this takes 1024² SDXL from a 9.97 GB resident/untiled peak down to
6.6 GB, byte-identically. The measured breakdown, design, and verification are in
the build sheet [`RUNTIME_SPEC.md`](RUNTIME_SPEC.md).

## 8. Extensibility

New work plugs in at the seams, without touching the loop:

- A new **sampler** = one σ-space stepper function in `samplers.py`.
- A new **schedule** = one function in `schedules.py` (e.g.
  `flow_matching_schedule` for SD3-style shifted rectified-flow models).
- A new **parameterization** = one `Scaling` subclass in `parameterization.py`
  (`EpsScaling`, `VScaling`, `FlowMatchingConstScaling`).
- A new **architecture** = a backbone module in `models/` + a detector entry +
  a `ModelSpec`. The loop and samplers are unaffected.
- **Adapters (LoRA/LoKr)** plug in *after* loading, not at a loop seam:
  `lora.py`'s `apply_lora` fuses the adapter's weight delta directly into the
  bundle's module weights (`W += multiplier·ΔW`). Because the fusion is in the
  weights, the sampling loop, the offload machinery, and every backbone are
  oblivious to it. LoRAs stack across calls; the first time a weight is touched a
  pristine CPU snapshot is taken so `remove_lora` / `clear_loras` can undo a fuse
  (restore the snapshot, replay the rest of the stack) without reloading the
  checkpoint. Supporting a new adapter family means adding a delta
  reconstruction (`_compose_*`) and/or a key→module mapping, nothing more.

These seams have now been exercised by Anima — a Cosmos-Predict2-family DiT
with a different VAE family (Wan 3D-causal-conv), a different text encoder
(Qwen3 decoder LM), an internal cross-encoder LLM-Adapter, and a different
prediction parameterization (CONST flow) — all integrated by adding modules
and one detector branch, without touching the σ-space samplers, the sigma
schedules (other than adding `flow_matching_schedule`), or the
denoising-loop scaffolding.

The pipeline layer was the one place where Anima needed a dispatch rather
than a drop-in: enough of SD's `_Pipeline` (Karras schedule, `EpsScaling`,
fixed 4-channel latents, scalar `latent_scale`) doesn't apply to flow-matching
DiTs that the Anima path lives in its own self-contained driver
(`pipelines/_anima.py`) that `TextToImage` dispatches into. This is by
design — see §9 in [`ROADMAP.md`](ROADMAP.md) for the DT0–DT7 build sheet.

## 9. Provenance and licensing

Diffucore is licensed Apache-2.0. Algorithms and architectures are implemented
from their original publications (DDPM, EDM/Karras schedules, DPM-Solver and
DPM-Solver++, ER-SDE-Solver, the LDM/Stable Diffusion and CLIP papers). Permissively licensed libraries (PyTorch,
safetensors, HF `tokenizers`, einops, Pillow) are used as dependencies, not
vendored.
