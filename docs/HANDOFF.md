# Handoff — continuing diffucore on the RTX 2060 PC

This is the pickup point for moving from the laptop (no CUDA) to the RTX 2060
12 GB PC, where the model components can finally be run and verified.

> **Update: M4–M7 are complete and verified on the RTX 2060.** The end-to-end
> SD1.5 text-to-image path works (`load_checkpoint` → `TextToImage`). The
> "What's left" and setup sections below are kept as the historical pickup
> record; current status lives in [`ROADMAP.md`](ROADMAP.md), and the
> implementation learnings are folded into "Gotchas" at the bottom.

## What's done and verified (laptop, CPU)

| Milestone | What | Verified |
|---|---|---|
| M0 | Scaffold: Apache-2.0, pyproject, package layout, docs | imports + `pytest` |
| M1 | Sampling foundation: σ schedules, σ⇄t, eps/v scalings | 15 tests |
| M2 | Samplers (Euler/Heun/ancestral) + denoising loop + CFG | 10 tests |
| M3 | Checkpoint loading + SD1.5 detection (from safetensors header) | 7 tests |

**32 tests pass.** The entire sampling/denoising path is real and tested against
analytic solutions — you should not need to touch it.

## What was left — now done (this PC)

Built in the recommended order (**VAE → CLIP → UNet → pipeline**), each mirroring
the on-disk key names so a `strict=True` load is the architecture check, then
verified numerically per [`IMPLEMENTATION_SPEC.md`](IMPLEMENTATION_SPEC.md).

| Milestone | File | Verified |
|---|---|---|
| M4 | `models/clip_text.py`, `conditioning/__init__.py` | strict load; bit-identical to `transformers.CLIPTextModel` (max\|Δ\|=0) |
| M5 | `models/vae.py` | strict load; `decode(encode(img))` PSNR 35.1 dB; O(1) latent stats |
| M6 | `models/unet.py`, `bundle.py` | strict load; bit-identical to diffusers `UNet2DConditionModel`; fp16 fwd ~1.8 GB |
| M7 | `pipelines/text_to_image.py` | coherent 512² image; fixed seed → bit-identical; ~3.6 s/20 steps, ~3.2 GB |

## Environment setup on the PC

```bash
git clone <repo> diffucore && cd diffucore       # or copy the folder over
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                           # installs numpy/safetensors/tokenizers/Pillow/einops/pytest
# Install CUDA torch for the 2060 (Turing, sm_75) — pick the index for your CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest                                             # the 32 CPU tests must stay green
```

> The laptop `.venv` is CPU-only torch and should **not** be copied to the PC —
> recreate it so you get the CUDA build. Everything else is portable.

Get the reference checkpoint (SD1.5): `v1-5-pruned-emaonly.safetensors`, place it
under `models/` (gitignored). First, dump its real keys to align names:
```bash
python -c "from diffucore.loading import read_header; ks=sorted(read_header('models/v1-5-pruned-emaonly.safetensors')); print(len(ks)); print('\n'.join(ks[:30]))"
```

## How to verify (the success criteria)

Use HF `diffusers`/`transformers` as a **numerical oracle** (compare outputs for
identical inputs; don't copy their code). Full per-module criteria are in the
spec's "Verification plan". The end goal (M7): a coherent 512² image from the
reference checkpoint, reproducible for a fixed seed.

## Gotchas (learned while building)

- **VAE in fp32.** fp16 VAE decode produces NaNs/artifacts on many SD1.5 weights;
  run UNet/CLIP in fp16 but keep the VAE (and all σ math) fp32. See
  `runtime.DevicePolicy`.
- **Latent scale 0.18215** is applied at the VAE boundary, not inside the nets.
- **`strict=True` loads are your friend** — mirror the on-disk parameter names so
  a clean load proves the architecture matches (spec §"mirror the on-disk names").
- **Timesteps are continuous.** The sampler passes `t = sigma_to_t(sigma)` (a
  float), not an int; the UNet's sinusoidal embedding handles floats fine.
- **Keep the context kwarg name consistent** across `Conditioner`, `CFGDenoiser`
  cond/uncond dicts, and `UNetModel.forward` (the spec uses `context=`).
- **Python 3.14 has no torch wheels** — use 3.11 (or whatever the current torch
  supports), which is why the venv is pinned.

Learned while implementing M4–M7:

- **GroupNorm in fp32 needs fp32 weights too.** The UNet's `GroupNorm32` casts
  the *input* to fp32; with fp16 weights that mismatches. Cast the weight/bias to
  fp32 in the norm forward (`F.group_norm(x.float(), …, weight.float(), …)`).
- **VAE and UNet downsample differently.** The VAE `Downsample` uses asymmetric
  `F.pad(x, (0,1,0,1))` then stride-2 conv with padding 0; the UNet `Downsample`
  uses a stride-2 conv with symmetric `padding=1`. Both match their checkpoints.
- **Tokenizer is vendored.** `conditioning/clip_tokenizer.json` is the OpenAI CLIP
  BPE (MIT), loaded via the `tokenizers` dep. Its post-processor already adds
  BOS/EOS; `CLIPTokenizer.encode` only truncates and pads to 77 with EOS.
- **Keep the schedule on-device.** `bundle.load_checkpoint` moves the
  `DiscreteSchedule` sigma table (fp32) onto the run device so `sigma_to_t`
  returns timesteps on the same device as the latents (avoids a CPU/GPU mismatch
  in the UNet's time embedding).
- **Verification oracles are dev-only.** HF `transformers`/`diffusers` are used to
  numerically check CLIP/UNet/VAE; they are **not** runtime deps. CLIP and the
  UNet match bit-for-bit; diffusers `UNet2DConditionModel.from_single_file` loads
  the same checkpoint for the UNet oracle.

Learned while implementing SDXL:

- **`transformers<5` for the SDXL oracle.** diffusers 0.38's single-file loader
  (`from_single_file` / `StableDiffusionXLPipeline`) reaches for
  `CLIPTextModel.text_model`, which transformers 5.x removed. Pin `transformers>=4.44,<5`
  in the dev env for the oracle scripts. (The engine itself doesn't use either.)
- **The two SDXL tokenizers pad differently.** CLIP-L pads to 77 with EOS (49407);
  OpenCLIP bigG pads with **0**. LPW's `_chunk(..., pad_token=...)` takes the fill
  and builds the L/G token windows separately. Pooled is unaffected (argmax finds
  the real EOS) but the penultimate hidden differs at pad positions if you get
  this wrong.
- **SDXL uses clip_skip=2** (penultimate hidden, no final LN) for *both* encoders;
  the 2048-d context is `cat([clip_l_hidden(768), big_g_hidden(1280)], dim=-1)`.
- **`SDXLConditioner` does LPW (long prompt weighting).** It parses A1111 attention
  syntax (`(word:1.3)`, `(word)`=1.1, `[word]`=1/1.1, `\(` escapes), splits the
  prompt into 75-token chunks (each a BOS…EOS 77-token window), encodes each chunk
  through both encoders, applies A1111 mean-preserving per-token weights to the
  2048-d context, and concatenates the chunk contexts along the sequence axis (so
  prompts can exceed 77 tokens). Pooled comes from the final chunk. A short,
  unweighted prompt collapses to one chunk and reproduces the plain encoding exactly.
- **SpatialTransformer proj differs by arch.** SD1.5 uses a 1×1 **conv**
  `proj_in/out` (applied before flattening); SDXL uses a **Linear** (after
  flattening). Same math, different param shapes — `use_linear_in_transformer`
  selects it. This was the SDXL UNet strict-load failure.
- **SDXL `y` conditioning** = `cat([pooled(1280), size_emb(1536)])` → 2816-d, added
  to the time embedding via `label_emb`. `size_emb` is the sinusoidal
  `timestep_embedding(time_ids, 256)` of the 6 `time_ids`
  `(orig_h, orig_w, crop_top, crop_left, target_h, target_w)`, flattened — it
  matches diffusers' `add_time_proj` bit-for-bit.
- **Generalizing didn't fork the UNet.** One config-driven `UNetModel` covers both;
  the SD1.5 bit-exact oracle + smoke test are the regression guard (both stayed
  green through the SDXL changes).
- **SDXL VAE latent scale is 0.13025** (vs SD1.5 0.18215); it comes from
  `ModelSpec.latent_scale` and is applied at the VAE boundary as before. Same VAE
  architecture, run fp32.

Learned while integrating Anima (the first DiT family — Cosmos-Predict2 / 2 B):

- **Anima ships as three files.** DiT + VAE + TE come from different
  HuggingFace repos and live under separate ComfyUI-style directories on disk.
  `load_anima_checkpoint(dit_path, vae_path, te_path)` mirrors
  `load_checkpoint` but takes all three; everything past that is the same
  `ModelBundle` and a `TextToImage` dispatch on `spec.architecture == "anima"`.
- **Two tokenizers, only one encoder.** Anima feeds the prompt through *both*
  Qwen2.5 (consumed by the Qwen3 encoder) **and** T5 (token IDs only — *not*
  encoded; embedded directly inside the DiT's 6-block LLM-Adapter via a
  `[32128, 1024]` embedding table). The adapter cross-attends T5 token
  embeddings to Qwen3 hidden states to produce the DiT's 1024-d context.
- **Bit-match against `transformers.Qwen3Model` requires eager attention.**
  SDPA's flash/efficient kernels reorder reductions and lose ~1 fp32 ULP per
  layer; over 28 layers that compounds to ~7e-5 final-state drift. The Qwen3
  encoder uses explicit `matmul→softmax→matmul`; the perf cost at
  ≤512 tokens is negligible, and ``max\|Δ\|=0`` matches the SDXL bar.
- **No in-process oracle for DT4/DT5.** The local ComfyUI install needs a
  private native dep (`comfy_aimdo`) we don't have in the venv, so the
  LLM-Adapter and the DiT backbone fall back to strict-load + behavioural
  tests. Correctness is checked end-to-end at DT7 against a ComfyUI-generated
  reference for the same prompt/seed/shift/cfg.
- **fp16 residual stream overflows in the 28-block DiT.** First-pass Anima
  generation came back all-black. Cosmos's `MiniTrainDIT` promotes the
  residual to fp32 inside `_forward` while keeping attention/MLP in compute
  dtype; my DiT now does the same — cast x to fp32 once, cast in/out of each
  attention and MLP block, gates cast back to residual_dtype before the add.
  fp32 / CPU path unaffected; the `if dtype == fp16: x.float()` branch is a
  no-op there.
- **`net.*` prefix.** The Anima safetensors prepends `net.` to every key
  (DiT, adapter, etc.) — the existing `_load_sub(module, sd, "net.")`
  prefix-stripping idiom handles this without remapping. The VAE and TE
  checkpoints have no prefix; load them directly.
- **3D RoPE needs broadcastable shape.** `VideoRopePosition3DEmb` outputs
  `(L, D/2, 2, 2)`; you must `.unsqueeze(1).unsqueeze(0)` to
  `(1, L, 1, D/2, 2, 2)` so the L axis aligns over batch and head dims when
  applying it to a `(B, L, H, D)` query. (The Cosmos reference does the
  unsqueeze inline in its `_forward`.)
- **Patch-input channel count = 17, not 16.** The Anima DiT prepends a
  one-channel `concat_padding_mask` to the 16-channel Qwen-Image latent, so
  the patch-embed sees `(16+1)·2·2·1 = 68` features per patch (this is why
  `x_embedder.proj.1.weight` is `[2048, 68]`). Output is `16·2·2·1 = 64`
  (no mask channel on the way out).
- **Anima uses CONST flow with `multiplier=1`, so σ is t.** ComfyUI's
  `ModelType.FLOW` pairs `CONST` (model sees raw x, predicts v = ε − x0;
  denoised = x − σ·v) with `ModelSamplingDiscreteFlow` (shift-transformed
  schedule). With `multiplier=1` the DiT consumes σ directly as its
  timestep — no σ⇄t bridge needed. The existing σ-space Euler integrator
  is exact for any constant x0 estimate (closed-form linear interpolation).
- **Qwen-Image VAE shares the Wan-2.1 latent normalization.** 16-channel
  per-channel `(latents_mean, latents_std)` rather than the scalar
  `latent_scale` SD uses. The VAE exposes `process_in` / `process_out`;
  the pipeline applies `process_out` after sampling, before decode.
- **Tokenizer is vendored — no runtime `transformers` dep.** `AnimaTokenizer`
  loads `conditioning/qwen3_tokenizer.json` (Qwen3-0.6B, Apache-2.0) and
  `conditioning/t5_tokenizer.json` (google-t5/t5-11b, Apache-2.0) via the
  `tokenizers` library, matching the `clip_tokenizer.json` pattern. Both files
  are bit-identical to the ComfyUI `qwen25_tokenizer/` + `t5_tokenizer/` dirs
  they replace (parity test in `tests/test_anima_pipeline.py`).
- **3D RoPE is cached per generation.** `_VideoRoPE3D.forward` is a pure
  function of `(H, W, T, fps)` — all fixed across a single run — so the rotation
  tensor is computed once, stored on CPU, and moved to device on hit. Avoids
  recomputing freqs + `einops` expansions every step.
- **Anima CFG can't batch cond+uncond into one forward.** Unlike the SDXL
  `CFGDenoiser` (fixed 77-token context), Anima's Qwen3 hidden states and
  `t5xxl_ids` are variable-length per prompt, and there is no cross-attention
  mask — `torch.cat([cond, uncond])` fails on the seq axis. Padding to a common
  length before the LLM adapter would change adapter numerics vs. the separate
  path, so the Euler loop and `denoise` closure keep two forwards. Don't re-attempt
  batching here without real padding + masking support.

Learned while adding img2img / inpainting and v-prediction:

- **img2img/inpaint share the pipeline base.** `_base._Pipeline` holds the
  conditioning / sigma build / staged sample / staged decode / `_encode_image`;
  `TextToImage`, `ImageToImage`, `Inpaint` are thin `__call__`s differing only in
  the initial latent. Strength slices the sigma schedule at `img2img_start`.
- **Inpainting needs no sampler change.** `MaskedDenoiser` forces the keep region
  of the x0 estimate to `z0`; the sampler ODE `dx/dσ = (x − z0)/σ` has the exact
  solution `z0 + noise·σ` (Euler/Heun integrate it exactly), so that region lands
  on `z0` at σ→0. Original pixels are composited back after decode for byte-exact
  preservation.
- **v-prediction is a one-flag switch.** Detection sets `spec.prediction = "v"`
  from the bare `v_pred` marker tensor; the pipeline then uses `VScaling`. The
  weights are identical to an eps model, so nothing else changes — but feeding a
  v-pred model through `EpsScaling` yields pure noise, so the flag must be right.
- **`position_ids` is non-persistent.** It's the constant `arange(0..77)`, not a
  weight; many SDXL finetunes (e.g. AnimaTensor) drop it. Registered
  `persistent=False` and stripped in `_load_sub`, so checkpoints with *or* without
  it both strict-load. (The marker tensors `v_pred`/`ztsnr` are likewise ignored —
  they match no module prefix.)
- **ZTSNR rescales the schedule.** Detected from the `ztsnr` marker;
  `DiscreteSchedule(..., zero_terminal_snr=True)` shifts/scales `alphas_cumprod`
  so terminal SNR≈0 (σ_max ~14.6 → ~4500, terminal clamped to ε to stay finite),
  anchored so σ_min is unchanged. Pairs with CFG rescale (`cfg_rescale`, default
  0.7 for ZTSNR) in `CFGDenoiser`.
- **σ² must be fp32 (ZTSNR's teeth).** The sampler runs fp16, but σ_max≈4500 makes
  `σ²=2e7` overflow fp16's 65504 ceiling → inf → `c_in`/`c_skip` zero out → the
  latent collapses to pure black. `Scaling.scalings` computes the σ²-terms in fp32
  and casts the coefficients back to the latent dtype, so the normal fp16 path is
  unchanged but ZTSNR's huge σ no longer overflows. (This is the concrete bite of
  the "σ math stays fp32" rule — it had been latent because normal σ_max≈14.6.)

Learned while adding LoRA/LoKr (`lora.py`):

- **Fuse, don't wrap.** `apply_lora` adds the adapter's ΔW straight into the
  module weights. Since the engine already moves *modules* around for offload,
  the fused weights travel with them for free — no forward hooks, no change to
  the sampling loop. Compute ΔW in fp32 and cast back (the up/down matmul is
  precision-sensitive and CPU fp16 matmul is patchy).
- **Unfuse by snapshot, not subtraction.** LoRAs stack; `remove_lora` /
  `clear_loras` undo a fuse by restoring a pristine CPU snapshot taken on a
  weight's first touch, then replaying the remaining stack — exact (bit-identical
  in fp16), where add-then-subtract would drift. The snapshot is keyed by
  `weight.data_ptr()` (`.data` returns a fresh object each access, so `id()` is
  not stable; the storage pointer is, and the held weight ref keeps it alive), so
  bigG's shared `in_proj_weight` is snapshotted once across its q/k/v keys. State
  lives on the bundle as `_lora_state` (created lazily — no `ModelBundle` field).
  Memory: one CPU copy per *touched* module (union across the stack, capped at
  model size), bounded regardless of how many LoRAs you swap.
- **Map keys by walking `named_modules()`, not by un-mangling.** kohya keys are
  the dotted module path with `.`→`_`, which is ambiguous to reverse (names
  contain underscores). Building `{mangled_name: module}` from the live module
  tree sidesteps it entirely.
- **bigG is the SD special case.** SDXL `te2` kohya keys are split q/k/v, but our
  OpenCLIP stores a fused `in_proj_weight`; the q/k/v deltas are added to its
  row-slices. (Algo-agnostic — works for LoRA and LoKr alike.)
- **LoKr full-matrix scale is 1.** `ΔW = kron(w1, w2)`. The `alpha/dim` scaling
  only applies when a low-rank `_b` factor supplies `dim` (matching ComfyUI);
  full `lokr_w1`/`lokr_w2` fuse unscaled. Conv layers carry the kernel in `w2`
  (4-D), so lift `w1` to 4-D before the Kronecker product.
- **Anima ships LoRAs in two conventions.** Native ComfyUI/musubi uses dotted
  `diffusion_model.<path>` + `lora_A`/`lora_B`; LyCORIS/kohya uses mangled
  `lora_unet_<path>` + `lokr_w1`/`lokr_w2`. The Anima target map registers the
  DiT under *both* so either file type maps.
- **The Anima runtime tokenizer is vendored (no `transformers`).** It now loads
  `tokenizer.json` files via the `tokenizers` library; `transformers` is only a
  dev-time oracle (e.g. the vendored-tokenizer parity test). (Surfaced while
  running the Anima LoRA end-to-end.)

Learned while expanding the sampler / scheduler set (`samplers.py`, `schedules.py`):

- **One sampler, two model families, via the half-logSNR mapping.** ComfyUI's
  modern k-diffusion samplers (DPM++ SDE family, ER-SDE) are flow-aware by going
  through `er_lambda = sigma/alpha` (`alpha = 1` for VE, `alpha = 1 − sigma` for
  flow/CONST). Reproducing that one mapping (`_half_log_snr` + a first-σ offset
  off 1.0, where `alpha` is 0 and the logSNR is infinite) lets a single function
  serve SD/SDXL and Anima. `dpmpp_2m` and `dpm_2` are the exception: ComfyUI runs
  them in the plain VE form on flow too, so they're left model-agnostic to match.
- **The Anima path routes non-Euler samplers through the registry.** It builds a
  CONST x0 denoiser closure (`x − σ·v`, with CFG, the 4D↔5D reshape, fp16→fp32)
  and calls `get_sampler(name)(...)`. Euler keeps its exact closed form. The
  solver math runs in fp32 (like ComfyUI), with the closure casting to the DiT's
  compute dtype at the backbone boundary.
- **Seeded Gaussian, not Brownian tree.** ComfyUI's SDE samplers default to a
  `torchsde` Brownian-tree noise sampler. Per-step independent Gaussians are a
  correct Euler–Maruyama realization (Brownian increments over disjoint intervals
  *are* independent), so quality/convergence are unaffected — what's lost is
  grid-consistency across step counts and exact ComfyUI parity. Chosen to avoid
  the dependency; revisit behind a flag if parity is ever wanted.
- **`simple` / `sgm_uniform` need the model's schedule, not just σ_min/σ_max.**
  They index the discrete σ table / timestep map, so they take the schedule
  object. For Anima (no discrete table) a `FlowSamplingView` mirrors ComfyUI's
  `ModelSamplingDiscreteFlow` (a 1000-entry flow σ table + invertible timestep
  map). `_base._sigmas` dispatches these via `_SCHEDULE_FROM_MODEL`.
- **SECANT is σ-space native, no λ mapping needed.** Where DPM++/ER-SDE all need
  the `_half_log_snr` + first-σ offset hack to handle Anima's bounded σ ∈ (0, 1],
  `sample_secant` operates in σ directly: it linearly extrapolates `x0` along the
  secant through the previous two denoised estimates, reconstructs
  `x_{i+1} = (1−σ_{i+1})·x0_pred + σ_{i+1}·ε`, and blends with Euler by
  `beta = curvature·(1 − |Δσ|/σ)`. Works on any descending σ schedule;
  `curvature=0` recovers Euler exactly.

## Project map

```
src/diffucore/
  sampling/      ✅ schedules, parameterization, samplers, denoiser
                   (incl. flow_matching_schedule + FlowMatchingConstScaling;
                    Euler/Heun/ancestral/DPM2/DPM++/ER-SDE/SECANT
                    + sgm_uniform/simple/flow_karras; OSS in optimal_steps.py)
  loading/       ✅ state_dict (safetensors), detect (SD1.5 + SDXL + Anima)
  models/        ✅ clip_text, open_clip_text, vae, unet (SD1.5 + SDXL)
                 ✅ qwen_image_vae, qwen3_text, llm_adapter, anima_dit (Anima)
  conditioning/  ✅ CLIPTokenizer (+clip_tokenizer.json), Conditioner, SDXLConditioner,
                   AnimaTokenizer (Qwen2.5 + T5; vendored tokenizer.json)
  runtime/       ✅ DevicePolicy + CPU offload + tiled VAE (R1–R4)
  pipelines/     ✅ TextToImage / ImageToImage / Inpaint (SD1.5 + SDXL; eps + v-pred)
                 ✅ TextToImage (Anima, via _anima.anima_text_to_image dispatch)
  bundle.py      ✅ load_checkpoint (SD1.5 + SDXL) + load_anima_checkpoint (Anima)
  lora.py        ✅ apply_lora / remove_lora / clear_loras: fuse + unfuse LoRA/LoKr (SD1.5 + SDXL + Anima)
docs/
  ARCHITECTURE.md         design + rationale (incl. DiT seams)
  ROADMAP.md              milestones + status (M0–SX, DT0–DT7)
  IMPLEMENTATION_SPEC.md  ← the build sheet for M4–M7 (SD1.5 specific)
  RUNTIME_SPEC.md         VRAM offload + tiled VAE (R1–R4 done; Anima TBD)
  HANDOFF.md              this file
tests/                    CPU suite + opt-in pipeline smoke (SD1.5 + SDXL + Anima)
```

## Licensing reminder

Apache-2.0. Implement architectures from the papers / your own understanding, not
by translating GPL-licensed source. You accepted the relicensing risk of
referencing such source as a guide; keeping the actual code original is what
gives the Apache-2.0 release its footing.
