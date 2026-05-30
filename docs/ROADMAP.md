# Diffucore Roadmap

Milestones are vertical slices. Each has an explicit, checkable success
criterion. "CPU" criteria are verifiable on the dev laptop; "CUDA" criteria are
verified on the RTX 2060 (12 GB).

| # | Milestone | Success criterion | Where |
|---|-----------|-------------------|-------|
| M0 | Project scaffold | Package imports; `pytest` collects and runs. | CPU ✅ |
| M1 | Sampling foundation | σ schedules + σ⇄t + eps/v scalings, unit-tested (monotonicity, endpoints, round-trip). | CPU ✅ |
| M2 | Samplers + loop | Euler/Heun step in σ-space; on a toy linear denoiser the loop drives `x` to the known `x₀` within tolerance. | CPU ✅ |
| M3 | Checkpoint loading | Load an SD1.5 `.safetensors`; detect architecture + prediction type from keys/shapes; populate `ModelSpec`. | CPU ✅ |
| M4 | Text conditioning | CLIP ViT-L/14 tokenizer + encoder; embeddings match reference shapes/values within tolerance. | CUDA ✅ |
| M5 | VAE | AutoencoderKL encode/decode; `decode(encode(img)) ≈ img` (PSNR threshold). | CUDA ✅ |
| M6 | UNet (SD1.5) | eps backbone forward; single denoise step runs at 512² in fp16 under ~6 GB. | CUDA ✅ |
| M7 | **End-to-end t2i** | `TextToImage` produces a coherent 512² image from a real SD1.5 checkpoint; fixed seed reproducible. | CUDA ✅ |
| SX | **SDXL** | Dual text encoders (CLIP-L + OpenCLIP bigG) + generalized UNet + size/pooled conditioning; coherent 1024² image, seed reproducible. | CUDA ✅ |

## Current status

- **M0–M3 implemented and verified on CPU** (32 tests). The full sampling /
  denoising path (schedules, samplers, loop, CFG, parameterization) and
  checkpoint detection are done.
- **M4–M7 implemented and verified on the RTX 2060** against
  `v1-5-pruned-emaonly.safetensors`:
  - M4 CLIP — strict load; output bit-identical to `transformers.CLIPTextModel`
    (max|Δ|=0). Tokenizer vendored as `conditioning/clip_tokenizer.json`.
  - M5 VAE — strict load; `decode(encode(img))` PSNR 35.1 dB; O(1) latent stats.
  - M6 UNet — strict load; output bit-identical to diffusers
    `UNet2DConditionModel` (max|Δ|=0); fp16 forward ~1.8 GB peak.
  - M7 t2i — coherent 512² image from the SD1.5 checkpoint; fixed seed →
    bit-identical image; ~3.6 s / 20 steps at ~3.2 GB peak.
- The end-to-end SD1.5 text-to-image path (`load_checkpoint` → `TextToImage`) is
  working. HF `transformers`/`diffusers` are used only as numerical oracles in
  verification, not as runtime dependencies.
- **SDXL implemented and verified on the RTX 2060** (detection → both text
  encoders → generalized UNet → pipeline):
  - bigG (OpenCLIP) — strict load; penultimate hidden bit-identical to diffusers
    `text_encoder_2` (max|Δ|=0), pooled max|Δ|~1e-6.
  - Dual conditioner — 2048-d context bit-identical to diffusers `encode_prompt`
    (max|Δ|=0).
  - UNet — generalized `UNetModel` strict-loads SDXL and matches diffusers
    (fp16 relative ~1e-3); **SD1.5 stays bit-exact** (max|Δ|=0, smoke test green).
  - t2i — coherent 1024² image, seed-reproducible; ~19 s / 20 steps at ~10.7 GB.

## After SDXL

The remaining extensions (each its own slice):

- **img2img (latent-init) — done.** `ImageToImage` encodes an init image, adds
  noise at a `strength`-chosen point on the schedule, denoises the rest, and
  decodes (reuses the shared pipeline base, so offload + tiled VAE apply). Verified
  end-to-end on **SD1.5 and SDXL** (`AkashicPulse-v3.0` on the RTX 2060): RGB
  output, seed-reproducible, strength changes the result.
- **inpainting (masks) — done.** `Inpaint` repaints the white region of a mask and
  keeps the black region. No sampler changes: a `MaskedDenoiser` pins the keep
  region of the x0 estimate to the original latent `z0`, so the sampler's ODE
  (`dx/dσ = (x − z0)/σ`, integrated exactly by Euler/Heun for a constant target)
  carries that region along `z0 + noise·σ` and lands on `z0` at σ→0. After decode,
  the original pixels are composited back over the keep region (hard edge) so
  untouched areas are byte-exact. Verified on **SD1.5 and SDXL** (RTX 2060):
  keep region byte-exact, masked region repainted, seed-reproducible; the
  keep-region-pinning is also checked at the sampler level on CPU (no checkpoint).
  Mask blur / "inpaint at full res" are later refinements.
- **v-prediction checkpoints — done.** Detection reads the bare `v_pred` marker
  tensor (NoobAI / A1111 / reForge convention) into `ModelSpec.prediction`, and
  the pipeline selects `VScaling` vs `EpsScaling` from it — no other code path
  changes (eps and v-pred weights are otherwise identical). Verified on the RTX
  2060 with `AnimaTensor-Pro` (v-pred + ZTSNR SDXL): coherent 1024² image. Along
  the way, CLIP's `position_ids` became a non-persistent buffer (a derived
  `arange` constant many finetunes omit), so such checkpoints load strictly.
- **zero-terminal-SNR (ZTSNR) — done.** Detected from the `ztsnr` marker; the
  schedule is rescaled (Lin et al., 2024) so terminal SNR ≈ 0, raising σ_max from
  ~14.6 to ~4500 — the model finally starts from true pure noise and can render
  full-range darks/brights. Paired with **CFG rescale** (a `cfg_rescale` knob on
  every pipeline, default 0.7 for ZTSNR checkpoints) to curb high-CFG
  over-exposure. Verified on `AnimaTensor-Pro`: a "pitch-black" prompt yields a
  genuinely dark, detailed image (mean 22 vs the plain schedule's washed floor),
  a bright prompt stays vivid. **Required an fp32 fix in `Scaling.scalings`** —
  σ_max≈4500 makes σ² overflow fp16 (→inf→black); the coefficients now compute in
  fp32 and cast back, leaving the normal fp16 path unchanged.
- **VRAM management for SDXL on smaller cards — R1–R4 done.** Sequential CPU
  offload + tiled VAE, specced in [`RUNTIME_SPEC.md`](RUNTIME_SPEC.md). Verified on
  the RTX 2060 against `AkashicPulse-v3.0` (SDXL, 1024²): R1–R3 — offload
  byte-identical to all-resident, tiled-VAE PSNR 37.55 dB, peak VRAM 9.97 GB →
  6.6 GB; R4 — `offload="encoders"` cheap mode (UNet stays resident) is also
  byte-identical, and a 1024² generation completes under an emulated 8 GB cap.
- **First DiT-style architecture — Anima (Cosmos-Predict2-2B family) — done.**
  `load_anima_checkpoint(dit, vae, te)` → `TextToImage` produces a coherent
  1024² image (RTX 2060, ~46 s / 20 steps / 8.6 GB peak). Components, each
  verified at landing:
  - **Detection** — Anima identified by ``net.llm_adapter.blocks.0.cross_attn.q_proj.weight``;
    spec carries ``architecture="anima"``, ``prediction="flow"``, ``latent_channels=16``,
    ``context_dim=1024``, ``image_size=1024``.
  - **Qwen-Image VAE** — 3D-causal-conv autoencoder shared with Wan2.1
    (16-channel latents, 8× spatial); image-only T=1 path. Strict-load 194/194
    keys; round-trip PSNR 49 dB on a smooth gradient (asserted > 40).
  - **Qwen3 0.6B text encoder** — 28 layers, GQA (16/8), head_dim 128, RoPE
    θ=1e6, per-head q/k RMSNorm before RoPE, SwiGLU MLP, eager attention.
    Strict-load 310/310; **max|Δ|=0 in fp32** vs `transformers.Qwen3Model`.
  - **LLM-Adapter** — 6-block bridge transformer at dim 1024; embeds T5 token
    IDs as the "query stream" and cross-attends to Qwen3 hidden states. RoPE
    θ=10000 (not Qwen's 1e6). Strict-load 118/118. Behavioral tests cover
    determinism + sensitivity to source/target/mask.
  - **Anima DiT** — 28-block adaLN-LoRA transformer (model_channels=2048,
    16 heads × head_dim 128) with 3D RoPE on self-attn only. Three independent
    adaLN modulators per block (self/cross/MLP). Strict-load 685/685; **2.09 B
    params** — matches the model's 2B label.
  - **Flow-matching parameterization + schedule** —
    `FlowMatchingConstScaling` (`c_skip=1, c_out=−σ, c_in=1`; model predicts
    velocity v = ε − x0) + `flow_matching_schedule(steps, shift)` (SD3-style
    shift; Anima uses 3.0). The existing σ-space Euler sampler integrates
    rectified-flow ODE *exactly*; no sampler changes needed.
  - **fp16 stability fix** — Cosmos's residual stream accumulates past
    fp16's ±65504 ceiling over 28 blocks → NaN. The DiT now promotes the
    residual to fp32 inside `_forward` while keeping attention/MLP in
    compute_dtype, casting back to fp32 before each gated residual add
    (same shape as NVIDIA's reference). fp32 / CPU path unaffected.
  - **Tokenizer** — `AnimaTokenizer` lazily loads Qwen2.5 + T5 tokenizers
    via `transformers` from a vendored ComfyUI tokenizer directory.
    Vendoring proper `tokenizer.json` files under `conditioning/` (to drop
    the runtime `transformers` dep on the Anima path) is a follow-up.
  - **Verification posture** — DT3 (Qwen3) bit-matches `transformers`
    in fp32. ComfyUI can't be imported in our venv (missing private
    `comfy_aimdo` native dep), so DT4 (adapter) and DT5 (DiT) rely on
    strict-load + behavioral tests; correctness is confirmed end-to-end at
    DT7 by visual inspection vs the same prompt/seed in ComfyUI.
- **LoRA / LoKr application — done.** `apply_lora(bundle, path, multiplier)`
  fuses adapter weight deltas into the loaded modules in place (no forward
  wrapping, so offload and the sampling path are untouched). Covers two
  factorizations — **LoRA** (`ΔW = (alpha/rank)·up@down`) and **LoKr**
  (`ΔW = kron(w1, w2)`, full matrices or low-rank `a@b` factors with
  ComfyUI-matching `alpha/dim` scaling) — and three naming families:
  kohya/A1111 (`lora_unet_`/`lora_te_`/`lora_te1_`/`lora_te2_`, mangled paths),
  PEFT `lora_A`/`lora_B`, and Anima's `diffusion_model.`-dotted form. bigG's
  fused `in_proj_weight` is the one special case (split q/k/v deltas land in
  row-slices). Verified on the RTX 2060 against real files: SDXL LoKr
  (1052/1052 modules), Anima LoRA (448/448) and Anima LoKr (280/280) — all
  0 unmatched, coherent output, fused ΔW matching an independent reference to
  fp16 precision. CPU unit tests run on tiny real-structure models
  (`tests/test_lora.py`). Diffusers-format SD LoRAs and the LoHa variant are
  not supported (their keys are reported in `LoraReport.unmatched`).

## Verification notes

- The laptop (Intel iGPU, no CUDA) runs the CPU suite via a Python 3.11 venv with
  CPU-only PyTorch.
- CUDA milestones (M4–M7, SDXL) are validated on the RTX 2060. fp16 is the
  default working dtype there; the VAE and σ math stay fp32.
- Numerical oracles (`transformers`/`diffusers`) require `transformers<5` to load
  SDXL checkpoints via diffusers `from_single_file` / `StableDiffusionXLPipeline`.
