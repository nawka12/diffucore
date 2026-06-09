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
    (max|Δ|=0) for a short, unweighted prompt. The conditioner also does LPW (long
    prompt weighting), which intentionally diverges for weighted / >77-token prompts.
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
    shift; Anima uses 3.0). The existing σ-space Euler sampler integrates the
    rectified-flow ODE *exactly*; no sampler changes were needed to land Anima.
    (Flow-aware DPM++ / ER-SDE samplers were added later — see the sampler /
    scheduler entry below.)
  - **fp16 stability fix** — Cosmos's residual stream accumulates past
    fp16's ±65504 ceiling over 28 blocks → NaN. The DiT now promotes the
    residual to fp32 inside `_forward` while keeping attention/MLP in
    compute_dtype, casting back to fp32 before each gated residual add
    (same shape as NVIDIA's reference). fp32 / CPU path unaffected.
  - **Tokenizer** — `AnimaTokenizer` loads vendored `qwen3_tokenizer.json`
    (Qwen3-0.6B, Apache-2.0) + `t5_tokenizer.json` (google-t5/t5-11b,
    Apache-2.0) via the `tokenizers` library — no runtime `transformers` dep.
    Both are bit-identical to the ComfyUI tokenizer dirs they replace.
  - **Verification posture** — DT3 (Qwen3) bit-matches `transformers`
    in fp32. ComfyUI can't be imported in our venv (missing private
    `comfy_aimdo` native dep), so DT4 (adapter) and DT5 (DiT) rely on
    strict-load + behavioral tests; correctness is confirmed end-to-end at
    DT7 by visual inspection vs the same prompt/seed in ComfyUI.
- **FLUX family (FLUX.1 + FLUX.2) — implemented, build-to-spec.**
  `load_flux_checkpoint(...)` → `TextToImage` runs both families end-to-end on
  tiny models (CPU), seed-reproducible. **Not yet GPU-verified against real
  weights** — implemented from the published architecture and cross-checked
  against the Black Forest Labs / ComfyUI reference (block structure, config
  detection, latent format, schedule); a strict no-missing-keys load is the
  correctness gate, with numerical parity deferred to hardware verification.
  Components:
  - **Detection** — both families carry double-stream blocks
    (`double_blocks.0.img_attn.qkv.weight`), located bare or under an all-in-one
    `model.diffusion_model.` prefix. FLUX.2 is told apart by its *global*
    modulators (`double_stream_modulation_img.lin.weight`). Widths/depths derive
    from tensor shapes; the family constants live in `bundle._flux_arch`.
  - **FLUX DiT** (`models/flux_dit.py`) — one config-driven MMDiT for both.
    **FLUX.1**: per-block `img_mod`/`txt_mod` AdaLN, GELU-tanh MLP, biases on,
    axial RoPE `(16,56,56)` θ=10000, `qkv_bias=True`; the pipeline 2×2-patchifies
    (in_channels 64). **FLUX.2**: three *shared* bias-free modulators
    (`double_stream_modulation_img`/`_txt`, `single_stream_modulation`) drive
    every block, SiLU-gated MLP, no biases, RoPE `(32,32,32,32)` θ=2000,
    patch_size 1 (in_channels 128, text ids positioned on axis 3).
  - **VAE** — the FLUX autoencoders reuse the LDM `AutoencoderKL` layout, so the
    config (channel_mult, z_channels, quant-conv presence) is inferred from the
    weights: FLUX.1 16-ch / 8× with scale 0.3611 + shift 0.1159; FLUX.2 128-ch /
    16× with no scale/shift. `VAEConfig` gained `shift_factor` + `use_quant_conv`
    (both default to the SD behaviour, so SD/SDXL load byte-identically).
  - **Text encoders** — **FLUX.1**: T5-XXL v1.1 encoder (`models/t5_text.py`,
    relative-position bias, gated-GELU; reuses the vendored `t5_tokenizer.json`)
    for the sequence context + CLIP-L's pooled vector (a `return_pooled` path
    added to `CLIPTextEncoder`, SD/SDXL output unchanged). **FLUX.2 Klein**: a
    single Qwen3-4B/8B encoder, reusing Anima's `Qwen3TextEncoder` with an
    intermediate-layer (`[9,18,27]`) capture; the context is those three layers
    concatenated (no final norm), with the Qwen chat template and the vendored
    Qwen2.5 tokenizer. **FLUX.2 Dev**: a Mistral-3 24B encoder
    (`models/mistral_text.py`, layers `[10,20,30]`, SYSTEM_PROMPT template) — the
    secondary path; its Tekken tokenizer is not vendored.
  - **Schedule / parameterization** — reuses `FlowMatchingConstScaling` and
    `flow_matching_schedule`; the shift matches ComfyUI's `flux_time_shift`
    (FLUX.1-dev resolution-interpolated `exp(mu)`, schnell 1.0, FLUX.2
    `exp(2.02)`). Euler is exact; the flow-aware DPM++/ER-SDE/SECANT samplers
    drive FLUX too, via the same CONST x0 closure the Anima path uses.
  - **Loading** — `load_flux_checkpoint` takes an all-in-one checkpoint or split
    files (transformer / VAE / text encoder(s)); each component is located by a
    fingerprint-leaf key and its on-disk prefix is stripped, so bare BFL files
    and nested all-in-one layouts both load. `load_checkpoint` routes a detected
    FLUX checkpoint here automatically.
  - **What to confirm on real weights** — the FLUX.2 VAE key layout (assumed
    LDM-style, mid-only attention), Qwen3/Mistral padding+mask handling, and the
    exact guidance/shift defaults. FLUX now supports img2img + soft (latent-mask)
    inpaint via `flux_img2img`, mirroring the Anima flow path; a dedicated inpaint
    model (FLUX.1 Fill) is not implemented.
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
  fp16 precision. LoRAs stack, and `remove_lora(bundle, path)` /
  `clear_loras(bundle)` unfuse without reloading the checkpoint — a pristine CPU
  snapshot (taken on first touch) is restored and the rest of the stack replayed,
  so an unfuse is bit-exact and swapping is memory-bounded. CPU unit tests run on
  tiny real-structure models (`tests/test_lora.py`, `tests/test_lora_swap.py`).
  Diffusers-format SD LoRAs and the LoHa variant are not supported (their keys
  are reported in `LoraReport.unmatched`).
- **Inference perf flags — PR-A + PR-B + PR-C done.** Five knobs on
  `DevicePolicy` (`cudnn_benchmark`, `tf32`, `channels_last`, `compile`,
  `cuda_graphs`); `cudnn_benchmark` defaults **on** (bit-exact, 3-17 % free
  win), the rest default off and opt-in only.
  PR-A wires the cuDNN/matmul backend flags through a `runtime.perf_context`
  context manager that flips state for the duration of a pipeline call and
  restores it on exit, and converts the SD/SDXL UNet + AutoencoderKL to
  channels_last NHWC at load. PR-B adds `torch.compile(backbone, dynamic=True)`
  at bundle finalize, gated on the offload mode (raises if compile would race
  CPU↔GPU shuttling under `offload=True`); LoRA's target walker unwraps
  `OptimizedModule._orig_mod` so `apply_lora`/`remove_lora` still find kohya
  paths through the compile wrapper. PR-C switches compile to
  `mode="reduce-overhead", dynamic=False` when `cuda_graphs=True` so Inductor
  captures a CUDA Graph and replays it per step; requires `compile=True` and
  re-records on any input-shape change. Validated on the RTX 2060 (Turing)
  against `v1-5-pruned-emaonly`, `AkashicPulse-v3.0` (SDXL) and
  `anima-base-v1.0`: Anima 52.2 s → **35.6 s (1.47×)** with `cudnn_benchmark +
  compile + cuda_graphs` (and *higher* PSNR than compile alone — 54.7 dB vs
  29.0 dB — because static shapes pick consistent kernels), SDXL 19.8 s →
  16.9 s (1.17×) with `cudnn_benchmark` alone (bit-exact). Anima + cuda_graphs
  is 1024²-only on 12 GB cards: CFG runs cond/uncond as two different-shape
  forwards, so each step holds two captured graph pools whose per-pool
  activation memory scales as O(tokens²); above 1024² the pools blow the
  budget — use `compile=True` alone there. See `docs/RUNTIME_SPEC.md` §Perf
  flags for the full results and per-architecture recommendations.
- **Sampler / scheduler set — done.** Beyond the original Euler/Heun/ancestral,
  the registry now carries **DPM2** (+ancestral), **DPM++** (`2m`, `sde`,
  `2m_sde`, `3m_sde`), **ER-SDE-Solver-3**, and **SECANT** (σ-space x0-secant
  multistep), plus the **`sgm_uniform`**, **`simple`**, and **`flow_karras`**
  schedulers. `flow_karras` is a Karras-ρ-warped rectified-flow schedule (one
  interpretable `rho` knob, `rho=1` reduces exactly to `flow`); it replaced the
  earlier hand-tuned ACAS schedule, whose multi-bump density had no error
  criterion. The genuinely model-aware path is **`oss`** — an optimal-stepsize
  schedule (Pu et al. 2025, arXiv:2503.21774) distilled offline by a DP over a
  teacher trajectory's per-step error (`sampling/optimal_steps.py`,
  `calibrate_oss.py`). The DPM++ and ER-SDE family are flow-aware via the
  half-logSNR mapping (with a first-σ offset for flow) so the same function
  drives both VE (SD/SDXL) and rectified-flow (Anima) models, matching ComfyUI's
  `model_sampling`-aware k-diffusion. SECANT is σ-space native — no λ mapping,
  no first-σ offset — and works on any descending σ schedule via a `curvature`
  knob. The Anima path routes any non-Euler sampler through the
  shared registry against a CONST x0 denoiser closure, and accepts
  `scheduler ∈ {flow, flow_karras, oss, sgm_uniform, simple}`. The SDE samplers re-inject
  **seeded Gaussian noise** (a standard Euler–Maruyama discretization) rather
  than ComfyUI's Brownian-tree noise — correct and seed-reproducible, but not
  bit-identical to a ComfyUI render, and avoiding a `torchsde` dependency.
  Sampler convergence (VE + flow) and the new schedules are unit-tested on CPU.

## Verification notes

- The laptop (Intel iGPU, no CUDA) runs the CPU suite via a Python 3.11 venv with
  CPU-only PyTorch.
- CUDA milestones (M4–M7, SDXL) are validated on the RTX 2060. fp16 is the
  default working dtype there; the VAE and σ math stay fp32.
- Numerical oracles (`transformers`/`diffusers`) require `transformers<5` to load
  SDXL checkpoints via diffusers `from_single_file` / `StableDiffusionXLPipeline`.
