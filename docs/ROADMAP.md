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
- **VRAM management for SDXL on smaller cards — R1–R4 done.** Sequential CPU
  offload + tiled VAE, specced in [`RUNTIME_SPEC.md`](RUNTIME_SPEC.md). Verified on
  the RTX 2060 against `AkashicPulse-v3.0` (SDXL, 1024²): R1–R3 — offload
  byte-identical to all-resident, tiled-VAE PSNR 37.55 dB, peak VRAM 9.97 GB →
  6.6 GB; R4 — `offload="encoders"` cheap mode (UNet stays resident) is also
  byte-identical, and a 1024² generation completes under an emulated 8 GB cap.
- A first DiT-style architecture to validate the §8 extensibility seams.

## Verification notes

- The laptop (Intel iGPU, no CUDA) runs the CPU suite via a Python 3.11 venv with
  CPU-only PyTorch.
- CUDA milestones (M4–M7, SDXL) are validated on the RTX 2060. fp16 is the
  default working dtype there; the VAE and σ math stay fp32.
- Numerical oracles (`transformers`/`diffusers`) require `transformers<5` to load
  SDXL checkpoints via diffusers `from_single_file` / `StableDiffusionXLPipeline`.
