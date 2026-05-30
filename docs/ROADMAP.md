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

## After SD1.5

Once M7 lands, the natural extensions (each its own slice): SDXL (dual text
encoders + larger UNet), img2img / inpainting (alternate initial latents +
masks), and a first DiT-style architecture to validate the §8 extensibility
seams. These are intentionally not scheduled until SD1.5 is solid.

## Verification notes

- The laptop (Intel iGPU, no CUDA) runs the CPU suite via a Python 3.11 venv with
  CPU-only PyTorch.
- CUDA milestones (M4–M7) are validated on the RTX 2060. fp16 is the default
  working dtype there; the VAE and σ math stay fp32.
