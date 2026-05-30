# Handoff — continuing diffucore on the RTX 2060 PC

This is the pickup point for moving from the laptop (no CUDA) to the RTX 2060
12 GB PC, where the model components can finally be run and verified.

## What's done and verified (laptop, CPU)

| Milestone | What | Verified |
|---|---|---|
| M0 | Scaffold: Apache-2.0, pyproject, package layout, docs | imports + `pytest` |
| M1 | Sampling foundation: σ schedules, σ⇄t, eps/v scalings | 15 tests |
| M2 | Samplers (Euler/Heun/ancestral) + denoising loop + CFG | 10 tests |
| M3 | Checkpoint loading + SD1.5 detection (from safetensors header) | 7 tests |

**32 tests pass.** The entire sampling/denoising path is real and tested against
analytic solutions — you should not need to touch it.

## What's left (this PC)

Skeletons exist (importable, `forward` raises `NotImplementedError`); fill them in
following [`IMPLEMENTATION_SPEC.md`](IMPLEMENTATION_SPEC.md), then run the
per-module verification there.

| Milestone | File | Builds |
|---|---|---|
| M4 | `models/clip_text.py`, `conditioning/__init__.py` | CLIP text encoder + tokenizer |
| M5 | `models/vae.py` | AutoencoderKL |
| M6 | `models/unet.py` | SD1.5 UNet; then finish `bundle.py` weight loading |
| M7 | `pipelines/text_to_image.py` | end-to-end (wiring is already spec'd) |

Recommended order: **VAE → CLIP → UNet → pipeline** (VAE is the most
self-contained and gives you `decode` to eyeball latents early).

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

## Project map

```
src/diffucore/
  sampling/      ✅ schedules, parameterization, samplers, denoiser   (DONE, tested)
  loading/       ✅ state_dict (safetensors), detect (ModelSpec)       (DONE, tested)
  models/        ⏳ clip_text, vae, unet                               (SKELETON → M4–M6)
  conditioning/  ⏳ CLIPTokenizer, Conditioner                         (SKELETON → M4)
  runtime/       ⏳ DevicePolicy (auto() works; offload/tiling TODO)
  pipelines/     ⏳ TextToImage (wiring spec'd in IMPLEMENTATION_SPEC)  (SKELETON → M7)
  bundle.py      ⏳ load_checkpoint (detection done; module build TODO)
docs/
  ARCHITECTURE.md         design + rationale
  ROADMAP.md              milestones + status
  IMPLEMENTATION_SPEC.md  ← the build sheet for M4–M7 (read this first)
  HANDOFF.md              this file
tests/                    32 passing CPU tests
```

## Licensing reminder

Apache-2.0. Implement architectures from the papers / your own understanding, not
by translating GPL-licensed source. You accepted the relicensing risk of
referencing such source as a guide; keeping the actual code original is what
gives the Apache-2.0 release its footing.
