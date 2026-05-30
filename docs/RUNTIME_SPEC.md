# Runtime / VRAM Spec — sequential offload + tiled VAE

> **Status: R1–R3 implemented and verified on the RTX 2060.** `DevicePolicy` now
> plumbs through `load_checkpoint` → `ModelBundle` → `TextToImage` (R1);
> `tiled_vae_decode` is built and auto-triggers ≥768 px (R2); sequential CPU
> offload (`offload=True`, via the `on_device` ctx manager) is built (R3).
> Verified against `AkashicPulse-v3.0` (SDXL) at 1024²: offload is **byte-identical**
> to all-resident; tiled vs untiled decode is **PSNR 37.55 dB**; peak VRAM drops
> from **9.97 GB (resident, untiled) → 6.6 GB (offload + tiled)**. The ~6 GB
> stretch target needs R4 (encoders-only offload) or attention slicing — both
> out of scope here. The design notes below are kept as the build record.

## Why this exists

SD1.5 at 512² fits comfortably (~3.2 GB peak). **SDXL at 1024² peaks at 10.7 GB**
on a 12 GB card — it runs, but with almost no headroom, and it will OOM on 8 GB
cards or at higher resolutions / batch > 1. The goal is to make SDXL comfortable
on 12 GB and *possible* on 8 GB, without changing the numerics (same seed → same
image, bit-for-bit).

### Measured breakdown (SDXL, 1024², fp16 UNet/CLIP, fp32 VAE, batch 1)

Resident weights (always loaded today): **7.10 GB**

| Module | Size | dtype | Used during |
|---|---|---|---|
| UNet | 5.13 GB | fp16 | the sampling loop only |
| OpenCLIP bigG | 1.39 GB | fp16 | conditioning only (start) |
| CLIP-L | 0.25 GB | fp16 | conditioning only (start) |
| VAE | 0.33 GB | fp32 | decode only (end) |

Per-stage peak allocation (reset between stages):

| Stage | Peak | Notes |
|---|---|---|
| UNet sampling loop | **7.66 GB** | 5.13 GB weights + ~0.5 GB activations + CFG batch |
| VAE decode (1024², fp32) | **10.71 GB** | a 0.33 GB module spilling **~3.5 GB of activations** |

**Reading of the data:**
- The **VAE decode is the single worst spike** — the activation tensors for a
  1024² fp32 decode dwarf the weights. This is the first thing to fix.
- The **resident floor is 7.1 GB** even though no stage needs all four modules at
  once. The text encoders (1.64 GB combined) sit idle through the entire sampling
  loop; the UNet (5.13 GB) sits idle through decode.

**Target after this work** (sequential offload + tiled VAE):
- UNet loop stays the bottleneck at ~5.7 GB (UNet weights + activations), because
  the text encoders are offloaded by then.
- VAE decode drops well under that with tiling.
- **Expected peak ≈ 5.7–6 GB** → fits 8 GB cards; ample headroom on 12 GB. To go
  *below* the UNet floor would need attention slicing or UNet chunking (a later,
  separate slice — see Non-goals).

## Current state

- `runtime/DevicePolicy` (skeleton): holds `device`, `compute_dtype`,
  `vae_dtype`, and an **`offload: bool` that nothing reads**. `auto()` picks
  CUDA+fp16 or CPU+fp32.
- `bundle.load_checkpoint(path, device, dtype)` builds all four modules and
  **eagerly `.to(device)`s every one of them**, then returns a `ModelBundle`.
  There is no `DevicePolicy` plumbed through; `device`/`dtype` are bare args.
- `pipelines/text_to_image.py` reads `device`/`dtype` off
  `next(model.backbone.parameters())` and assumes every module is already on the
  device. VAE decode is a single `model.vae.decode(x0)` call (no tiling).

So today placement is "everything resident, always." The seams to change are
small and localized: the bundle's placement, and the pipeline's decode call.

## Goals

1. **Sequential CPU offload** (opt-in): keep only the module a stage needs on the
   GPU; park the others in CPU RAM and move them on/off around each stage.
2. **Tiled VAE decode** (opt-in, auto-triggered by resolution): decode the latent
   in overlapping spatial tiles and blend, bounding decode activation memory.
3. **Wire `DevicePolicy` through** `load_checkpoint` → `ModelBundle` → `TextToImage`
   so placement decisions live in one object, not scattered `.to()` calls.
4. **Zero numerical change when offload is off**; **negligible change when on**
   (offload must be bit-identical; tiling is approximate — see Verification).

## Non-goals (explicitly out of scope for this slice)

- Attention slicing / sub-quadratic attention in the UNet (would lower the ~5.7 GB
  UNet floor, but it's a separate slice with its own verification).
- Tiled VAE *encode* (only needed for img2img, which isn't built yet).
- Multi-GPU, CPU-only generation speedups, quantization (int8/fp8), torch.compile.
- A general "accelerate"-style hook system. Keep it to two concrete techniques.

## Design

### A. DevicePolicy as the single placement authority

Extend `DevicePolicy` with the offload/tiling knobs and a couple of helpers.
Sketch (names are suggestions, not binding):

```python
@dataclass
class DevicePolicy:
    device: torch.device
    compute_dtype: torch.dtype = torch.float16
    vae_dtype: torch.dtype = torch.float32
    offload: bool = False            # sequential CPU offload of idle modules
    vae_tile: bool = False           # force tiled VAE decode
    vae_tile_threshold: int = 768    # auto-tile when latent H or W*8 exceeds this (px)

    @property
    def offload_device(self) -> torch.device:
        return torch.device("cpu")
```

`auto()` should keep returning the same defaults (offload off) so existing
behavior is unchanged unless the caller opts in.

**Plumbing:** add an optional `policy: DevicePolicy | None` param to
`load_checkpoint`. When `None`, construct one from the existing `device`/`dtype`
args (back-compat). Store the policy on `ModelBundle` (new field, default `None`
→ pipeline falls back to "all resident"). The pipeline reads placement from the
policy instead of sniffing `backbone.parameters()`.

### B. Sequential CPU offload

**Principle:** the three stages (conditioning → sampling → decode) are sequential
and disjoint in what they need resident. Move each module to the GPU just before
its stage and back to CPU right after.

- With `offload=True`, `load_checkpoint` leaves modules on **CPU** (pinned memory
  if cheap to do) and does *not* eagerly place them.
- The pipeline wraps each stage:
  - **Conditioning:** CLIP-L + bigG → GPU, run `SDXLConditioner` (or
    `Conditioner`), move both back to CPU. Keep the small output tensors
    (`context`, `pooled`, `y`) on GPU — they're tiny.
  - **Sampling loop:** UNet → GPU, run all steps, move back to CPU.
  - **Decode:** VAE → GPU, decode, move back to CPU.
- A tiny context manager keeps it readable and exception-safe:

```python
@contextmanager
def on_device(module, device):
    module.to(device)
    try:
        yield module
    finally:
        module.to("cpu")
        torch.cuda.empty_cache()   # release the freed blocks back to the driver
```

**Expected peak with offload (no tiling):** the UNet stage (~5.7 GB) since the
text encoders are gone by then. The text-encoder stage is ~1.7 GB; decode is
~3.8 GB untiled. So offload alone takes the 10.7 GB peak down to ~5.7 GB.

**Cost:** host↔device copies. UNet is 5.13 GB moved twice per image (~once each
way). On PCIe 3.0 x16 (~12 GB/s) that's ~0.4 s each way — a few percent on a
~19 s SDXL run. Acceptable; document it. (A later optimization could keep the
UNet resident and only offload the encoders + VAE, which is the cheaper 80/20:
frees ~2 GB for ~0 copy cost. Consider exposing `offload="encoders"` vs
`offload="full"`.)

### C. Tiled VAE decode

**Principle:** the decoder is convolutional, so a large feature map can be decoded
in overlapping spatial tiles and stitched, bounding peak activation memory to one
tile regardless of output resolution.

- Split the latent `[B,4,h,w]` into tiles of latent size `T` (e.g. 64 → 512 px
  output) with an **overlap** `O` (e.g. 16 latent px) on each inner edge.
- Decode each tile through `vae.decode`, producing `[B,3,8T,8T]` pixel tiles.
- **Blend the overlaps** with a linear (or raised-cosine) ramp so seams vanish.
  A hard cut leaves visible grid lines; the ramp is what makes tiling acceptable.
- Trigger automatically when `vae_tile` is set or output ≥ `vae_tile_threshold`.

Put this as a function in `runtime/` (e.g. `tiled_vae_decode(vae, latent, tile,
overlap)`) and have the pipeline call it instead of `vae.decode` when the policy
says so. The VAE module itself stays untouched (don't bake tiling into the model).

**Expected peak for decode with tile=64, overlap=16:** roughly the cost of a 512²
decode (~1–1.5 GB activations) instead of 3.5 GB, independent of final size.

## Integration points (where the edits land)

| File | Change |
|---|---|
| `runtime/__init__.py` | Extend `DevicePolicy` (knobs + `offload_device`); add `on_device` ctx manager and `tiled_vae_decode`. (Or split tiling into `runtime/vae_tiling.py`.) |
| `bundle.py` | Accept `policy`; when `offload`, leave modules on CPU; store policy on `ModelBundle`. |
| `pipelines/text_to_image.py` | Read placement from `model.policy`; wrap conditioning / sampling / decode in `on_device` when offloading; route decode through `tiled_vae_decode` when tiling. |
| `ModelBundle` | New `policy` field (default `None` → current all-resident behavior). |

Keep `model.backbone`/`vae`/etc. as the public attributes; offload only changes
*where* they live between calls, not the bundle's shape.

## Verification plan (the success criteria)

Offload must be **exactly** numerically transparent; tiling is approximate but
must be visually seamless and still seed-deterministic.

1. **Offload determinism (bit-exact).** Generate an SDXL image with `offload=False`
   and with `offload=True`, same seed/prompt/steps. Assert the PNGs are
   **byte-identical**. Moving weights across devices must not change results.
2. **Offload VRAM.** With `offload=True`, assert measured
   `torch.cuda.max_memory_allocated()` at 1024² drops to **≤ ~6 GB** (from 10.7).
   Add a printout to the smoke path or a manual bench script.
3. **Tiled-VAE seam quality.** Decode the *same* final latent with tiled vs
   untiled decode; assert **PSNR > 35 dB** between them (overlap blend should make
   the difference imperceptible). A hard-cut implementation will fail this — that's
   the point of the test.
4. **Tiled-VAE determinism.** Same seed → byte-identical image across two tiled
   runs (tiling itself is deterministic).
5. **8 GB emulation (optional).** `torch.cuda.set_per_process_memory_fraction` to
   cap at ~8 GB and confirm a 1024² SDXL generation completes with offload+tiling.
6. **No SD1.5 regression.** The existing smoke test (offload off) stays
   byte-for-byte unchanged.

A natural home for 1–4 is an opt-in test (skipped without a checkpoint, like
`test_pipeline_smoke.py`), e.g. `tests/test_runtime_vram.py`, gated on CUDA.

## Risks & gotchas

- **`empty_cache()` is needed after moving a module off-GPU**, or the freed blocks
  stay in torch's caching allocator and the "peak" won't actually drop for the
  next stage. Measure with `max_memory_allocated`, but the *reserved* pool matters
  for OOM — call `empty_cache()` at stage boundaries when offloading.
- **The sigma schedule already lives on `device`** (`bundle` moves
  `schedule.sigmas`). Offload must not move *that* to CPU — only the weight-heavy
  `nn.Module`s. Keep schedule + small conditioning tensors resident.
- **VAE stays fp32.** Don't "save memory" by decoding fp16 — it NaNs on many SD1.5
  weights (the reason VAE is fp32 in the first place). Tiling is the fp32-safe way
  to cut decode memory.
- **Tile overlap blend must be in fp32** and clamp to `[-1,1]` after blending, to
  match the existing decode→uint8 path exactly.
- **Don't bake placement into model code.** The whole point of `DevicePolicy` is
  that `models/` never calls `.cuda()`. Keep offload logic in `runtime/` +
  pipeline, not in the backbones.
- **CFG batches two passes.** If a later change batches cond+uncond into one UNet
  call (currently two sequential calls), the UNet activation estimate doubles —
  revisit the ~5.7 GB number then.

## Suggested incremental milestones

Each is independently shippable and verifiable; do them in order.

| # | Slice | Verify | Status |
|---|---|---|---|
| R1 | Plumb `DevicePolicy` through bundle + pipeline (no behavior change; offload still off) | All tests green; SD1.5 + SDXL images byte-identical to today | ✅ |
| R2 | Tiled VAE decode (auto at ≥768 px) | Tiled vs untiled PSNR > 35 dB; SDXL decode peak drops; seed-deterministic | ✅ PSNR 37.55 dB |
| R3 | Sequential CPU offload (`offload=True`) | offload vs no-offload **byte-identical**; SDXL peak ≤ ~6 GB | ✅ byte-identical; peak 6.6 GB (UNet-floor bound) |
| R4 | (optional) `offload="encoders"` cheap mode + 8 GB emulation test | 1024² SDXL completes under an 8 GB cap | ☐ open |

## Pointers

- Current peaks/footprints were measured on the RTX 2060 (see the table above);
  re-measure if weights or batch change.
- Related design context: `ARCHITECTURE.md` §7 (device/memory strategy) and
  `HANDOFF.md` gotchas (VAE fp32, schedule-on-device). Update `ROADMAP.md`'s
  "After SDXL" list when R1–R3 land.
