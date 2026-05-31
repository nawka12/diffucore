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
    vae_tile: bool = False           # force tiled VAE decode (else auto via free-VRAM check)

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
- Trigger automatically when `vae_tile` is set, or when `can_decode_untiled` (a
  free-VRAM check using `torch.cuda.mem_get_info` against per-family activation
  estimates) reports that an untiled decode won't fit at decode time.

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
| R2 | Tiled VAE decode (auto from a free-VRAM check via `can_decode_untiled`; original 768 px threshold superseded) | Tiled vs untiled PSNR > 35 dB; SDXL decode peak drops; seed-deterministic | ✅ PSNR 37.55 dB |
| R3 | Sequential CPU offload (`offload=True`) | offload vs no-offload **byte-identical**; SDXL peak ≤ ~6 GB | ✅ byte-identical; peak 6.6 GB (UNet-floor bound) |
| R4 | `offload="encoders"` cheap mode + 8 GB emulation test | 1024² SDXL completes under an 8 GB cap | ✅ byte-identical (both modes); fits 8 GB cap |

**R4 notes.** `offload` now accepts `"encoders"` alongside `False`/`True` (`True`
== `"full"`). `DevicePolicy` splits placement into two groups via `offload_idle`
(text encoders + VAE — parked in every mode) and `offload_unet` (the ~5 GB UNet —
parked only in full offload); `bundle.py` and the pipeline read those per stage.
`"encoders"` keeps the UNet resident (saving the ~5 GB copy each image) and only
shuttles the ~2 GB of encoders + VAE — 1024² peak ~7 GB (UNet resident + tiled
decode), which clears the 8 GB cap. **Verified on the RTX 2060** against
`AkashicPulse-v3.0`: `test_offload_is_byte_identical` (parametrized over `True`
and `"encoders"`) — both byte-identical to resident; full-offload peak ≤ 7 GB;
`test_encoders_offload_fits_8gb_cap` completes a 1024² generation under an 8 GB
allocator cap (`set_per_process_memory_fraction`). Mode-flag + invalid-mode unit
tests run on CPU. Each SDXL test loads **one** pipeline at a time and frees it
(`del` + `gc` + `empty_cache`) before the next — holding two resident copies
overflows the 12 GB card.

## Perf flags (PR-A + PR-B + PR-C)

> **Status:** PR-A (cuDNN benchmark + TF32 + channels_last), PR-B (torch.compile),
> and PR-C (CUDA Graphs via `mode="reduce-overhead"`) implemented and validated
> on the RTX 2060 against SD1.5, SDXL (AkashicPulse-v3.0), and Anima
> (anima-base-v1.0). Headline: Anima 52.2 s → 35.6 s (**1.47×**) with
> `cudnn_benchmark + compile + cuda_graphs`; SDXL 19.8 s → 16.9 s (1.17×) with
> `cudnn_benchmark` alone (bit-exact).

Five flags on `DevicePolicy`. `cudnn_benchmark` defaults **on** (bit-exact,
free win, measured 3-17 % across the lineup). The other four default off and
opt in only:

| Flag | Default | What it does | Cost | Bit-exact? |
|---|---|---|---|---|
| `cudnn_benchmark` | **True** | Enables cuDNN's per-shape kernel autotune for the run | one-step autotune on first call | yes (kernel choice doesn't change values) |
| `tf32` | False | TF32 matmul + cuDNN on Ampere+ for **fp32 paths only** | tiny precision loss in fp32 ops | no (~1e-3 relative) |
| `channels_last` | False | Converts SD UNet + AutoencoderKL to NHWC | one layout reorder at load + per-step input reorder | within fp16 tolerance (different kernel paths) |
| `compile` | False | Wraps the backbone with `torch.compile(dynamic=True)` | one-time warmup ~30-180 s, paid at load | within fp16 tolerance (Inductor codegen) |
| `cuda_graphs` | False | Switches compile to `mode="reduce-overhead", dynamic=False` — Inductor captures a CUDA Graph and replays it each step | re-records on any shape change (resolution, LPW chunk count) | within fp16 tolerance — *more* deterministic than compile alone (54.7 dB vs 29 dB on Anima) |

### Mechanism

`runtime.perf_context(policy)` is a context manager wrapping each pipeline call;
it flips `torch.backends.cudnn.benchmark` / `cuda.matmul.allow_tf32` /
`cudnn.allow_tf32` for the run and restores the previous values on exit. A
**no-op** when both flags are off — the global state is read but never written,
so the bit-exact path is preserved.

`runtime.to_channels_last(module)` calls `.to(memory_format=torch.channels_last)`
on the module. Applied to the SD/SDXL UNet and AutoencoderKL at bundle load.
Latents passed into the conv backbones are reordered to NHWC in
`_base._sample` / `_decode` / `_encode_image`.

`runtime.maybe_compile_backbone(backbone, policy)` returns the eager module
when `compile=False`. With `compile=True` alone it returns
`torch.compile(backbone, dynamic=True)`. With `compile=True, cuda_graphs=True`
it returns `torch.compile(backbone, mode="reduce-overhead", dynamic=False)` —
Inductor captures the per-step forward into a CUDA Graph that the sampling
loop replays each step, eliminating nearly all Python/dispatcher overhead.
The **Anima DiT** and **SD/SDXL UNet** are the targets; the VAE is
intentionally not compiled (single decode call per image — not worth the
warmup). `dynamic=True` (compile-only mode) lets LPW-produced variable-length
contexts share one graph; `dynamic=False` (CUDA-Graphs mode) re-records on
each new shape.

### Compose rules

- `compile=True` + `offload=True/"full"` → **raises `ValueError` at load**. The
  compiled artifact specializes on the resident device; round-tripping it
  through CPU each image defeats the point and can crash.
- `compile=True` + `offload="encoders"` → **allowed**. The UNet stays resident
  in encoders mode; only the text encoders + VAE move.
- `compile=True` + `apply_lora(...)` → **works**. The LoRA fuse path unwraps
  `torch.compile`'s `OptimizedModule._orig_mod` to walk the original kohya
  paths; weight values are mutated in place and the compiled graph picks them
  up at the next call (Inductor specializes on shape, not value).
- `cuda_graphs=True` without `compile=True` → **raises `ValueError` at load**.
  CUDA Graphs are captured by `torch.compile`'s reduce-overhead mode, not
  separately.
- `cuda_graphs=True` + LPW prompts of varying token chunk counts → graph
  re-records per chunk count. Acceptable: first image at each length pays the
  warmup, subsequent images at the same length replay fast.
- `cuda_graphs=True` on **Anima above 1024²** → **VRAM blow-up on 12 GB cards**.
  Anima's CFG runs cond/uncond as two separate forwards with different
  conditioning shapes, so each step captures two graph pools. Per-pool
  activation memory scales as O(tokens²) (DiT self-attention), and the Anima
  tokenizer doesn't pad — so at 1024×1536 each pool is roughly 2× the size of
  the 1024² pool. The two resident pools + ~4 GB resident DiT exceed 12 GB.
  At 1024² the combo fits and gives the 1.47× speedup (see benchmark below);
  above that, use `compile=True` alone (no `cuda_graphs`) — eager activations
  are transient, so the higher resolution still fits.

### Measured on RTX 2060 (Turing sm_75), fp16, 20 steps

```
SD 1.5 (512²)               SDXL (1024²)                Anima (1024², er_sde)
case            time speedup case            time speedup case            time speedup
baseline       3.01s 1.00x   baseline      19.84s 1.00x   baseline      52.20s 1.00x
+cudnn         2.83s 1.06x   +cudnn        16.89s 1.17x   +cudnn        46.33s 1.13x
+tf32          2.96s 1.02x   +tf32         16.91s 1.17x   +tf32         47.72s 1.09x
+channels_last 2.79s 1.08x   +channels_last 18.77s 1.06x  (skip – DiT)
+compile       2.78s 1.08x   +compile      18.02s 1.10x   +compile      36.99s 1.41x
                                                          +cuda_graphs  35.55s 1.47x
```

Findings:

- **`cudnn_benchmark` is the universal win** — 3-17 %, bit-exact, free.
- **`compile` lands big on Anima** (33 %), modest on SD1.5 (8 %), and slightly
  *regresses* on SDXL on Turing. The Anima DiT is 28 transformer blocks where
  Inductor has the most room to fuse; the SDXL UNet's conv-heavy path already
  hits cuDNN's tuned kernels.
- **`channels_last` is hardware-sensitive.** On Turing, NHWC fp16 conv kernels
  aren't always faster than NCHW (cuDNN's autotune already picks well). On
  Ampere+ (RTX 30/40-series, A100) NHWC + tensor cores typically give a clear
  win. Recommend gating in docs by GPU generation.
- **`tf32` is a no-op for fp16 backbones.** Only the fp32 VAE benefits, and
  that's one call per image — buried in noise.

### Recommendations by model + GPU

| Stack | Recommended |
|---|---|
| Anima at 1024² on 12 GB | `cudnn_benchmark=True, compile=True, cuda_graphs=True` |
| Anima above 1024² (e.g. 1024×1536) on 12 GB | `cudnn_benchmark=True, compile=True` (skip `cuda_graphs` — two cond/uncond pools × O(tokens²) per pool OOMs above 1024², see Compose rules) |
| Anima, varying resolutions / heavy LPW | `cudnn_benchmark=True, compile=True` (skip `cuda_graphs` to avoid per-shape re-records and pool growth) |
| SDXL on Turing (RTX 20-series) | `cudnn_benchmark=True` only |
| SDXL on Ampere+ (RTX 30/40-series) | `cudnn_benchmark=True, channels_last=True, compile=True` (validate locally; try cuda_graphs=True if shapes are stable) |
| SD1.5, any GPU | flags optional — small absolute speedup |

A power-user `offload="encoders" + compile=True` combo fits SDXL on 8 GB with
the UNet compiled and resident.

## Pointers

- Current peaks/footprints were measured on the RTX 2060 (see the table above);
  re-measure if weights or batch change.
- Related design context: `ARCHITECTURE.md` §7 (device/memory strategy) and
  `HANDOFF.md` gotchas (VAE fp32, schedule-on-device). Update `ROADMAP.md`'s
  "After SDXL" list when R1–R3 land.
