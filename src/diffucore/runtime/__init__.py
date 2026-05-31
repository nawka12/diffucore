"""Device / dtype policy + VRAM techniques.

A single place that decides where modules live and in what precision, so model
code never hardcodes ``.cuda()`` or a dtype. Also home to the two opt-in memory
techniques that read the policy: sequential CPU offload (``on_device``) and
tiled VAE decode (``tiled_vae_decode``). See ``docs/RUNTIME_SPEC.md``.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch

_CPU = torch.device("cpu")


@dataclass
class DevicePolicy:
    """Resolved placement for a run.

    Defaults target the RTX 2060: fp16 weights on CUDA, fp32 for the VAE and the
    sigma math. ``offload`` moves idle submodules to CPU RAM between stages;
    ``vae_tile`` decodes the VAE in tiles (also auto-triggered above
    ``vae_tile_threshold`` px). All knobs default off → unchanged behavior.

    ``offload`` modes:
      * ``False`` — everything resident (no offload).
      * ``True`` / ``"full"`` — shuttle every weight-heavy module (encoders, UNet,
        VAE) on/off the GPU around its stage. Lowest peak (~UNet stage), at the
        cost of moving the 5 GB UNet each image.
      * ``"encoders"`` — keep the UNet resident; only park the ~2 GB of text
        encoders + VAE between stages. Frees most of the headroom for ~0 copy cost
        (the cheap 80/20 — see ``docs/RUNTIME_SPEC.md`` R4).
    """

    device: torch.device
    compute_dtype: torch.dtype = torch.float16
    vae_dtype: torch.dtype = torch.float32
    offload: bool | str = False
    vae_tile: bool = False
    vae_tile_threshold: int = 768
    # --- opt-in perf flags (PR-A). ``cudnn_benchmark`` defaults **on** because
    # it's bit-exact and gives 3-17 % across SD1.5/SDXL/Anima (measured RTX 2060)
    # for the cost of one autotune step on first call; set False to disable.
    # ``tf32`` and ``channels_last`` default off — they're hardware-sensitive
    # (Ampere+) and not bit-exact, so opt-in only. ``channels_last`` converts
    # conv backbones (SD UNet, AutoencoderKL) to NHWC; transformer-only modules
    # (Anima DiT) and 3D-conv modules (Qwen-Image VAE) are unaffected.
    cudnn_benchmark: bool = True
    tf32: bool = False
    channels_last: bool = False
    # --- opt-in compile (PR-B). When True, the backbone (SD/SDXL UNet or Anima
    # DiT) is wrapped with ``torch.compile(dynamic=True)`` at load. Pays a one-
    # time warmup on the first call (10-60s depending on platform), persisted
    # for the rest of the process. Incompatible with offload modes that move
    # the UNet on/off the GPU each image (the compiled artifact specializes on
    # the resident device); rejected with a clear error at load.
    compile: bool = False
    # --- opt-in CUDA Graphs (PR-C). When True (requires compile=True), the
    # backbone is compiled with ``mode="reduce-overhead"`` so Inductor captures
    # the per-step forward into a CUDA Graph and replays it each step. Near-zero
    # Python/dispatcher overhead on top of PR-B. Recompiles on any input shape
    # change (resolution, LPW chunk count), so the first image at each new
    # shape pays a warmup; subsequent images at the same shape are fast.
    cuda_graphs: bool = False

    def __post_init__(self):
        if self.offload not in (False, True, "full", "encoders"):
            raise ValueError(
                f"offload must be False, True/'full', or 'encoders'; got {self.offload!r}"
            )

    @property
    def offload_device(self) -> torch.device:
        return _CPU

    @property
    def offload_idle(self) -> bool:
        """Park the text encoders + VAE on CPU between stages. On in every offload
        mode (``True``/``"full"`` and ``"encoders"``)."""
        return self.offload is not False

    @property
    def offload_unet(self) -> bool:
        """Also shuttle the UNet (the ~5 GB module) per image — full offload only.
        ``"encoders"`` keeps it resident; the copy each way isn't worth the ~2 GB."""
        return self.offload is True or self.offload == "full"

    @classmethod
    def auto(cls) -> "DevicePolicy":
        if torch.cuda.is_available():
            return cls(device=torch.device("cuda"), compute_dtype=torch.float16)
        # CPU fallback (testing only): fp16 is unsupported on most CPUs.
        return cls(device=torch.device("cpu"), compute_dtype=torch.float32)


def maybe_compile_backbone(backbone: torch.nn.Module, policy: "DevicePolicy") -> torch.nn.Module:
    """Wrap a backbone with ``torch.compile`` when ``policy.compile`` is on.

    Returns the module unchanged when the flag is off (default). When on, raises
    if the policy also offloads the backbone (the compiled artifact specializes
    on the resident device — round-tripping it through CPU each image defeats
    the point and can crash).

    Two compile modes:
      * ``compile=True`` alone — ``torch.compile(dynamic=True)``. Inductor codegen
        with shape-flexible graphs (LPW, varying resolutions just re-trace
        cheaply). One warmup at load, fast steps thereafter.
      * ``compile=True, cuda_graphs=True`` — ``torch.compile(mode="reduce-overhead",
        dynamic=False)``. Inductor + CUDA Graphs: the per-step forward is
        captured once and replayed, removing nearly all Python/dispatcher
        overhead. Re-records on any input shape change, so the first image at
        each new resolution / LPW chunk count pays a warmup.
    """
    if not policy.compile:
        if policy.cuda_graphs:
            raise ValueError(
                "policy.cuda_graphs=True requires policy.compile=True (CUDA Graphs "
                "are captured by torch.compile's reduce-overhead mode)."
            )
        return backbone
    if policy.offload_unet:
        raise ValueError(
            "policy.compile=True is incompatible with offload modes that move the "
            "backbone on/off the GPU (True/'full'); use offload='encoders' or False."
        )
    if policy.cuda_graphs:
        return torch.compile(backbone, mode="reduce-overhead", dynamic=False)
    return torch.compile(backbone, dynamic=True)


def to_channels_last(module: torch.nn.Module) -> torch.nn.Module:
    """Convert a conv-heavy module's parameters to NHWC memory format in place.
    Intended for the SD UNet and AutoencoderKL — on Ampere+ fp16, cuDNN picks
    faster NHWC conv kernels and skips the implicit layout transpose. No-op
    semantically (output values unchanged within fp16 tolerance); skip for
    transformer-only modules (Anima DiT) where there's no conv to benefit.
    """
    return module.to(memory_format=torch.channels_last)


@contextmanager
def perf_context(policy: "DevicePolicy"):
    """Flip cuDNN / matmul backend flags for the duration of a pipeline call.

    Reads ``policy.cudnn_benchmark`` and ``policy.tf32`` and toggles the
    corresponding ``torch.backends`` flags, restoring the previous values on
    exit. A no-op when both flags are False (the default) — the global state
    is read but not written, so the existing bit-exact path is unchanged.

    ``cudnn_benchmark`` lets cuDNN pick the fastest conv kernel for each input
    shape; helpful for the SD/SDXL UNet (fixed shapes per run, so the autotune
    cost is paid once). ``tf32`` enables TF32 on Ampere+ for fp32 matmul and
    cuDNN — affects only fp32 paths (the VAE), fp16 weights are unchanged.
    """
    if not (policy.cudnn_benchmark or policy.tf32):
        yield
        return
    prev_bench = torch.backends.cudnn.benchmark
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        if policy.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if policy.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        yield
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


@contextmanager
def staged(modules, device, offload):
    """Bring ``modules`` onto ``device`` for the duration when ``offload`` is on,
    parking them back on CPU afterward. A no-op when offload is off (modules are
    already resident — never touch them)."""
    if not offload:
        yield
        return
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(on_device(module, device))
        yield


@contextmanager
def on_device(module, device):
    """Move ``module`` to ``device`` for the duration, then park it back on CPU.

    The primitive for sequential offload: bring only the active stage's module
    onto the GPU, then free it. ``empty_cache`` hands the freed blocks back to
    the driver so the *next* stage actually sees the lower peak (otherwise they
    sit in torch's caching allocator). Numerically transparent — moving weights
    across devices does not change results.
    """
    module.to(device)
    try:
        yield module
    finally:
        module.to(_CPU)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _ramp(n: int, edge: int) -> torch.Tensor:
    """A 1-D feather: ramps up over the first ``edge`` samples, plateaus at 1,
    ramps down over the last ``edge``. Strictly positive so the blend's
    accumulate-then-normalize never divides by zero."""
    r = torch.ones(n, dtype=torch.float32)
    e = min(edge, (n + 1) // 2)
    if e > 0:
        up = torch.arange(1, e + 1, dtype=torch.float32) / (e + 1)
        r[:e] = up
        r[n - e:] = up.flip(0)
    return r


def tiled_vae_decode(vae, latent: torch.Tensor, tile: int = 64, overlap: int = 16) -> torch.Tensor:
    """Decode ``latent`` in overlapping spatial tiles and blend the overlaps.

    Bounds decode activation memory to ~one tile regardless of output size. The
    decoder is convolutional, so each tile is decoded independently and stitched.
    ``tile``/``overlap`` are in latent pixels (output is 8x). Overlaps are blended
    with a linear feather in fp32 so seams vanish (a hard cut leaves grid lines).
    Returns the full image ``[B, 3, 8h, 8w]`` in fp32, matching ``vae.decode``.

    When the latent already fits one tile, returns ``vae.decode`` unchanged
    (bit-identical) — no blend math runs.
    """
    _, _, height, width = latent.shape
    if height <= tile and width <= tile:
        return vae.decode(latent)

    step = tile - overlap
    ys = _tile_starts(height, tile, step)
    xs = _tile_starts(width, tile, step)

    out = weight = None
    for y in ys:
        for x in xs:
            dec = vae.decode(latent[:, :, y:y + tile, x:x + tile]).float()
            if out is None:
                scale = dec.shape[-1] // min(tile, width)
                out = latent.new_zeros((dec.shape[0], dec.shape[1], height * scale, width * scale), dtype=torch.float32)
                weight = latent.new_zeros((1, 1, height * scale, width * scale), dtype=torch.float32)
            th, tw = dec.shape[-2], dec.shape[-1]
            oy, ox = y * scale, x * scale
            w2d = (_ramp(th, overlap * scale)[:, None] * _ramp(tw, overlap * scale)[None, :]).to(dec.device)
            out[:, :, oy:oy + th, ox:ox + tw] += dec * w2d
            weight[:, :, oy:oy + th, ox:ox + tw] += w2d
    return out / weight


def _tile_starts(size: int, tile: int, step: int) -> list[int]:
    """Tile start offsets covering ``size`` with stride ``step``; the last tile is
    snapped flush to the edge so every tile is exactly ``tile`` wide."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, step))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


__all__ = ["DevicePolicy", "maybe_compile_backbone", "on_device", "perf_context", "staged", "tiled_vae_decode", "to_channels_last"]
