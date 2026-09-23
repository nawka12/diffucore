"""Device / dtype policy and VRAM techniques.

One place decides where modules live and in what precision, so model code
never hardcodes ``.cuda()`` or a dtype. Also home to sequential CPU offload,
block streaming and tiled VAE decode. See ``docs/RUNTIME_SPEC.md``.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch

from .cond_cache import ConditioningCache

_CPU = torch.device("cpu")


@dataclass
class DevicePolicy:
    """Resolved placement for a run: fp16 weights on CUDA, fp32 VAE and sigma
    math by default. ``vae_tile`` forces tiled decode (otherwise
    :func:`can_decode_untiled` decides per call).

    ``offload`` modes:
      * ``False``: everything resident.
      * ``True`` / ``"full"``: shuttle encoders, backbone and VAE around their
        stages. Lowest peak, but moves the backbone every image.
      * ``"encoders"``: backbone resident; park only the text encoders + VAE.
      * ``"stream"``: like ``"encoders"``, plus stream the backbone's blocks
        (see :func:`stream_blocks`); fits FLUX.1 on 24 GB and SD/SDXL or Anima
        on ~4 GB (ComfyUI's --lowvram analog).
    """

    device: torch.device
    compute_dtype: torch.dtype = torch.float16
    vae_dtype: torch.dtype = torch.float32
    offload: bool | str = False
    vae_tile: bool = False
    # Block streaming (offload == "stream" only): ``stream_blocks_per_group``
    # moves N consecutive blocks per transfer; ``stream_prefetch`` overlaps the
    # next group's copy with compute on a side stream (~2 groups resident).
    stream_blocks_per_group: int = 1
    stream_prefetch: bool = False
    # ``cudnn_benchmark`` is on by default: bit-exact, 3-17% faster. ``tf32`` and
    # ``channels_last`` are opt-in (Ampere+, not bit-exact); ``channels_last``
    # only affects conv backbones.
    cudnn_benchmark: bool = True
    tf32: bool = False
    channels_last: bool = False
    # ``fp16_accumulation``: cuBLAS accumulates fp16 matmuls in fp16 (torch >=
    # 2.7), 2x the tensor-core rate on consumer GPUs. Not bit-exact.
    fp16_accumulation: bool = False
    # ``attention`` (Anima + FLUX): "sdpa" (default, bit-exact), "fa2_turing"
    # (locally built sm75 FlashAttention-2 port; raises at load if unusable) or
    # "auto" (fa2_turing when usable, else sdpa). Not bit-exact; incompatible
    # with ``compile``.
    attention: str = "sdpa"
    # ``compile`` wraps the backbone with torch.compile(dynamic=True) at load
    # (10-60 s warmup). Incompatible with offload modes that move the backbone.
    compile: bool = False
    # ``cuda_graphs`` (requires compile) uses mode="reduce-overhead"; each new
    # input shape pays a warmup.
    cuda_graphs: bool = False

    def __post_init__(self):
        if self.offload not in (False, True, "full", "encoders", "stream"):
            raise ValueError(
                f"offload must be False, True/'full', 'encoders', or 'stream'; "
                f"got {self.offload!r}"
            )
        if self.stream_blocks_per_group < 1:
            raise ValueError(
                f"stream_blocks_per_group must be >= 1; got {self.stream_blocks_per_group}"
            )
        if self.attention not in ("sdpa", "auto", "fa2_turing"):
            raise ValueError(
                f"attention must be 'sdpa', 'auto', or 'fa2_turing'; "
                f"got {self.attention!r}"
            )

    @property
    def offload_device(self) -> torch.device:
        return _CPU

    @property
    def offload_idle(self) -> bool:
        """Park the text encoders + VAE on CPU between stages (every offload mode)."""
        return self.offload is not False

    @property
    def offload_unet(self) -> bool:
        """Also shuttle the whole backbone per image (full offload only)."""
        return self.offload is True or self.offload == "full"

    @property
    def offload_stream(self) -> bool:
        """Stream the backbone's blocks per forward (see ``stream_blocks``)."""
        return self.offload == "stream"

    @classmethod
    def auto(cls) -> "DevicePolicy":
        if torch.cuda.is_available():
            return cls(device=torch.device("cuda"), compute_dtype=torch.float16)
        # CPU fallback (testing only): fp16 is unsupported on most CPUs.
        return cls(device=torch.device("cpu"), compute_dtype=torch.float32)


def maybe_compile_backbone(backbone: torch.nn.Module, policy: "DevicePolicy") -> torch.nn.Module:
    """Wrap a backbone with ``torch.compile`` when ``policy.compile`` is on.

    Raises if the policy also moves the backbone (the compiled artifact
    specializes on the resident device). ``cuda_graphs`` switches to
    ``mode="reduce-overhead", dynamic=False``: near-zero dispatch overhead, but
    each new resolution or LPW chunk count re-records.
    """
    if not policy.compile:
        if policy.cuda_graphs:
            raise ValueError(
                "policy.cuda_graphs=True requires policy.compile=True (CUDA Graphs "
                "are captured by torch.compile's reduce-overhead mode)."
            )
        return backbone
    if policy.offload_unet or policy.offload_stream:
        raise ValueError(
            "policy.compile=True is incompatible with offload modes that move the "
            "backbone on/off the GPU (True/'full'/'stream'); use offload='encoders' "
            "or False."
        )
    if policy.cuda_graphs:
        return torch.compile(backbone, mode="reduce-overhead", dynamic=False)
    return torch.compile(backbone, dynamic=True)


def to_channels_last(module: torch.nn.Module) -> torch.nn.Module:
    """Convert a conv-heavy module (SD UNet, AutoencoderKL) to NHWC in place;
    cuDNN picks faster NHWC kernels on Ampere+ fp16.
    """
    return module.to(memory_format=torch.channels_last)


@contextmanager
def perf_context(policy: "DevicePolicy"):
    """Set cuDNN / matmul backend flags from the policy for the duration of a
    pipeline call, restoring them on exit. ``fp16_accumulation`` is skipped on
    torch < 2.7.
    """
    fp16_acc = policy.fp16_accumulation and hasattr(
        torch.backends.cuda.matmul, "allow_fp16_accumulation"
    )
    if not (policy.cudnn_benchmark or policy.tf32 or fp16_acc):
        yield
        return
    prev_bench = torch.backends.cudnn.benchmark
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_fp16_acc = (
        torch.backends.cuda.matmul.allow_fp16_accumulation if fp16_acc else None
    )
    try:
        if policy.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if policy.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if fp16_acc:
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        yield
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        if fp16_acc:
            torch.backends.cuda.matmul.allow_fp16_accumulation = prev_fp16_acc


@contextmanager
def staged(modules, device, offload):
    """Bring ``modules`` onto ``device`` for the duration when ``offload`` is on,
    parking them on CPU afterward. No-op otherwise."""
    if not offload:
        yield
        return
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(on_device(module, device))
        yield


@contextmanager
def on_device(module, device):
    """Move ``module`` to ``device`` for the duration, then park it on CPU and
    ``empty_cache`` so the next stage sees the lower peak.
    """
    module.to(device)
    try:
        yield module
    finally:
        module.to(_CPU)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _stream_tensors(module):
    """Every owned tensor block streaming relocates: parameters, then persistent
    buffers, de-duplicated by identity."""
    seen = set()
    for t in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if t is not None and id(t) not in seen:
            seen.add(id(t))
            yield t


class _GroupStreamer:
    """Double-buffered group streaming on a side CUDA stream.

    The pinned CPU copy is the permanent home: :meth:`_onload` copies a group
    to the GPU on the side stream, overlapping the current group's compute, and
    :meth:`_offload` re-points each tensor at its CPU master and drops the GPU
    copy. At most two groups are resident.
    """

    def __init__(self, groups, device, offload_device):
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.n = len(groups)
        self.resident = [False] * self.n
        self.copy_done: list = [None] * self.n
        # Pinned masters, so the side-stream H2D copy is truly async.
        self.tensors = []
        for group in groups:
            entries = []
            for module in group:
                for t in _stream_tensors(module):
                    master = t.data.to(offload_device).pin_memory()
                    t.data = master
                    entries.append((t, master))
            self.tensors.append(entries)

    def _onload(self, gi):
        if self.resident[gi]:
            return
        with torch.cuda.stream(self.stream):
            for t, master in self.tensors[gi]:
                t.data = master.to(self.device, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        self.copy_done[gi] = ev
        self.resident[gi] = True

    def _offload(self, gi):
        if not self.resident[gi]:
            return
        cur = torch.cuda.current_stream(self.device)
        for t, master in self.tensors[gi]:
            # The GPU copy was allocated on the side stream; keep it alive until
            # the compute stream's kernels that read it have run.
            t.data.record_stream(cur)
            t.data = master
        self.resident[gi] = False
        self.copy_done[gi] = None

    def pre(self, gi):
        self._onload(gi)              # usually a no-op: already prefetched
        torch.cuda.current_stream(self.device).wait_event(self.copy_done[gi])
        if gi + 1 < self.n:
            self._onload(gi + 1)      # prefetch the next group

    def post(self, gi):
        self._offload(gi)


def stream_blocks(backbone, block_attrs, device, offload_device=_CPU, *, keep_resident=(),
                  num_blocks_per_group=1, prefetch=False):
    """Grouped sequential offload for a block-list backbone (FLUX DiT, SD/SDXL
    UNet, Anima DiT).

    Non-block submodules stay resident on ``device``; the blocks live on
    ``offload_device`` and forward hooks bring each group onto ``device`` for
    its own forward. Resident VRAM is the small modules plus one group (two with
    ``prefetch``). ``block_attrs`` names ``nn.ModuleList`` / ``nn.Sequential``
    attributes to stream. ``keep_resident`` blocks are read outside their own
    forward (Anima's block 0, probed by TeaCache) and stay put.
    ``num_blocks_per_group`` groups consecutive blocks within one attr.
    Returns the number of streamed blocks.
    """
    keep = {id(m) for m in keep_resident}
    for name, child in backbone.named_children():
        if name not in block_attrs:
            child.to(device)

    # Contiguous groups within each attr; kept-resident blocks are excluded.
    groups, n = [], 0
    for attr in block_attrs:
        current = []
        for block in getattr(backbone, attr):
            if id(block) in keep:
                block.to(device)
                continue
            block.to(offload_device)
            current.append(block)
            n += 1
            if len(current) == num_blocks_per_group:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

    if prefetch and device.type == "cuda":
        streamer = _GroupStreamer(groups, device, offload_device)
        backbone._block_streamer = streamer  # keep the side stream / masters alive
        for gi, group in enumerate(groups):
            group[0].register_forward_pre_hook(lambda m, i, gi=gi: streamer.pre(gi))
            group[-1].register_forward_hook(lambda m, i, o, gi=gi: streamer.post(gi))
        return n

    for group in groups:
        def pre(mod, _inp, group=group):
            for m in group:
                m.to(device, non_blocking=True)

        def post(mod, _inp, _out, group=group):
            for m in group:
                m.to(offload_device, non_blocking=True)

        group[0].register_forward_pre_hook(pre)
        group[-1].register_forward_hook(post)
    return n


def _ramp(n: int, edge: int) -> torch.Tensor:
    """A 1-D feather ramping up over ``edge`` samples, flat at 1, ramping down.
    Strictly positive, so the normalize never divides by zero."""
    r = torch.ones(n, dtype=torch.float32)
    e = min(edge, (n + 1) // 2)
    if e > 0:
        up = torch.arange(1, e + 1, dtype=torch.float32) / (e + 1)
        r[:e] = up
        r[n - e:] = up.flip(0)
    return r


def tiled_vae_decode(vae, latent: torch.Tensor, tile: int = 64, overlap: int = 16) -> torch.Tensor:
    """Decode ``latent`` in overlapping tiles blended with a linear fp32
    feather, bounding activation memory to ~one tile. ``tile``/``overlap`` are
    in latent pixels. Returns ``[B, 3, 8h, 8w]`` fp32; a latent that fits one
    tile decodes untiled, bit-identical to ``vae.decode``.
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
    """Tile starts covering ``size`` with stride ``step``; the last tile is flush
    with the edge."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, step))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


# Per-VAE decode activation cost in bytes per output pixel, with headroom,
# calibrated on an RTX 2060: AutoencoderKL ~3.5 GB at 1024² (3.3 KB/px raw,
# 6 KB/px set); QwenImageVAE ~2.25 GB at 1024×1536 (1.5 KB/px raw, 3.5 KB/px
# set). Unknown classes use the conservative SDXL value.
_VAE_BYTES_PER_PX = {
    "AutoencoderKL": 6 * 1024,
    "QwenImageVAE": 3584,
}
_VAE_BYTES_PER_PX_DEFAULT = 6 * 1024
_DECODE_FREE_VRAM_MARGIN = 0.85


def can_decode_untiled(
    vae: torch.nn.Module,
    latent_shape: tuple[int, int, int, int],
    device: torch.device,
    *,
    free_bytes: int | None = None,
) -> bool:
    """Whether ``vae.decode(latent)`` should fit on ``device``: always on CPU;
    on CUDA, the per-pixel estimate (:data:`_VAE_BYTES_PER_PX`) against free
    VRAM with a safety margin. Call after staging the VAE. ``free_bytes``
    overrides the reading for tests.
    """
    measured = free_bytes is None  # real GPU read (vs a test budget)
    if free_bytes is None:
        if device.type != "cuda":
            return True
        # Release the caching allocator's cached blocks first: mem_get_info
        # counts them as used, which would trip fine decodes into tiling.
        torch.cuda.empty_cache()
        # mem_get_info needs an index; a bare torch.device("cuda") raises.
        index = device.index if device.index is not None else torch.cuda.current_device()
        free_bytes, _ = torch.cuda.mem_get_info(index)
    _, _, h_lat, w_lat = latent_shape
    px = (w_lat * 8) * (h_lat * 8)
    bytes_per_px = _VAE_BYTES_PER_PX.get(type(vae).__name__, _VAE_BYTES_PER_PX_DEFAULT)
    # The constants were calibrated on fp32 decodes; fp16 halves them.
    param = next(vae.parameters(), None)
    if param is not None and param.dtype == torch.float16:
        bytes_per_px //= 2
    need = px * bytes_per_px
    budget = free_bytes * _DECODE_FREE_VRAM_MARGIN
    fits = need < budget
    if measured:
        _GB = 1024**3
        print(
            f"[vae-decode] {type(vae).__name__} {w_lat * 8}x{h_lat * 8} | "
            f"need {need / _GB:.2f} GB ({bytes_per_px / 1024:.1f} KB/px) | "
            f"free {free_bytes / _GB:.2f} GB → budget {budget / _GB:.2f} GB "
            f"(×{_DECODE_FREE_VRAM_MARGIN:g}) | {'untiled' if fits else 'TILED'}",
            flush=True,
        )
    return fits


def vae_fallback_to_fp32(vae: torch.nn.Module, policy: "DevicePolicy") -> None:
    """Permanently drop an fp16 VAE to fp32 after a non-finite output. Which
    checkpoints overflow fp16 isn't knowable up front (A1111's
    ``--no-half-vae``), so the failure costs one pass on one image.
    """
    print("[vae] fp16 output was not finite; falling back to fp32 for this model", flush=True)
    vae.float()
    policy.vae_dtype = torch.float32


def vae_decode_safe(vae, latent: torch.Tensor, policy: "DevicePolicy"):
    """Decode ``latent`` (already in the VAE's dtype), tiled when forced or when
    free VRAM is short, retrying in fp32 on a non-finite fp16 result. Returns
    ``(image, "tiled" | "untiled")``. Call with the VAE staged on the device.
    """
    def _decode(z):
        tile = policy.vae_tile or not can_decode_untiled(vae, z.shape, policy.device)
        return (tiled_vae_decode(vae, z) if tile else vae.decode(z)), tile

    image, tile = _decode(latent)
    if latent.dtype == torch.float16 and not torch.isfinite(image).all():
        vae_fallback_to_fp32(vae, policy)
        image, tile = _decode(latent.float())
    return image, ("tiled" if tile else "untiled")


__all__ = ["ConditioningCache", "DevicePolicy", "can_decode_untiled", "maybe_compile_backbone", "on_device", "perf_context", "staged", "stream_blocks", "tiled_vae_decode", "to_channels_last", "vae_decode_safe", "vae_fallback_to_fp32"]
