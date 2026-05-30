"""Device / dtype policy + VRAM techniques.

A single place that decides where modules live and in what precision, so model
code never hardcodes ``.cuda()`` or a dtype. Also home to the two opt-in memory
techniques that read the policy: sequential CPU offload (``on_device``) and
tiled VAE decode (``tiled_vae_decode``). See ``docs/RUNTIME_SPEC.md``.
"""

from __future__ import annotations

from contextlib import contextmanager
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
    """

    device: torch.device
    compute_dtype: torch.dtype = torch.float16
    vae_dtype: torch.dtype = torch.float32
    offload: bool = False
    vae_tile: bool = False
    vae_tile_threshold: int = 768

    @property
    def offload_device(self) -> torch.device:
        return _CPU

    @classmethod
    def auto(cls) -> "DevicePolicy":
        if torch.cuda.is_available():
            return cls(device=torch.device("cuda"), compute_dtype=torch.float16)
        # CPU fallback (testing only): fp16 is unsupported on most CPUs.
        return cls(device=torch.device("cpu"), compute_dtype=torch.float32)


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


__all__ = ["DevicePolicy", "on_device", "tiled_vae_decode"]
