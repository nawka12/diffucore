# SPDX-FileCopyrightText: Copyright (c) 2026 Dzakwan Haq (KayfaHaarukku, github.com/nawka12)
# SPDX-License-Identifier: Apache-2.0

"""Fused elementwise glue for the Anima DiT blocks (Triton).

Eager, each block stage re-reads the fp32 residual stream ~11 times (LayerNorm,
``*(1+scale)``, ``+shift``, casts, ``gate*h``, residual add) and every q/k
RMSNorm + RoPE another ~8; on an RTX 2060 at 1024x1536 that glue was 39% of a
forward. These kernels do each pattern in one read and one write per token row
(×1.38 per forward there). Cast points match eager and FMA contraction is off,
so outputs differ from eager by at most 1 fp16 ulp on ~0.01% of elements: a
whole forward then moves by the same amount as nudging 0.01% of the input
latent by 1 ulp.

Triton specializes nothing on the sequence length, so a resolution change
never recompiles; the first use per process loads the kernels from Triton's
disk cache (a few seconds the very first time). Any launch failure falls back
to eager for the rest of the process.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  no Triton (e.g. a Windows install): always eager
    triton = None

_broken = False
_NO_FMA = {"enable_fp_fusion": False}


def fused_glue_available(device=None) -> bool:
    """Whether the fused kernels can run here: Triton importable and CUDA."""
    if triton is None or _broken or not torch.cuda.is_available():
        return False
    return not (device is not None and getattr(device, "type", None) == "cpu")


def set_fused_glue(module: torch.nn.Module, on: bool) -> int:
    """Stamp ``on`` on every submodule with a ``fused_glue`` slot (the Anima
    blocks and attention modules). Returns the number stamped."""
    n = 0
    for m in module.modules():
        if hasattr(m, "fused_glue"):
            m.fused_glue = on
            n += 1
    return n


def usable(*ts: torch.Tensor) -> bool:
    return (triton is not None and not _broken and all(t.is_cuda for t in ts)
            and not torch.compiler.is_compiling())


def _launch(kernel, grid, *args, **kw):
    """Run a kernel; on any failure switch the process to eager and return
    False so the caller recomputes eagerly."""
    global _broken
    try:
        kernel[grid](*args, **kw, **_NO_FMA)
        return True
    except Exception as e:  # noqa: BLE001  any compile/launch failure: eager from here on
        print(f"[fused] Triton kernels unavailable ({type(e).__name__}: {e}); using eager glue", flush=True)
        _broken = True
        return False


if triton is not None:

    @triton.jit(do_not_specialize=["rows_per_mod"])
    def _ln_mod_kernel(x_ptr, sc1_ptr, s_ptr, out_ptr, rows_per_mod, eps, D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, D)
        x = tl.load(x_ptr + row * D + offs)
        mean = tl.sum(x, 0) / D
        xc = x - mean
        y = xc * tl.math.rsqrt(tl.sum(xc * xc, 0) / D + eps)
        m = row // rows_per_mod
        sc1 = tl.load(sc1_ptr + m * D + offs).to(tl.float32)
        s = tl.load(s_ptr + m * D + offs).to(tl.float32)
        tl.store(out_ptr + row * D + offs, (y * sc1 + s).to(tl.float16))

    @triton.jit(do_not_specialize=["rows_per_mod"])
    def _res_ln_mod_kernel(x_ptr, h_ptr, g_ptr, sc1_ptr, s_ptr, xout_ptr, out_ptr, rows_per_mod, eps,
                           D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, D)
        m = row // rows_per_mod
        x = tl.load(x_ptr + row * D + offs)
        h = tl.load(h_ptr + row * D + offs).to(tl.float32)
        g = tl.load(g_ptr + m * D + offs).to(tl.float32)
        x = x + g * h
        tl.store(xout_ptr + row * D + offs, x)
        mean = tl.sum(x, 0) / D
        xc = x - mean
        y = xc * tl.math.rsqrt(tl.sum(xc * xc, 0) / D + eps)
        sc1 = tl.load(sc1_ptr + m * D + offs).to(tl.float32)
        s = tl.load(s_ptr + m * D + offs).to(tl.float32)
        tl.store(out_ptr + row * D + offs, (y * sc1 + s).to(tl.float16))

    @triton.jit(do_not_specialize=["rows_per_mod"])
    def _res_kernel(x_ptr, h_ptr, g_ptr, xout_ptr, rows_per_mod, D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, D)
        m = row // rows_per_mod
        x = tl.load(x_ptr + row * D + offs)
        h = tl.load(h_ptr + row * D + offs).to(tl.float32)
        g = tl.load(g_ptr + m * D + offs).to(tl.float32)
        tl.store(xout_ptr + row * D + offs, x + g * h)

    @triton.jit(do_not_specialize=["seq_len", "row_stride"])
    def _qk_norm_rope_kernel(x_ptr, w_ptr, f_ptr, out_ptr, seq_len, row_stride, eps,
                             H: tl.constexpr, HD: tl.constexpr, ROPE: tl.constexpr):
        # One token per program, all heads at once, so the RoPE table row is
        # read once. RoPE rotates channel pairs (i, i + HD/2).
        row = tl.program_id(0).to(tl.int64)
        HALF: tl.constexpr = HD // 2
        hh = tl.arange(0, H)[:, None]
        dd = tl.arange(0, HALF)[None, :]
        src = x_ptr + row * row_stride + hh * HD + dd
        xa = tl.load(src).to(tl.float32)
        xb = tl.load(src + HALF).to(tl.float32)
        r = tl.math.rsqrt((tl.sum(xa * xa, 1) + tl.sum(xb * xb, 1)) / HD + eps)[:, None]
        wa = tl.load(w_ptr + dd).to(tl.float32)
        wb = tl.load(w_ptr + HALF + dd).to(tl.float32)
        na = (wa * (xa * r)).to(tl.float16)
        nb = (wb * (xb * r)).to(tl.float16)
        if ROPE:
            fb = f_ptr + (row % seq_len) * (HALF * 4) + dd * 4
            a = na.to(tl.float32)
            b = nb.to(tl.float32)
            na = (tl.load(fb) * a + tl.load(fb + 1) * b).to(tl.float16)
            nb = (tl.load(fb + 2) * a + tl.load(fb + 3) * b).to(tl.float16)
        dst = out_ptr + row * (H * HD) + hh * HD + dd
        tl.store(dst, na)
        tl.store(dst + HALF, nb)


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def rows_ok(x2: torch.Tensor) -> bool:
    """Row kernels hold a whole (power-of-two) feature row in one program."""
    return x2.dtype == torch.float32 and x2.is_contiguous() and _pow2(x2.shape[1])


def ln_mod(x2, sc1, s, rows_per_mod, eps):
    """fp16((LayerNorm(x) * sc1 + s)) per row; ``sc1`` is the fp16 ``1 + scale``."""
    out = torch.empty(x2.shape, device=x2.device, dtype=torch.float16)
    ok = _launch(_ln_mod_kernel, (x2.shape[0],), x2, sc1, s, out, rows_per_mod, eps,
                 D=x2.shape[1], num_warps=8)
    return out if ok else None


def res_ln_mod(x2, h, g, sc1, s, rows_per_mod, eps, xout):
    """``xout = x + g*h`` (fp32), then the next stage's LayerNorm-modulate."""
    out = torch.empty(x2.shape, device=x2.device, dtype=torch.float16)
    ok = _launch(_res_ln_mod_kernel, (x2.shape[0],), x2, h, g, sc1, s, xout, out, rows_per_mod, eps,
                 D=x2.shape[1], num_warps=8)
    return out if ok else None


def res(x2, h, g, rows_per_mod, xout):
    ok = _launch(_res_kernel, (x2.shape[0],), x2, h, g, xout, rows_per_mod, D=x2.shape[1], num_warps=8)
    return xout if ok else None


def qk_ok(x: torch.Tensor, n_heads: int, head_dim: int) -> bool:
    return (x.dtype == torch.float16 and x.dim() == 3 and x.stride(2) == 1
            and x.stride(0) == x.shape[1] * x.stride(1)
            and _pow2(n_heads) and head_dim % 2 == 0 and _pow2(head_dim // 2))


def qk_norm_rope(x, weight, eps, n_heads, head_dim, freqs=None):
    """RMSNorm per head (then RoPE with ``freqs`` ``(L, head_dim/2, 2, 2)``) of
    ``x`` ``(B, S, H*HD)`` fp16 -> ``(B, S, H, HD)`` fp16."""
    B, S, _ = x.shape
    out = torch.empty(B, S, n_heads, head_dim, device=x.device, dtype=torch.float16)
    ok = _launch(_qk_norm_rope_kernel, (B * S,), x, weight, freqs if freqs is not None else weight, out,
                 S, x.stride(1), eps, H=n_heads, HD=head_dim, ROPE=freqs is not None, num_warps=2)
    return out if ok else None
