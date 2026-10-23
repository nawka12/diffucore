# SPDX-FileCopyrightText: Copyright (c) 2026 Dzakwan Haq (KayfaHaarukku, github.com/nawka12)
# SPDX-License-Identifier: Apache-2.0

"""INT8 weight + activation (W8A8) linears.

Reads ComfyUI's ``int8_tensorwise`` checkpoints (int8 weights, one fp32 scale
per output row, a ``comfy_quant`` JSON tag per layer) and quantizes fp16/bf16
weights the same way at load. ConvRot files store each weight rotated as
``W @ H`` per 256-wide input group, ``H`` the normalized regular Hadamard
(Kronecker powers of ``[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]``). ``H``
is symmetric and its own inverse, so ``x @ W.T == (x @ H) @ (W @ H).T``: the
activation gets the same rotation, which spreads outlier channels before the
per-token quantization.

A forward rotates and quantizes each activation row, runs the int8 GEMM
(``torch._int_mm``, cuBLASLt) and rescales the int32 result. On CUDA with
Triton the rotate+quantize and rescale steps are one row pass each, and an int8
MLP fuses rescale + GELU + rotate + quantize between its GEMMs; elsewhere (no
Triton, ``torch.compile``) the same math runs as torch ops.
"""

from __future__ import annotations

import json
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # noqa: BLE001  no Triton (e.g. a Windows install): torch ops
    triton = None

GROUP = 256          # ConvRot group used when quantizing at load
_broken = False
_NO_FMA = {"enable_fp_fusion": False}
_HADAMARD: dict = {}


def int8_fast_available(device=None) -> bool:
    """Whether int8 layers run fast here: CUDA with int8 tensor cores (sm75+)
    and Triton. Elsewhere an INT8 file is dequantized to plain linears."""
    if triton is None or _broken or not torch.cuda.is_available():
        return False
    device = torch.device(device) if device is not None else torch.device("cuda")
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) >= (7, 5)


def _hadamard(n: int, device, dtype=torch.float32) -> torch.Tensor:
    key = (n, str(device), dtype)
    h = _HADAMARD.get(key)
    if h is None:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = h4
        while h.shape[0] < n:
            h = torch.kron(h, h4)
        if h.shape[0] != n:
            raise ValueError(f"ConvRot group must be a power of 4, got {n}")
        h = (h / n ** 0.5).to(device, dtype)
        _HADAMARD[key] = h
    return h


def _rotate(t: torch.Tensor, group: int) -> torch.Tensor:
    """``t @ H`` per ``group``-wide slice of the last dim (no-op for group 0)."""
    if not group:
        return t
    shape = t.shape
    return (t.reshape(-1, shape[-1] // group, group) @ _hadamard(group, t.device, t.dtype)).reshape(shape)


def _quantize_rows(t: torch.Tensor, generator: torch.Generator | None = None):
    """Per-row symmetric int8: ``scale = amax / 127``. ``generator`` switches
    to stochastic rounding (unbiased, for re-quantizing on-grid weights)."""
    t = t.float()
    scale = (t.abs().amax(-1) / 127.0).clamp(min=1e-30)
    t = t / scale.unsqueeze(-1)
    if generator is None:
        t = t.round()
    else:
        t = (t + torch.rand(t.shape, device=t.device, generator=generator)).floor()
    return t.clamp(-128, 127).to(torch.int8), scale


def quantize_weight(w: torch.Tensor, group: int = GROUP):
    """``(int8 [N, K], fp32 scale [N])`` for ``w`` ``[N, K]``, ConvRot-rotated."""
    return _quantize_rows(_rotate(w.float(), group))


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """The fp32 weight in the original basis (``H`` undoes itself)."""
    return _rotate(q.float() * scale.unsqueeze(-1), group)


class Int8Linear(nn.Module):
    """``nn.Linear`` with int8 weights. The per-row fp32 scale lives as int32
    bits (``scale_bits``) so ``module.to(dtype)`` never rounds it."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, group: int = GROUP,
                 device=None):
        super().__init__()
        if group and in_features % group:
            raise ValueError(f"in_features {in_features} is not a multiple of the ConvRot group {group}")
        self.in_features, self.out_features, self.group = in_features, out_features, group
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("scale_bits", torch.empty(out_features, dtype=torch.int32, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device), requires_grad=False) if bias else None

    @property
    def scale(self) -> torch.Tensor:
        return self.scale_bits.view(torch.float32)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group: int = GROUP, device=None) -> "Int8Linear":
        device = device if device is not None else linear.weight.device
        q, s = quantize_weight(linear.weight.detach().to(device), group)
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, group, device=device)
        m.weight.copy_(q)
        m.scale_bits.copy_(s.view(torch.int32))
        if linear.bias is not None:
            m.bias = nn.Parameter(linear.bias.detach().to(device), requires_grad=False)
        return m

    def to_linear(self, dtype: torch.dtype) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                        device="meta")
        lin.weight = nn.Parameter(dequantize_weight(self.weight, self.scale, self.group).to(dtype),
                                  requires_grad=False)
        if self.bias is not None:
            lin.bias = nn.Parameter(self.bias.detach().to(dtype), requires_grad=False)
        return lin

    @torch.no_grad()
    def add_delta_(self, delta: torch.Tensor, seed: int) -> None:
        """``W += delta``, re-quantized with seeded stochastic rounding: the
        weights sit on the int8 grid, so rounding to nearest would erase any
        delta under half a step (most of a typical LoRA)."""
        w = self.weight.float() * self.scale.unsqueeze(-1) + _rotate(delta.float(), self.group)
        q, s = _quantize_rows(w, torch.Generator(device=w.device).manual_seed(seed))
        self.weight.copy_(q)
        self.scale_bits.copy_(s.view(torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, self.in_features)
        q, s = quantize_activation(x2, self.group)
        y = rescale(int_mm(q, self.weight), s, self.scale, self.bias, x.dtype)
        return y.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, group={self.group}"


class Int8Embedding(nn.Module):
    """``nn.Embedding`` over an int8 ConvRot table with one fp32 scale per row
    (ComfyUI's int8 text-encoder files quantize ``embed_tokens`` too). A lookup
    gathers the rows, rescales and un-rotates only those, so the table is never
    expanded; ``out_dtype`` is the activation dtype the lookup returns."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group: int = GROUP, device=None):
        super().__init__()
        if group and embedding_dim % group:
            raise ValueError(f"embedding_dim {embedding_dim} is not a multiple of the ConvRot group {group}")
        self.num_embeddings, self.embedding_dim, self.group = num_embeddings, embedding_dim, group
        self.out_dtype = torch.float32
        self.register_buffer("weight", torch.empty(num_embeddings, embedding_dim, dtype=torch.int8, device=device))
        self.register_buffer("scale_bits", torch.empty(num_embeddings, dtype=torch.int32, device=device))

    @property
    def scale(self) -> torch.Tensor:
        return self.scale_bits.view(torch.float32)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = dequantize_weight(self.weight[ids.reshape(-1)], self.scale[ids.reshape(-1)], self.group)
        return rows.reshape(*ids.shape, self.embedding_dim).to(self.out_dtype)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, group={self.group}"


def int_mm(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """int32 ``q @ w.T``. ``w.t()`` stays a transposed view: cuBLASLt is ~4x
    slower on a row-major ``[K, N]`` copy. ``_int_mm`` needs M > 16."""
    m = q.shape[0]
    if m <= 16:
        q = F.pad(q, (0, 0, 0, 32 - m))
    out = torch._int_mm(q, w.t())
    return out[:m] if m <= 16 else out


def _fast(t: torch.Tensor) -> bool:
    return triton is not None and not _broken and t.is_cuda and not torch.compiler.is_compiling()


def _launch(kernel, grid, *args, **kw) -> bool:
    """Run a kernel; on any failure use torch ops for the rest of the process."""
    global _broken
    try:
        kernel[grid](*args, **kw, **_NO_FMA)
        return True
    except Exception as e:  # noqa: BLE001  any compile/launch failure
        print(f"[int8] Triton kernels unavailable ({type(e).__name__}: {e}); using torch ops", flush=True)
        _broken = True
        return False


def _warps(block: int) -> int:
    return 4 if block <= 2048 else 8


def quantize_activation(x2: torch.Tensor, group: int):
    """``(int8 [M, K], fp32 scale [M])``: rotate, then per-row int8."""
    if _fast(x2):
        x2 = x2.contiguous()
        M, K = x2.shape
        q = torch.empty(M, K, device=x2.device, dtype=torch.int8)
        s = torch.empty(M, device=x2.device, dtype=torch.float32)
        block = triton.next_power_of_2(K)
        if _launch(_rot_quant_kernel, (M,), x2, q, s, K, BLOCK=block, ROT=bool(group),
                   num_warps=_warps(block)):
            return q, s
    return _quantize_rows(_rotate(x2.float(), group))


def rescale(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor, bias, dtype) -> torch.Tensor:
    """``acc * sx[:, None] * sw[None, :] (+ bias)`` in ``dtype``."""
    if _fast(acc):
        M, N = acc.shape
        out = torch.empty(M, N, device=acc.device, dtype=dtype)
        block = triton.next_power_of_2(N)
        if _launch(_rescale_kernel, (M,), acc, sx, sw, bias if bias is not None else sw, out, N,
                   BLOCK=block, HAS_BIAS=bias is not None, num_warps=_warps(block)):
            return out
    y = (acc.float() * (sx.unsqueeze(-1) * sw)).to(dtype)
    return y + bias.to(dtype) if bias is not None else y


def int8_mlp(x: torch.Tensor, l1: Int8Linear, l2: Int8Linear) -> torch.Tensor:
    """``l2(gelu(l1(x)))`` with l1's rescale, the GELU and l2's
    rotate+quantize in one row pass (casts to ``x.dtype`` where eager does)."""
    x2 = x.reshape(-1, l1.in_features)
    if _fast(x2) and l1.bias is None and x.dtype in (torch.float16, torch.bfloat16):
        q, s = quantize_activation(x2, l1.group)
        acc = int_mm(q, l1.weight)
        M, N = acc.shape
        q2 = torch.empty(M, N, device=acc.device, dtype=torch.int8)
        s2 = torch.empty(M, device=acc.device, dtype=torch.float32)
        block = triton.next_power_of_2(N)
        if _launch(_gelu_rot_quant_kernel, (M,), acc, s, l1.scale, q2, s2, N, BLOCK=block,
                   ROT=bool(l2.group), BF16=x.dtype == torch.bfloat16, num_warps=_warps(block)):
            y = rescale(int_mm(q2, l2.weight), s2, l2.scale, l2.bias, x.dtype)
            return y.reshape(*x.shape[:-1], l2.out_features)
    return l2(F.gelu(l1(x)))


if triton is not None:

    @triton.jit
    def _hadamard_rows(x, BLOCK: tl.constexpr):
        # x @ H per 256-group: H4 along each of the four base-4 digits of the
        # in-group index is ``sum - 2 * reversed`` (H4[i, j] = -1 iff i + j == 3).
        y = tl.reshape(x, (BLOCK // 256, 4, 4, 4, 4))
        y = tl.sum(y, 1, keep_dims=True) - 2.0 * tl.flip(y, 1)
        y = tl.sum(y, 2, keep_dims=True) - 2.0 * tl.flip(y, 2)
        y = tl.sum(y, 3, keep_dims=True) - 2.0 * tl.flip(y, 3)
        y = tl.sum(y, 4, keep_dims=True) - 2.0 * tl.flip(y, 4)
        return tl.reshape(y, (BLOCK,)) * 0.0625

    # The row width is a constexpr (a model has a handful): with it known,
    # Triton proves row alignment and vectorizes; a runtime width made the
    # int8 stores bytewise and these kernels ~10x slower. Masks only appear
    # when the width isn't a power of two.

    @triton.jit
    def _load_row(ptr, row, offs, K: tl.constexpr, BLOCK: tl.constexpr):
        if BLOCK == K:
            return tl.load(ptr + row * K + offs)
        return tl.load(ptr + row * K + offs, mask=offs < K, other=0)

    @triton.jit
    def _store_row(ptr, row, offs, v, K: tl.constexpr, BLOCK: tl.constexpr):
        if BLOCK == K:
            tl.store(ptr + row * K + offs, v)
        else:
            tl.store(ptr + row * K + offs, v, mask=offs < K)

    @triton.jit
    def _store_quantized(x, q_ptr, s_ptr, row, offs, K: tl.constexpr, BLOCK: tl.constexpr):
        scale = tl.maximum(tl.max(tl.abs(x), 0) / 127.0, 1e-30)
        q = tl.minimum(tl.maximum(libdevice.rint(x / scale), -128.0), 127.0)
        _store_row(q_ptr, row, offs, q.to(tl.int8), K, BLOCK)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _rot_quant_kernel(x_ptr, q_ptr, s_ptr, K: tl.constexpr, BLOCK: tl.constexpr, ROT: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        x = _load_row(x_ptr, row, offs, K, BLOCK).to(tl.float32)
        if ROT:
            x = _hadamard_rows(x, BLOCK)
        _store_quantized(x, q_ptr, s_ptr, row, offs, K, BLOCK)

    @triton.jit
    def _rescale_kernel(acc_ptr, sx_ptr, sw_ptr, b_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
                        HAS_BIAS: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        a = _load_row(acc_ptr, row, offs, N, BLOCK).to(tl.float32)
        y = (a * (tl.load(sx_ptr + row) * _load_row(sw_ptr, 0, offs, N, BLOCK))).to(out_ptr.dtype.element_ty)
        if HAS_BIAS:
            y = y + _load_row(b_ptr, 0, offs, N, BLOCK).to(out_ptr.dtype.element_ty)
        _store_row(out_ptr, row, offs, y, N, BLOCK)

    @triton.jit
    def _gelu_rot_quant_kernel(acc_ptr, sx_ptr, sw_ptr, q_ptr, s_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
                               ROT: tl.constexpr, BF16: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        dt = tl.bfloat16 if BF16 else tl.float16
        a = _load_row(acc_ptr, row, offs, N, BLOCK).to(tl.float32)
        h = (a * (tl.load(sx_ptr + row) * _load_row(sw_ptr, 0, offs, N, BLOCK))).to(dt).to(tl.float32)
        h = (0.5 * h * (1.0 + tl.math.erf(h * 0.7071067811865476))).to(dt).to(tl.float32)
        if ROT:
            h = _hadamard_rows(h, BLOCK)
        _store_quantized(h, q_ptr, s_ptr, row, offs, N, BLOCK)


# ─── checkpoint plumbing ─────────────────────────────────────────────────────

def _parent(model: nn.Module, name: str):
    parent, _, attr = name.rpartition(".")
    return (model.get_submodule(parent) if parent else model), attr


def int8_layers(state_dict: dict) -> dict[str, int]:
    """``{layer: ConvRot group (0 = none)}`` for each ``comfy_quant``-tagged
    layer of a ComfyUI quantized checkpoint; empty for a plain one."""
    layers = {}
    for key, tag in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        name = key[: -len(".comfy_quant")]
        meta = json.loads(bytes(tag.tolist()).decode())
        if meta.get("format") != "int8_tensorwise":
            raise ValueError(f"{name}: quantized format {meta.get('format')!r} is not supported "
                             f"(diffucore reads int8_tensorwise only)")
        layers[name] = int(meta.get("convrot_groupsize", 256)) if meta.get("convrot") else 0
    return layers


def adopt_int8_layers(model: nn.Module, state_dict: dict, layers: dict[str, int]) -> dict:
    """Swap each tagged ``nn.Linear`` of a meta-built ``model`` for an
    :class:`Int8Linear` and rename its checkpoint entries to match, so a strict
    ``load_state_dict`` follows."""
    for name, group in layers.items():
        parent, attr = _parent(model, name)
        lin = getattr(parent, attr)
        setattr(parent, attr, Int8Linear(lin.in_features, lin.out_features, lin.bias is not None,
                                         group, device="meta"))
        scale = state_dict.pop(name + ".weight_scale").float().reshape(-1)
        state_dict[name + ".scale_bits"] = scale.expand(lin.out_features).contiguous().view(torch.int32)
        del state_dict[name + ".comfy_quant"]
    return state_dict


def adopt_int8_embedding(model: nn.Module, state_dict: dict, name: str, group: int) -> dict:
    """:func:`adopt_int8_layers` for an ``nn.Embedding`` (``name`` is the
    embedding module, e.g. ``embed_tokens``)."""
    parent, attr = _parent(model, name)
    emb = getattr(parent, attr)
    setattr(parent, attr, Int8Embedding(emb.num_embeddings, emb.embedding_dim, group, device="meta"))
    scale = state_dict.pop(name + ".weight_scale").float().reshape(-1)
    state_dict[name + ".scale_bits"] = scale.expand(emb.num_embeddings).contiguous().view(torch.int32)
    del state_dict[name + ".comfy_quant"]
    return state_dict


def quantize_linears(model: nn.Module, names, device) -> int:
    """Replace the named ``nn.Linear`` layers with ConvRot :class:`Int8Linear`,
    quantizing on ``device`` (moved back to each layer's own device)."""
    for name in names:
        parent, attr = _parent(model, name)
        lin = getattr(parent, attr)
        setattr(parent, attr, Int8Linear.from_linear(lin, GROUP, device).to(lin.weight.device))
    return len(names)


def dequantize_linears(model: nn.Module, dtype: torch.dtype, device=None) -> int:
    """Turn every :class:`Int8Linear` back into an fp ``nn.Linear`` (the
    fallback where int8 can't run fast), working on ``device`` if given."""
    names = [n for n, m in model.named_modules() if isinstance(m, Int8Linear)]
    for name in names:
        parent, attr = _parent(model, name)
        m = getattr(parent, attr)
        home = m.weight.device
        setattr(parent, attr, (m.to(device) if device is not None else m).to_linear(dtype).to(home))
    return len(names)


def lora_seed(name: str, path: str) -> int:
    """Deterministic stochastic-rounding seed per (layer, LoRA file)."""
    return zlib.crc32(f"{name}|{path}".encode())
