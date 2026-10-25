"""Attention dispatch: PyTorch SDPA (default), the optional FA2-Turing port, or
the in-tree INT8 Turing kernel.

Flash SDPA needs sm80+, so Turing (sm75) falls back to the mem-efficient kernel.
The community sm75 FlashAttention-2 port (github.com/ssiu/flash-attention-turing,
package ``flash_attn_turing``) is ×1.3–1.6 faster there on DiT shapes. It is a
locally built optional extra; without it everything resolves to plain SDPA,
and ``auto`` never picks it off sm75.

``int8_turing`` runs Q K^T on INT8 and P V on fp16-accumulate tensor cores:
``kernels/int8_attn_sm75.cu`` on Turing (~×1.75 over FA2-Turing on an RTX 2060)
and ``kernels/int8_attn_sm80.cu`` on sm80+ (~×1.85 over SDPA on an RTX 5090;
GeForce cards run fp16 with fp32 accumulation at half rate). It is
approximate (per-call error ~0.4% vs ~0.02% for FA2), so ``auto`` never picks it
and it only replaces self-attention: Anima's cross-attention (512 text keys, the
largest INT8 error, ~1% of the attention time) keeps the exact backend. It is
JIT-built with torch's cpp_extension (nvcc + ninja + a C++ compiler) when a model
first selects it.
"""

from __future__ import annotations

import functools
import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  no Triton: torch ops
    triton = None

_CHOICES = ("sdpa", "auto", "fa2_turing", "int8_turing")
_KERNELS = Path(__file__).resolve().parent.parent / "kernels"
_int8_build_error = ""
_keep_broken = False


@functools.cache
def _fa2_module():
    try:
        import flash_attn_turing
    except Exception:  # noqa: BLE001  absent or broken build: "not installed"
        return None
    return flash_attn_turing


def _is_sm75(device=None) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) == (7, 5)


def fa2_turing_available(device=None) -> bool:
    """Whether the FA2-Turing kernel can run here: package installed and the
    CUDA device is sm75 (the extension is compiled with ``-arch=sm_75`` only)."""
    if _fa2_module() is None or not torch.cuda.is_available():
        return False
    if device is not None and getattr(device, "type", None) == "cpu":
        return False
    return _is_sm75(device)


def _toolchain_available() -> bool:
    from torch.utils import cpp_extension
    # The host compiler too: without it (MSVC's cl on Windows) the build fails at load.
    return (cpp_extension.CUDA_HOME is not None and cpp_extension.is_ninja_available()
            and shutil.which(cpp_extension.get_cxx_compiler()) is not None)


def _int8_arch(device=None) -> str | None:
    """The sm arch an INT8 kernel is built for here ("75" or "80"+), else None."""
    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability(device)
    return f"{cap[0]}{cap[1]}" if cap == (7, 5) or cap >= (8, 0) else None


def int8_turing_available(device=None) -> bool:
    """Whether the INT8 kernel can be used here: an sm75 or sm80+ CUDA device
    and a CUDA toolchain to build it. Cheap; the build happens at model load."""
    if device is not None and getattr(device, "type", None) == "cpu":
        return False
    return _int8_arch(device) is not None and _toolchain_available()


@functools.cache
def _int8_module():
    """Build (first time; torch caches the binary by source hash) and import the
    INT8 kernel for this GPU's arch. ``None`` when the build fails, with the
    reason kept for the explicit-backend error."""
    global _int8_build_error
    arch = _int8_arch()
    try:
        from torch.utils.cpp_extension import load
        src = _KERNELS / ("int8_attn_sm75.cu" if arch == "75" else "int8_attn_sm80.cu")
        return load(
            name=f"diffucore_int8_attn_sm{arch}", sources=[str(src)],
            extra_cuda_cflags=["-O3", "-std=c++20", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            extra_cflags=["-O3", "-std=c++20"], verbose=False,
        )
    except Exception as e:  # noqa: BLE001  any toolchain failure means "unavailable"
        _int8_build_error = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        return None


def _common_requirements(policy, arch_note, arch_ok):
    yield "device is not CUDA", policy.device.type == "cuda"
    yield arch_note, policy.device.type == "cuda" and arch_ok(policy.device)
    yield "compute dtype is not fp16", policy.compute_dtype == torch.float16
    yield ("incompatible with compile=True (the custom op graph-breaks in "
           "every block)", not policy.compile)


def _fa2_requirements(policy):
    """(message, satisfied) pairs for running FA2-Turing under ``policy``."""
    yield "flash_attn_turing is not installed", _fa2_module() is not None
    yield from _common_requirements(policy, "GPU is not Turing; the kernel is compiled for sm_75 only", _is_sm75)


def _int8_requirements(policy):
    """(message, satisfied) pairs for the INT8 kernel; builds it only when
    everything else holds."""
    checks = list(_common_requirements(policy, "GPU is older than Turing; the INT8 kernels need sm75 or sm80+",
                                       lambda d: _int8_arch(d) is not None))
    checks.append(("no CUDA toolchain to build the INT8 kernel (needs nvcc, ninja and a C++ compiler)",
                   _toolchain_available()))
    yield from checks
    if all(ok for _, ok in checks):
        built = _int8_module() is not None
        yield f"INT8 kernel build failed: {_int8_build_error}", built


def resolve_attention_backend(policy) -> str:
    """Map ``policy.attention`` to a backend: ``"sdpa"`` as is, ``"auto"`` to
    FA2-Turing only when every requirement holds, and an explicit
    ``"fa2_turing"`` / ``"int8_turing"`` raises on the first unmet requirement
    (so an A/B never silently runs the same backend twice).
    """
    if policy.attention == "sdpa":
        return "sdpa"
    if policy.attention == "int8_turing":
        failures = [msg for msg, ok in _int8_requirements(policy) if not ok]
        if failures:
            raise ValueError(
                "policy.attention='int8_turing' can't run: " + "; ".join(failures)
            )
        return "int8_turing"
    failures = [msg for msg, ok in _fa2_requirements(policy) if not ok]
    if not failures:
        return "fa2_turing"
    if policy.attention == "auto":
        return "sdpa"
    raise ValueError(
        "policy.attention='fa2_turing' can't run: " + "; ".join(failures)
    )


def set_attention_backend(module: torch.nn.Module, backend: str) -> int:
    """Stamp ``backend`` on every submodule that declares an ``attn_backend``
    slot (the Anima/FLUX attention modules). Returns the number stamped.

    ``int8_turing`` goes on self-attention only; modules that declare
    ``is_selfattn = False`` get the exact backend (FA2-Turing if usable, else
    SDPA)."""
    exact = "fa2_turing" if fa2_turing_available() else "sdpa"
    n = 0
    for m in module.modules():
        if hasattr(m, "attn_backend"):
            cross = not getattr(m, "is_selfattn", True)
            m.attn_backend = exact if backend == "int8_turing" and cross else backend
            n += 1
    return n


def _fa2_eligible(q: torch.Tensor) -> bool:
    """Per-call guard so a stamped module still runs correctly off the happy
    path (CPU test forwards, fp32 detours, foreign head_dims)."""
    return q.is_cuda and q.dtype == torch.float16 and q.shape[-1] in (64, 128)


def _int8_eligible(*ts: torch.Tensor) -> bool:
    """Same guard for the INT8 kernel, which also reads strided inputs as long
    as the head dim is packed and 16-byte aligned."""
    return all(
        t.is_cuda and t.dtype == torch.float16 and t.shape[-1] == 128
        and t.stride(-1) == 1 and all(s % 8 == 0 for s in t.stride()[:-1])
        and t.data_ptr() % 16 == 0
        for t in ts
    ) and _int8_module() is not None


def attention_blhd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   backend: str = "sdpa", null_keys: int = 0) -> torch.Tensor:
    """Attention over ``(B, L, H, D)`` q/k/v → ``(B, Lq, H·D)``. FA2 takes that
    layout natively but ignores strides, so its inputs are made contiguous.

    ``null_keys`` counts all-zero key/value rows left off ``k``/``v``: each
    scores exactly 0 and adds only to the softmax denominator. FA2 runs on the
    real keys and rescales by its log-sum-exp; the other backends get the zero
    rows back, so their result is the padded one.
    """
    B, Lq = q.shape[0], q.shape[1]
    if null_keys:
        if backend == "fa2_turing" and _fa2_eligible(q):
            out, lse = _fa2_module().fwd(
                q.contiguous(), k.contiguous(), v.contiguous(),
                q.shape[-1] ** -0.5, False,
            )
            _scale_to_real_share(out, lse, null_keys)
            return out.reshape(B, Lq, -1)
        k = F.pad(k, (0, 0, 0, 0, 0, null_keys))
        v = F.pad(v, (0, 0, 0, 0, 0, null_keys))
    if backend == "int8_turing" and _int8_eligible(q, k, v):
        return _int8_module().fwd(q, k, v, q.shape[-1] ** -0.5).reshape(B, Lq, -1)
    if backend == "fa2_turing" and _fa2_eligible(q):
        out, _ = _fa2_module().fwd(
            q.contiguous(), k.contiguous(), v.contiguous(),
            q.shape[-1] ** -0.5, False,
        )
        return out.reshape(B, Lq, -1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).reshape(B, Lq, -1)


def _scale_to_real_share(out: torch.Tensor, lse: torch.Tensor, n: int) -> None:
    """``out *= Z / (Z + n)`` in place, ``Z = e^lse``: the real keys' share of a
    softmax that also holds ``n`` zero-score keys. ``out`` is FA2's packed
    ``(B, L, H, D)``, ``lse`` its ``(B, H, L)`` fp32 log-sum-exp. One Triton
    pass (torch's broadcast mixed-dtype multiply is ~2.5x slower)."""
    global _keep_broken
    B, L, H, D = out.shape
    pow2 = H & (H - 1) == 0 and D & (D - 1) == 0
    if triton is not None and not _keep_broken and out.is_cuda and lse.is_contiguous() and pow2:
        try:
            _keep_kernel[(B * L,)](out, lse, math.log(n), L, H=H, D=D, num_warps=4)
            return
        except Exception as e:  # noqa: BLE001  any compile/launch failure
            print(f"[attention] Triton unavailable ({type(e).__name__}); using torch ops", flush=True)
            _keep_broken = True
    out.mul_(torch.sigmoid(lse - math.log(n)).transpose(1, 2).unsqueeze(-1))


if triton is not None:

    @triton.jit
    def _keep_kernel(o_ptr, lse_ptr, log_n, L, H: tl.constexpr, D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)   # token: b * L + l
        b, l = row // L, row % L
        h = tl.arange(0, H)
        keep = tl.sigmoid(tl.load(lse_ptr + (b * H + h) * L + l) - log_n)
        offs = row * H * D + h[:, None] * D + tl.arange(0, D)[None, :]
        o = tl.load(o_ptr + offs).to(tl.float32)
        tl.store(o_ptr + offs, (o * keep[:, None]).to(o_ptr.dtype.element_ty))


def attention_bhld(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   backend: str = "sdpa") -> torch.Tensor:
    """Attention over ``(B, H, L, D)`` q/k/v → ``(B, Lq, H·D)`` (FLUX layout,
    post-RoPE). The FA2 branch pays a transpose-copy per tensor (the kernel
    assumes packed ``(B, L, H, D)``); small next to the kernel win. The INT8
    kernel reads the transposed views directly."""
    B, Lq = q.shape[0], q.shape[2]
    if backend == "int8_turing":
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if _int8_eligible(qt, kt, vt):
            return _int8_module().fwd(qt, kt, vt, q.shape[-1] ** -0.5).reshape(B, Lq, -1)
    if backend == "fa2_turing" and _fa2_eligible(q):
        out, _ = _fa2_module().fwd(
            q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(), q.shape[-1] ** -0.5, False,
        )
        return out.reshape(B, Lq, -1)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(B, Lq, -1)
