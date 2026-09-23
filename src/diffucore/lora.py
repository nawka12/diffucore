"""LoRA application by fusing weight deltas in place (``W += multiplier · ΔW``),
so sampling and offload are untouched: the fused weights travel with the module.

Factorizations: LoRA (``ΔW = (alpha / rank) · up @ down``) and LyCORIS LoKr
(``ΔW = kron(w1, w2)``, each factor full or low-rank ``a @ b``; ``alpha / dim``
applies only with a low-rank ``_b``, as in ComfyUI). No Tucker, LoHa or
diffusers-format SD files; their keys land in :attr:`LoraReport.unmatched`.

Keys: SD1.5 / SDXL use kohya's ``lora_unet_`` / ``lora_te_`` / ``lora_te1_`` /
``lora_te2_`` with ``.`` mangled to ``_`` (matched by re-mangling
``named_modules()``); bigG's split q/k/v keys add into row slices of its fused
``in_proj_weight``. Anima uses dotted ``diffusion_model.<path>`` with
``lora_A``/``lora_B`` (or kohya-mangled names).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .loading import load_state_dict

_CPU = torch.device("cpu")


def _eager(module):
    """Unwrap ``torch.compile``'s ``OptimizedModule`` so ``named_modules()`` yields
    the original paths instead of ``_orig_mod.<name>``."""
    return getattr(module, "_orig_mod", module)

# Tensor-name suffix -> factor role; a module's keys share its kohya base name.
_SUFFIXES = [
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora_A.weight", "down"),      # PEFT/diffusers naming (A=down, B=up); Anima LoRA
    (".lora_B.weight", "up"),
    (".lokr_w1_a", "w1_a"),
    (".lokr_w1_b", "w1_b"),
    (".lokr_w2_a", "w2_a"),
    (".lokr_w2_b", "w2_b"),
    (".lokr_w1", "w1"),
    (".lokr_w2", "w2"),
    (".lokr_t2", "t2"),
    (".alpha", "alpha"),
]


@dataclass
class LoraReport:
    """Outcome of :func:`apply_lora`: modules fused, and LoRA module names with no
    target (unsupported variants or a wrong-arch file)."""

    applied: int
    unmatched: list[str]


def apply_lora(bundle, path: str, multiplier: float = 1.0) -> LoraReport:
    """Fuse the LoRA at ``path`` into ``bundle`` scaled by ``multiplier``, in
    place. LoRAs stack; each touched weight is snapshotted to CPU first so
    :func:`remove_lora` / :func:`clear_loras` can undo it. Check the report's
    ``unmatched`` for keys that mapped to no layer.
    """
    state = _lora_state(bundle)
    targets = _build_targets(bundle)
    report = _fuse(path, multiplier, targets, state["base"])
    state["stack"].append((path, multiplier))
    return report


def remove_lora(bundle, path: str) -> None:
    """Remove the most recent application of ``path`` and re-fuse the rest, as
    if it had never been applied. Raises ``ValueError`` if it isn't applied."""
    stack = _lora_state(bundle)["stack"]
    for i in range(len(stack) - 1, -1, -1):
        if stack[i][0] == path:
            del stack[i]
            break
    else:
        raise ValueError(f"{path!r} is not in the active LoRA stack")
    _refuse(bundle)


def clear_loras(bundle) -> None:
    """Remove all LoRAs, restoring the base weights. No-op if none are applied."""
    state = _lora_state(bundle)
    _restore(state["base"])
    state["base"].clear()
    state["stack"].clear()


def _fuse(path: str, multiplier: float, targets, base) -> LoraReport:
    """Add the LoRA's scaled deltas into ``targets``, snapshotting each weight
    into ``base`` before its first modification."""
    groups = _group(load_state_dict(path, device="cpu"))
    applied, unmatched = 0, []
    for name, factors in groups.items():
        target = targets.get(name)
        if target is None:
            unmatched.append(name)          # no such layer (e.g. wrong-arch file)
            continue
        weight, row_start, row_end = target
        view = weight if row_start is None else weight[row_start:row_end]
        delta = _compose(factors, multiplier, view.shape)
        if delta is None:
            unmatched.append(name)          # unsupported factorization (e.g. Tucker)
            continue
        _snapshot(base, weight)
        with torch.no_grad():
            view.add_(delta.to(view.device, view.dtype))
        applied += 1

    return LoraReport(applied=applied, unmatched=unmatched)


def _refuse(bundle) -> None:
    """Restore the base weights, then replay the active stack from scratch."""
    state = _lora_state(bundle)
    _restore(state["base"])
    targets = _build_targets(bundle)
    for path, multiplier in state["stack"]:
        _fuse(path, multiplier, targets, state["base"])


def _snapshot(base, weight) -> None:
    """Save a pristine CPU copy of ``weight`` on first touch, keyed by storage
    address (``.data`` returns a new object each access). bigG's shared
    ``in_proj_weight`` is snapshotted once."""
    key = weight.data_ptr()
    if key not in base:
        base[key] = (weight, weight.detach().to(_CPU, copy=True))


def _restore(base) -> None:
    """Write every snapshotted weight back to its pristine values, in place."""
    with torch.no_grad():
        for weight, snapshot in base.values():
            weight.copy_(snapshot.to(weight.device, weight.dtype))


def _lora_state(bundle) -> dict:
    """The bundle's LoRA bookkeeping, ``{"stack": [(path, multiplier), ...],
    "base": {data_ptr: (weight, cpu_snapshot)}}``, created on first use."""
    state = getattr(bundle, "_lora_state", None)
    if state is None:
        state = {"stack": [], "base": {}}
        bundle._lora_state = state
    return state


def _group(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Collect ``{module_name: {role: tensor}}`` over all recognized suffixes."""
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        for suffix, role in _SUFFIXES:
            if key.endswith(suffix):
                groups.setdefault(key[: -len(suffix)], {})[role] = value
                break
    return groups


def _compose(factors: dict[str, torch.Tensor], multiplier: float, shape):
    """The weight delta for one module reshaped to ``shape``, or ``None`` if
    unsupported. Computed in fp32; the caller casts."""
    if "down" in factors and "up" in factors:
        return _compose_lora(factors, multiplier, shape)
    if "w1" in factors or "w1_a" in factors:
        return _compose_lokr(factors, multiplier, shape)
    return None


def _compose_lora(factors, multiplier, shape):
    """LoRA: ``(alpha/rank) · multiplier · (up @ down)``."""
    down = factors["down"].float()
    up = factors["up"].float()
    rank = down.shape[0]
    alpha = factors["alpha"].item() if "alpha" in factors else float(rank)
    scale = (alpha / rank) * multiplier
    # Flattening makes Linear and Conv factors one (out, r)·(r, ·) matmul.
    delta = up.reshape(up.shape[0], -1) @ down.reshape(rank, -1)
    return delta.reshape(shape) * scale


def _compose_lokr(factors, multiplier, shape):
    """LoKr: ``multiplier · scale · kron(w1, w2)``; ``scale = alpha/dim`` only
    when a ``_b`` factor supplies ``dim``."""
    if "t2" in factors:                          # Tucker conv decomposition
        return None
    w1 = factors["w1"].float() if "w1" in factors else factors["w1_a"].float() @ factors["w1_b"].float()
    if "w2" in factors:
        w2 = factors["w2"].float()
    elif "w2_a" in factors and "w2_b" in factors:
        w2 = factors["w2_a"].float() @ factors["w2_b"].float()
    else:
        return None

    dim = factors["w1_b"].shape[0] if "w1_b" in factors else (
        factors["w2_b"].shape[0] if "w2_b" in factors else None)
    scale = multiplier
    if dim is not None and "alpha" in factors:
        scale *= factors["alpha"].item() / dim

    # Conv: lift the 2-D w1 so kron yields (out1·out2, in1·in2, kh, kw).
    if w2.dim() == 4 and w1.dim() == 2:
        w1 = w1.reshape(*w1.shape, 1, 1)
    return torch.kron(w1, w2).reshape(shape) * scale


def _build_targets(bundle) -> dict[str, tuple[torch.Tensor, int | None, int | None]]:
    """Map kohya module names to ``(weight, row_start, row_end)``; the row slice
    addresses bigG's fused ``in_proj_weight``."""
    targets: dict[str, tuple[torch.Tensor, int | None, int | None]] = {}

    backbone = _eager(bundle.backbone)
    if getattr(bundle.spec, "architecture", None) == "anima":
        # Anima LoRAs come dotted under ``diffusion_model.`` (ComfyUI/musubi) or
        # kohya-mangled as ``lora_unet_blocks_...``; register both.
        _add_dotted_targets(targets, backbone, "diffusion_model.")
        _add_module_targets(targets, backbone, ["lora_unet_"])
        return targets

    _add_module_targets(targets, backbone, ["lora_unet_"])
    if bundle.text_encoder is not None:
        # lora_te_ (SD1.5) and lora_te1_ (SDXL) both name the CLIP-L encoder.
        _add_module_targets(targets, bundle.text_encoder, ["lora_te_", "lora_te1_"])
    if bundle.text_encoder_2 is not None:
        _add_bigg_targets(targets, bundle.text_encoder_2, "lora_te2_")

    return targets


def _add_module_targets(targets, model, prefixes) -> None:
    """Register every Linear/Conv2d weight under ``<prefix><dotted_name_as_>``."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            mangled = name.replace(".", "_")
            for prefix in prefixes:
                targets[prefix + mangled] = (module.weight.data, None, None)


def _add_dotted_targets(targets, model, prefix) -> None:
    """Register every Linear/Conv2d weight under ``<prefix><dotted_name>``, the
    Anima/DiT convention."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            targets[prefix + name] = (module.weight.data, None, None)


def _add_bigg_targets(targets, model, prefix) -> None:
    """OpenCLIP bigG: map kohya's HF-style split q/k/v keys onto its fused
    ``in_proj_weight`` and ``transformer.resblocks`` layout."""
    for i, block in enumerate(model.transformer.resblocks):
        in_proj = block.attn.in_proj_weight.data  # [3*width, width], rows q|k|v
        width = in_proj.shape[1]
        attn = f"{prefix}text_model_encoder_layers_{i}_self_attn_"
        targets[attn + "q_proj"] = (in_proj, 0, width)
        targets[attn + "k_proj"] = (in_proj, width, 2 * width)
        targets[attn + "v_proj"] = (in_proj, 2 * width, 3 * width)
        targets[attn + "out_proj"] = (block.attn.out_proj.weight.data, None, None)
        mlp = f"{prefix}text_model_encoder_layers_{i}_mlp_"
        targets[mlp + "fc1"] = (block.mlp.c_fc.weight.data, None, None)
        targets[mlp + "fc2"] = (block.mlp.c_proj.weight.data, None, None)


__all__ = ["apply_lora", "remove_lora", "clear_loras", "LoraReport"]
