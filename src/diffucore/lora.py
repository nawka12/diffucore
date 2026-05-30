"""LoRA application (kohya / A1111 format) by fusing weight deltas in place.

Each entry stores, per targeted linear/conv layer, the factors of a weight delta
``ΔW``; we *fuse* that delta into the existing module weight
(``W += multiplier · ΔW``) rather than wrapping the forward, so the sampling path
and the offload machinery (which just moves modules around) are untouched — the
fused weights travel with the module.

Two factorizations are supported:

  - **LoRA** (kohya/A1111): ``ΔW = (alpha / rank) · (lora_up @ lora_down)``.
  - **LoKr** (LyCORIS): ``ΔW = kron(w1, w2)``, where each factor is either a full
    matrix (``lokr_w1`` / ``lokr_w2``) or itself low-rank
    (``lokr_w{1,2}_a @ lokr_w{1,2}_b``). The ``alpha / dim`` scaling applies only
    when a low-rank ``_b`` factor supplies a ``dim`` (matching ComfyUI); full
    matrices fuse at scale 1. Tucker (``lokr_t2``) decomposition is not supported.

Key naming follows the checkpoint's family:

  - **SD1.5 / SDXL** (kohya/A1111): ``lora_unet_<path>`` / ``lora_te_`` /
    ``lora_te1_`` (CLIP-L) / ``lora_te2_`` (bigG), where ``<path>`` is the dotted
    module name with ``.`` replaced by ``_``. We map it back by walking
    ``named_modules()`` and re-mangling, sidestepping the ambiguity of
    underscores inside names. These names mirror the CompVis/LDM module names the
    backbones already mirror. bigG is the one special case: its attention stores a
    fused ``in_proj_weight`` while kohya keys are split q/k/v, so those deltas are
    added to the matching row-slice of the fused weight.
  - **Anima** (DiT): ``diffusion_model.<dotted_path>`` with ``lora_A``/``lora_B``
    factors — dotted module names matching the DiT's own submodules, no mangling.

Diffusers-format SD LoRAs and the LoHa LyCORIS variant are not supported; their
keys land in :attr:`LoraReport.unmatched`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .loading import load_state_dict

# Tensor-name suffixes -> the role we file them under. A module's keys share a
# base (the kohya module name); the suffix says which factor each tensor is.
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
    """Outcome of :func:`apply_lora`: how many modules were fused and which LoRA
    module names found no target (unsupported variants or a wrong-arch file)."""

    applied: int
    unmatched: list[str]


def apply_lora(bundle, path: str, multiplier: float = 1.0) -> LoraReport:
    """Fuse a kohya-format LoRA at ``path`` into ``bundle``'s UNet and text
    encoder(s), scaled by ``multiplier``. Mutates the module weights in place.

    Returns a :class:`LoraReport`; inspect ``unmatched`` to spot keys that did
    not map to a layer (e.g. a SD1.5 LoRA applied to an SDXL bundle, or an
    unsupported LyCORIS file).
    """
    state_dict = load_state_dict(path, device="cpu")
    groups = _group(state_dict)
    targets = _build_targets(bundle)

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
        with torch.no_grad():
            view.add_(delta.to(view.device, view.dtype))
        applied += 1

    return LoraReport(applied=applied, unmatched=unmatched)


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
    """The weight delta for one module, reshaped to ``shape``, or ``None`` if the
    factorization is unsupported. Computed in fp32 (matmuls are precision
    sensitive and CPU fp16 matmul is patchy); the caller casts to the weight dtype."""
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
    # Linear: down [r, in], up [out, r]. Conv: down [r, in, kh, kw],
    # up [out, r, 1, 1]. Flattening makes both a plain (out, r)·(r, ·) matmul.
    delta = up.reshape(up.shape[0], -1) @ down.reshape(rank, -1)
    return delta.reshape(shape) * scale


def _compose_lokr(factors, multiplier, shape):
    """LoKr: ``multiplier · scale · kron(w1, w2)``. Each factor is a full matrix or
    a low-rank ``a @ b`` product; ``scale = alpha/dim`` applies only when a ``_b``
    factor supplies ``dim`` (full-matrix LoKr fuses at scale 1)."""
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

    # Conv: w2 carries the kernel dims (out2, in2, kh, kw); lift the 2-D w1 so the
    # Kronecker product aligns and yields (out1·out2, in1·in2, kh, kw).
    if w2.dim() == 4 and w1.dim() == 2:
        w1 = w1.reshape(*w1.shape, 1, 1)
    return torch.kron(w1, w2).reshape(shape) * scale


def _build_targets(bundle) -> dict[str, tuple[torch.Tensor, int | None, int | None]]:
    """Map kohya module names to ``(weight, row_start, row_end)``. ``row_start`` is
    ``None`` for whole-weight targets; the slice form addresses bigG's fused
    ``in_proj_weight``."""
    targets: dict[str, tuple[torch.Tensor, int | None, int | None]] = {}

    if getattr(bundle.spec, "architecture", None) == "anima":
        # Anima LoRAs come in two conventions, both targeting the DiT blocks:
        #   - native ComfyUI/musubi: dotted paths under ``diffusion_model.``
        #     (e.g. diffusion_model.blocks.0.self_attn.q_proj), lora_A/lora_B;
        #   - LyCORIS/kohya: underscore-mangled ``lora_unet_blocks_0_self_attn_q_proj``.
        # Register the DiT under both so either file type maps.
        _add_dotted_targets(targets, bundle.backbone, "diffusion_model.")
        _add_module_targets(targets, bundle.backbone, ["lora_unet_"])
        return targets

    _add_module_targets(targets, bundle.backbone, ["lora_unet_"])
    if bundle.text_encoder is not None:
        # lora_te_ is SD1.5's prefix, lora_te1_ is SDXL's CLIP-L prefix; both name
        # the same module structure, so register the CLIP-L encoder under both.
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
    """Register every Linear/Conv2d weight under ``<prefix><dotted_name>`` (no
    underscore-mangling) — the convention Anima/DiT LoRAs use."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            targets[prefix + name] = (module.weight.data, None, None)


def _add_bigg_targets(targets, model, prefix) -> None:
    """OpenCLIP bigG: HF-style kohya keys (split q/k/v, ``text_model.encoder.layers``)
    don't match bigG's module layout (fused ``in_proj_weight``,
    ``transformer.resblocks``), so map them explicitly."""
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


__all__ = ["apply_lora", "LoraReport"]
