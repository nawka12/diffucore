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

Krea 2 is not fused: its streamed blocks may live on the checkpoint's
read-only mmap, and a full-coverage snapshot would be ~12 GB. Each LoRA adds a
forward hook per layer, ``out += scale · up(down(x))``, with the factors on
the GPU; keys are ``diffusion_model.<path>``, kohya-mangled, or diffusers'
``transformer.<path>``. LoKr is not supported there.

MOSS-Image is hooked the same way. Its q/k/v LoRAs (separate in every file)
add into the row thirds of the fused ``proj.qkv`` output; the text embedder's
``text_embedder.blocks.N.*`` layers are targets too. ``alpha`` comes from the
``.alpha`` tensor, else the file's metadata (``alpha``, or PEFT's
``lora_adapter_metadata``), else the rank. A LoRA that carries a weight of a
different shape (a control adapter) is rejected.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loading import load_state_dict, read_metadata
from .models._int8 import Int8Linear, lora_seed
from .models.krea2_dit import from_diffusers_name

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

# Anima-2.9B is anima-base with 12 blocks interleaved at these indices; the
# other 28 are base's blocks in order, bit-identical (CircleStone's 28->40
# expand manifest, verified against the weights). A 28-block LoRA maps onto
# those; the inserted blocks get no delta.
_ANIMA_INSERTED_BLOCKS = {40: (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)}
_ANIMA_BASE_BLOCKS = 28
_ANIMA_BLOCK_KEY = re.compile(r"^(diffusion_model\.blocks\.|lora_unet_blocks_)(\d+)(?=[._])")


@dataclass
class LoraReport:
    """Outcome of :func:`apply_lora`: modules fused, and LoRA module names with no
    target (unsupported variants or a wrong-arch file). ``remapped`` is set when
    a 28-block Anima LoRA was mapped onto an expanded (e.g. 2.9B) model."""

    applied: int
    unmatched: list[str]
    remapped: bool = False


def apply_lora(bundle, path: str, multiplier: float = 1.0) -> LoraReport:
    """Fuse the LoRA at ``path`` into ``bundle`` scaled by ``multiplier``, in
    place. LoRAs stack; each touched weight is snapshotted to CPU first so
    :func:`remove_lora` / :func:`clear_loras` can undo it. Check the report's
    ``unmatched`` for keys that mapped to no layer.
    """
    state = _lora_state(bundle)
    if _is_hooked(bundle):
        attach = _moss_attach if _is_moss(bundle) else _krea2_attach
        report, handles = attach(bundle, path, multiplier)
        state["stack"].append((path, multiplier))
        state.setdefault("hooks", []).append(handles)
        return report
    targets = _build_targets(bundle)
    report = _fuse(path, multiplier, targets, state["base"], _anima_depth(bundle))
    state["stack"].append((path, multiplier))
    return report


def remove_lora(bundle, path: str) -> None:
    """Remove the most recent application of ``path`` and re-fuse the rest, as
    if it had never been applied. Raises ``ValueError`` if it isn't applied."""
    state = _lora_state(bundle)
    stack = state["stack"]
    for i in range(len(stack) - 1, -1, -1):
        if stack[i][0] == path:
            del stack[i]
            break
    else:
        raise ValueError(f"{path!r} is not in the active LoRA stack")
    if _is_hooked(bundle):
        for h in state["hooks"].pop(i):
            h.remove()
        _krea2_invalidate(bundle)
        return
    _refuse(bundle)


def clear_loras(bundle) -> None:
    """Remove all LoRAs, restoring the base weights. No-op if none are applied."""
    state = _lora_state(bundle)
    hooks = state.pop("hooks", [])
    for handles in hooks:
        for h in handles:
            h.remove()
    if hooks:
        _krea2_invalidate(bundle)
    _restore(state["base"])
    state["base"].clear()
    state["stack"].clear()


def _fuse(path: str, multiplier: float, targets, base, anima_depth: int | None = None) -> LoraReport:
    """Add the LoRA's scaled deltas into ``targets``, snapshotting each weight
    into ``base`` before its first modification."""
    groups = _group(load_state_dict(path, device="cpu"))
    remapped = False
    if anima_depth in _ANIMA_INSERTED_BLOCKS:
        groups, remapped = _remap_anima_blocks(groups, anima_depth)
    applied, unmatched = 0, []
    for name, factors in groups.items():
        target = targets.get(name)
        if target is None:
            unmatched.append(name)          # no such layer (e.g. wrong-arch file)
            continue
        weight, row_start, row_end = target
        int8 = weight if isinstance(weight, Int8Linear) else None
        if int8 is not None:
            weight = int8.weight
        view = weight if row_start is None else weight[row_start:row_end]
        # Compose on the weight's device: the fp32 up@down / kron is ~4x slower
        # on the CPU, plus an fp32 host-to-device copy of every full delta.
        factors = {k: v if k == "alpha" else v.to(view.device) for k, v in factors.items()}
        delta = _compose(factors, multiplier, view.shape)
        if delta is None:
            unmatched.append(name)          # unsupported factorization (e.g. Tucker)
            continue
        _snapshot(base, weight)
        if int8 is not None:
            _snapshot(base, int8.scale_bits)
            int8.add_delta_(delta, lora_seed(name, path))
        else:
            with torch.no_grad():
                view.add_(delta.to(view.dtype))
        applied += 1

    return LoraReport(applied=applied, unmatched=unmatched, remapped=remapped)


def _anima_depth(bundle) -> int | None:
    """Block count of an Anima backbone, else ``None``."""
    if getattr(bundle.spec, "architecture", None) != "anima":
        return None
    return len(_eager(bundle.backbone).blocks)


def _remap_anima_blocks(groups, depth: int):
    """Rename a 28-block Anima LoRA's block indices onto the base blocks'
    positions in a ``depth``-block expansion. A LoRA that already reaches past
    block 27 is native to the expanded model and passes through."""
    indices = [int(m.group(2)) for k in groups if (m := _ANIMA_BLOCK_KEY.match(k))]
    if not indices or max(indices) >= _ANIMA_BASE_BLOCKS:
        return groups, False
    inserted = _ANIMA_INSERTED_BLOCKS[depth]
    positions = [i for i in range(depth) if i not in inserted]
    rename = lambda m: f"{m.group(1)}{positions[int(m.group(2))]}"
    return {_ANIMA_BLOCK_KEY.sub(rename, k): v for k, v in groups.items()}, True


def _refuse(bundle) -> None:
    """Restore the base weights, then replay the active stack from scratch."""
    state = _lora_state(bundle)
    _restore(state["base"])
    targets = _build_targets(bundle)
    depth = _anima_depth(bundle)
    for path, multiplier in state["stack"]:
        _fuse(path, multiplier, targets, state["base"], depth)


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


def _target_weight(module):
    """A layer's fusable weight; an :class:`Int8Linear` is its own target
    (re-quantized on fuse)."""
    return module if isinstance(module, Int8Linear) else module.weight.data


def _add_module_targets(targets, model, prefixes) -> None:
    """Register every Linear/Conv2d weight under ``<prefix><dotted_name_as_>``."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, Int8Linear)):
            mangled = name.replace(".", "_")
            for prefix in prefixes:
                targets[prefix + mangled] = (_target_weight(module), None, None)


def _add_dotted_targets(targets, model, prefix) -> None:
    """Register every Linear/Conv2d weight under ``<prefix><dotted_name>``, the
    Anima/DiT convention."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, Int8Linear)):
            targets[prefix + name] = (_target_weight(module), None, None)


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


def _is_krea2(bundle) -> bool:
    return getattr(bundle.spec, "architecture", None) == "krea2"


def _is_moss(bundle) -> bool:
    return getattr(bundle.spec, "architecture", None) == "moss_image"


def _is_hooked(bundle) -> bool:
    """Families whose LoRAs are forward hooks rather than fused weights."""
    return _is_krea2(bundle) or _is_moss(bundle)


def _krea2_invalidate(bundle) -> None:
    """The cached text-stream tokens went through LoRA-able layers."""
    if getattr(bundle, "cond_cache", None) is not None:
        bundle.cond_cache.clear()


class _LoraBranch:
    """Forward hook adding ``x @ down.T @ (scale · up).T`` to a layer's output."""

    def __init__(self, down: torch.Tensor, up: torch.Tensor):
        self.down, self.up = down, up

    def __call__(self, module, args, out):
        x = args[0]
        return out + F.linear(F.linear(x.to(self.down.dtype), self.down), self.up).to(out.dtype)


def _krea2_targets(backbone) -> dict:
    targets = {}
    for name, module in backbone.named_modules():
        if isinstance(module, (nn.Linear, Int8Linear)):
            targets["diffusion_model." + name] = module
            targets["lora_unet_" + name.replace(".", "_")] = module
    return targets


def _krea2_attach(bundle, path: str, multiplier: float):
    """Hook each LoRA module onto its Krea 2 layer; ``(report, hook handles)``."""
    policy = getattr(bundle, "policy", None)
    device = policy.device if policy is not None else _CPU
    dtype = policy.compute_dtype if policy is not None else torch.float32
    targets = _krea2_targets(_eager(bundle.backbone))
    handles, unmatched = [], []
    for name, factors in _group(load_state_dict(path, device="cpu")).items():
        key = name
        if key.startswith("transformer."):
            key = "diffusion_model." + from_diffusers_name(key[len("transformer."):])
        module = targets.get(key)
        if module is None or "down" not in factors or "up" not in factors:
            unmatched.append(name)
            continue
        down, up = factors["down"].float(), factors["up"].float()
        rank = down.shape[0]
        if down.shape[1] != module.in_features or up.shape[0] != module.out_features:
            unmatched.append(name)
            continue
        alpha = factors["alpha"].item() if "alpha" in factors else float(rank)
        up = up * (alpha / rank * multiplier)
        handles.append(module.register_forward_hook(
            _LoraBranch(down.to(device, dtype), up.to(device, dtype))))
    _krea2_invalidate(bundle)
    return LoraReport(applied=len(handles), unmatched=unmatched), handles


class _LoraSlices:
    """Forward hook adding several low-rank branches into row slices of one
    layer's output (the three q/k/v LoRAs on the fused ``proj.qkv``)."""

    def __init__(self, branches):
        self.branches = branches            # [(row_start, row_end, down, up)]

    def __call__(self, module, args, out):
        x = args[0]
        for lo, hi, down, up in self.branches:
            out[..., lo:hi] += F.linear(F.linear(x.to(down.dtype), down), up).to(out.dtype)
        return out


def _moss_targets(backbone) -> dict:
    """LoRA module name -> ``(layer, (row_start, row_end) or None)``, under the
    bare, ``diffusion_model.`` and ``transformer.`` dotted names and kohya's
    ``lora_unet_`` mangling. ``proj.q/k/v`` are the row thirds of ``proj.qkv``."""
    targets = {}
    for name, module in backbone.named_modules():
        if not isinstance(module, (nn.Linear, Int8Linear)):
            continue
        entries = {name: (module, None)}
        if name.endswith(".proj.qkv"):
            third = module.out_features // 3
            for i, c in enumerate("qkv"):
                entries[name[:-3] + c] = (module, (i * third, (i + 1) * third))
        for key, target in entries.items():
            for prefix in ("", "diffusion_model.", "transformer."):
                targets[prefix + key] = target
            targets["lora_unet_" + key.replace(".", "_")] = target
    return targets


def _moss_metadata_alpha(path: str) -> float | None:
    """The file-wide alpha: safetensors metadata ``alpha`` (kohya / trainers) or
    ``lora_alpha`` of PEFT's ``lora_adapter_metadata``."""
    meta = read_metadata(path)
    try:
        if "alpha" in meta:
            return float(meta["alpha"])
        if "lora_adapter_metadata" in meta:
            return float(json.loads(meta["lora_adapter_metadata"])["lora_alpha"])
    except (ValueError, KeyError, TypeError):
        pass
    return None


def _moss_attach(bundle, path: str, multiplier: float):
    """Hook each LoRA module onto its MOSS-Image layer; ``(report, hook handles)``."""
    policy = getattr(bundle, "policy", None)
    device = policy.device if policy is not None else _CPU
    dtype = policy.compute_dtype if policy is not None else torch.float32
    targets = _moss_targets(_eager(bundle.backbone))
    state_dict = load_state_dict(path, device="cpu")
    file_alpha = _moss_metadata_alpha(path)
    unmatched = [k for k in state_dict
                 if not any(k.endswith(suffix) for suffix, _ in _SUFFIXES)]
    for key in unmatched:
        # A full weight / bias riding along (a control adapter such as the
        # color-depth one widens video_embedder) can't be a side branch.
        base, _, leaf = key.rpartition(".")
        target = targets.get(base)
        if target is not None and leaf in ("weight", "bias"):
            have = getattr(target[0], leaf)
            if have is not None and tuple(have.shape) != tuple(state_dict[key].shape):
                raise ValueError(
                    f"this LoRA changes the shape of {base}.{leaf} {tuple(have.shape)} -> "
                    f"{tuple(state_dict[key].shape)} (a control / adapter LoRA); only plain LoRAs "
                    f"are supported for MOSS-Image")

    per_layer: dict = {}
    for name, factors in _group(state_dict).items():
        target = targets.get(name)
        if target is None or "down" not in factors or "up" not in factors:
            unmatched.append(name)
            continue
        module, rows = target
        down, up = factors["down"].float(), factors["up"].float()
        rank = down.shape[0]
        out_features = module.out_features if rows is None else rows[1] - rows[0]
        if down.shape[1] != module.in_features or up.shape[0] != out_features or up.shape[1] != rank:
            raise ValueError(
                f"this LoRA's {name} is {tuple(down.shape)} / {tuple(up.shape)}, which doesn't fit the "
                f"layer ({module.in_features} -> {out_features})")
        alpha = factors["alpha"].item() if "alpha" in factors else (
            file_alpha if file_alpha is not None else float(rank))
        up = up * (alpha / rank * multiplier)
        per_layer.setdefault(id(module), (module, []))[1].append(
            (rows, down.to(device, dtype), up.to(device, dtype)))

    handles = []
    for module, branches in per_layer.values():
        for rows, down, up in branches:
            if rows is None:
                handles.append(module.register_forward_hook(_LoraBranch(down, up)))
        sliced = [(*rows, down, up) for rows, down, up in branches if rows is not None]
        if sliced:
            handles.append(module.register_forward_hook(_LoraSlices(sliced)))
    applied = sum(len(b) for _, b in per_layer.values())
    _krea2_invalidate(bundle)
    print(f"[lora] MOSS-Image: {applied} modules hooked, {len(unmatched)} unmapped", flush=True)
    return LoraReport(applied=applied, unmatched=unmatched), handles


__all__ = ["apply_lora", "remove_lora", "clear_loras", "LoraReport"]
