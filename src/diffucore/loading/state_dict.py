"""Checkpoint I/O.

Only ``.safetensors`` is supported: it is the safe, standard distribution format
(pickle-based ``.ckpt`` files can execute arbitrary code on load and are
intentionally not accepted). :func:`read_header` reports tensor shapes without
materializing the weights, which is what architecture detection needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def _check_safetensors(path: str) -> None:
    if not path.endswith(".safetensors"):
        raise ValueError(f"unsupported checkpoint format: {path!r} (only .safetensors is accepted)")


def load_state_dict(path: str | Path, device: str = "cpu",
                    keep: Callable[[str], bool] | None = None) -> dict[str, torch.Tensor]:
    """Load the tensors of a ``.safetensors`` file into a plain dict: all of
    them, or only the keys ``keep`` accepts (the rest are never read)."""
    path = str(path)
    _check_safetensors(path)
    if keep is None:
        return load_file(path, device=device)
    with safe_open(path, framework="pt", device=device) as f:
        return {key: f.get_tensor(key) for key in f.keys() if keep(key)}


def read_metadata(path: str | Path) -> dict[str, str]:
    """The ``__metadata__`` strings of a ``.safetensors`` file (empty if none)."""
    path = str(path)
    _check_safetensors(path)
    with safe_open(path, framework="pt", device="cpu") as f:
        return dict(f.metadata() or {})


def read_header(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Return ``{key: shape}`` for every tensor, without loading the data."""
    path = str(path)
    _check_safetensors(path)
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            shapes[key] = tuple(f.get_slice(key).get_shape())
    return shapes
