"""Device / dtype policy. [SKELETON]

A single place that decides where modules live and in what precision, so model
code never hardcodes ``.cuda()`` or a dtype. See ``docs/IMPLEMENTATION_SPEC.md``
§Runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DevicePolicy:
    """Resolved placement for a run.

    Defaults target the RTX 2060: fp16 weights on CUDA, fp32 for the VAE and the
    sigma math. ``offload`` moves idle submodules to CPU RAM between stages.
    """

    device: torch.device
    compute_dtype: torch.dtype = torch.float16
    vae_dtype: torch.dtype = torch.float32
    offload: bool = False

    @classmethod
    def auto(cls) -> "DevicePolicy":
        if torch.cuda.is_available():
            return cls(device=torch.device("cuda"), compute_dtype=torch.float16)
        # CPU fallback (testing only): fp16 is unsupported on most CPUs.
        return cls(device=torch.device("cpu"), compute_dtype=torch.float32)


__all__ = ["DevicePolicy"]
