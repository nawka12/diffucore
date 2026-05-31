"""Manual perf validation for PR-A (cudnn_benchmark + tf32 + channels_last) and
PR-B (torch.compile). Runs a fixed-prompt/seed generation under each flag
combination, reports wall-clock + visual similarity (PSNR) to the baseline.

Not part of the test suite — requires CUDA + a checkpoint. Run from repo root:
    .venv/bin/python tests/_perf_validation.py --model {sd15,sdxl,anima}
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch

from diffucore import TextToImage, load_anima_checkpoint, load_checkpoint
from diffucore.runtime import DevicePolicy

REPO = Path(__file__).resolve().parents[1]

# Override via env vars: DIFFUCORE_SD15_CKPT, DIFFUCORE_SDXL_CKPT,
# DIFFUCORE_ANIMA_DIT / _VAE / _TE. Defaults point at the in-repo ``models/``
# tree so a clean clone with the SD1.5 checkpoint already there just works.
SD15_CKPT = Path(os.environ.get(
    "DIFFUCORE_SD15_CKPT", REPO / "models" / "v1-5-pruned-emaonly.safetensors"))
SDXL_CKPT = Path(os.environ.get(
    "DIFFUCORE_SDXL_CKPT", REPO / "models" / "sdxl.safetensors"))
ANIMA_DIT = Path(os.environ.get(
    "DIFFUCORE_ANIMA_DIT", REPO / "models" / "anima" / "dit.safetensors"))
ANIMA_VAE = Path(os.environ.get(
    "DIFFUCORE_ANIMA_VAE", REPO / "models" / "anima" / "qwen_image_vae.safetensors"))
ANIMA_TE = Path(os.environ.get(
    "DIFFUCORE_ANIMA_TE", REPO / "models" / "anima" / "qwen3_te.safetensors"))

PROMPT = "a red cube on a wooden table, studio lighting"
NEG = "blurry, low quality"
SEED = 0
STEPS = 20


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _load(model: str, policy: DevicePolicy):
    if model == "sd15":
        return TextToImage(load_checkpoint(str(SD15_CKPT), policy=policy))
    if model == "sdxl":
        return TextToImage(load_checkpoint(str(SDXL_CKPT), policy=policy))
    if model == "anima":
        return TextToImage(load_anima_checkpoint(
            str(ANIMA_DIT), str(ANIMA_VAE), str(ANIMA_TE), policy=policy
        ))
    raise ValueError(f"unknown model {model!r}")


def _generate(pipe, model: str, **kwargs):
    """Call the pipeline with model-appropriate args."""
    if model == "sd15":
        return pipe(PROMPT, NEG, steps=STEPS, seed=SEED, **kwargs)
    if model == "sdxl":
        return pipe(PROMPT, NEG, steps=STEPS, seed=SEED,
                    width=1024, height=1024, **kwargs)
    if model == "anima":
        # Anima native is 1024² with er_sde / flow defaults.
        return pipe(PROMPT, NEG, steps=STEPS, seed=SEED,
                    width=1024, height=1024, cfg_scale=4.0, shift=3.0,
                    sampler="er_sde", **kwargs)
    raise ValueError(f"unknown model {model!r}")


def _run(label: str, model: str, policy: DevicePolicy, warmups: int = 0, measured: int = 1):
    """Load, generate ``warmups + measured`` times, return (array, per-call seconds)."""
    print(f"\n--- {label} ---", flush=True)
    pipe = _load(model, policy)
    try:
        for i in range(warmups):
            t0 = time.perf_counter()
            _generate(pipe, model)
            torch.cuda.synchronize()
            print(f"  warmup {i}: {time.perf_counter() - t0:.2f}s", flush=True)

        times = []
        last_img = None
        for i in range(measured):
            t0 = time.perf_counter()
            last_img = _generate(pipe, model)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"  measured {i}: {dt:.2f}s", flush=True)
        return np.asarray(last_img), times
    finally:
        del pipe
        _free()


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def _cases(model: str, device: torch.device):
    """Per-model case list. SDXL uses encoders-offload to fit a 12 GB card with
    a 6.6 GB checkpoint + compile residency; Anima skips channels_last (DiT is
    pure transformer)."""
    dtype = torch.float16

    def P(**kw):
        return DevicePolicy(device=device, compute_dtype=dtype, **kw)

    if model == "sd15":
        return [
            ("baseline",          P()),
            ("+cudnn_benchmark",  P(cudnn_benchmark=True)),
            ("+tf32",             P(cudnn_benchmark=True, tf32=True)),
            ("+channels_last",    P(cudnn_benchmark=True, tf32=True, channels_last=True)),
            ("+compile (PR-B)",   P(cudnn_benchmark=True, tf32=True, channels_last=True, compile=True)),
        ]
    if model == "sdxl":
        # SDXL 1024² peak is ~10.7 GB resident; on a 12 GB card we still fit
        # without offload, but only just. compile is incompatible with
        # offload='True/full' (raises) — 'encoders' is allowed and pairs well.
        return [
            ("baseline",          P()),
            ("+cudnn_benchmark",  P(cudnn_benchmark=True)),
            ("+tf32",             P(cudnn_benchmark=True, tf32=True)),
            ("+channels_last",    P(cudnn_benchmark=True, tf32=True, channels_last=True)),
            ("+compile (PR-B)",   P(cudnn_benchmark=True, tf32=True, channels_last=True, compile=True)),
        ]
    if model == "anima":
        # DiT is pure transformer — channels_last is meaningless. Skip that case.
        return [
            ("baseline",          P()),
            ("+cudnn_benchmark",  P(cudnn_benchmark=True)),
            ("+tf32",             P(cudnn_benchmark=True, tf32=True)),
            ("+compile (PR-B)",   P(cudnn_benchmark=True, tf32=True, compile=True)),
            ("+cuda_graphs (C)",  P(cudnn_benchmark=True, tf32=True, compile=True, cuda_graphs=True)),
        ]
    raise ValueError(f"unknown model {model!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sd15", "sdxl", "anima"], required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    paths = {
        "sd15": [SD15_CKPT],
        "sdxl": [SDXL_CKPT],
        "anima": [ANIMA_DIT, ANIMA_VAE, ANIMA_TE],
    }[args.model]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"checkpoint not found: {p}")

    print(f"model:      {args.model}")
    print(f"paths:      {[str(p) for p in paths]}")
    print(f"device:     {torch.cuda.get_device_name(0)}")
    print(f"torch:      {torch.__version__}")
    print(f"steps={STEPS}, seed={SEED}, prompt={PROMPT!r}")

    device = torch.device("cuda")
    results = []
    base_img = None
    for label, policy in _cases(args.model, device):
        # cudnn benchmark + compile both need a warmup pass (autotune / specialize).
        warmups = 1 if (policy.cudnn_benchmark or policy.compile) else 0
        img, times = _run(label, args.model, policy, warmups=warmups, measured=1)
        if base_img is None:
            base_img = img
            psnr = float("inf")
        else:
            psnr = _psnr(base_img, img)
        results.append((label, times[0], psnr))

    print(f"\n=== {args.model} summary ===")
    print(f"{'case':<24}{'time (s)':>10}{'speedup':>10}{'PSNR vs base':>16}")
    base_t = results[0][1]
    for label, t, psnr in results:
        speedup = base_t / t if t > 0 else 0
        psnr_s = "inf" if psnr == float("inf") else f"{psnr:.1f} dB"
        print(f"{label:<24}{t:>10.2f}{speedup:>9.2f}x{psnr_s:>16}")


if __name__ == "__main__":
    main()
