"""FLUX.2 [klein] reference-edit parity against diffusers' Flux2KleinPipeline
(manual, not collected by pytest). The two stacks do not fit in RAM together,
so run them one after the other and let the second compare:

    # 1. diffusers side (a Python env that has diffusers >= 0.42; torch/transformers shared)
    python tests/_flux2_edit_parity.py diffusers OUT_DIR REF.png [--size 512]
    # 2. diffucore side + comparison (needs PYTHONPATH=src)
    python tests/_flux2_edit_parity.py diffucore OUT_DIR REF.png [--size 512]

The diffusers stage loads the transformer / Qwen3 text encoder from the local
single files (the diffusers repo is only read for configs, tokenizer and the
small diffusers-format VAE), runs the reference pipeline once from fixed noise
and a fixed reference image, and writes ``diffusers.pt``: text context,
reference tokens + ids, noise, sigmas, one transformer velocity, the per-step
latents and the final (denormalised, unpatchified) latent. The diffucore stage
reloads the model the way the UI does (offload "stream") and reports rel-L2 of
each quantity, feeding identical inputs where a quantity is a single forward.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = Path(__file__).resolve().parents[2] / "models"
DIT = MODELS / "diffusion-models" / "flux-2-klein-4b.safetensors"
TE = MODELS / "text-encoders" / "qwen_3_4b.safetensors"
VAE = MODELS / "vae" / "flux2-vae.safetensors"
REPO = "black-forest-labs/FLUX.2-klein-4B"
PROMPT = "change her hair to blue and make it night time"
STEPS, SEED = 4, 1234


def rel_l2(a, b) -> float:
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    return float((a - b).norm() / b.norm())


def load_ref(path: str, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# diffusers side
# --------------------------------------------------------------------------- #

def run_diffusers(out: Path, ref_path: str, size: int) -> None:
    import gc

    from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                           Flux2KleinPipeline, Flux2Transformer2DModel)
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    ref = load_ref(ref_path, size)
    saved: dict = {}

    # -- text: the local Comfy-format LM (HF key names), fp16 on the GPU
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(REPO, subfolder="text_encoder")
    sd = {k: v.half() for k, v in load_file(str(TE)).items()}
    te = Qwen3ForCausalLM.from_pretrained(None, config=cfg, state_dict=sd, torch_dtype=torch.float16)
    del sd
    te = te.to(dev).eval()
    with torch.no_grad():
        embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            te, tok, PROMPT, dtype=torch.float16, device=dev)
    saved["prompt_embeds"] = embeds.cpu()
    del te
    gc.collect(); torch.cuda.empty_cache()
    print(f"[diffusers] text {tuple(embeds.shape)} {time.time()-t0:.0f}s", flush=True)

    # -- transformer + VAE + scheduler
    t0 = time.time()
    transformer = Flux2Transformer2DModel.from_single_file(
        str(DIT), config=REPO, subfolder="transformer", torch_dtype=torch.float16).to(dev).eval()
    vae = AutoencoderKLFlux2.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).to(dev)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=tok,
                              transformer=transformer, is_distilled=True)
    print(f"[diffusers] loaded {time.time()-t0:.0f}s", flush=True)

    # -- inputs the pipeline would build (reference preprocessing as in __call__)
    proc = pipe.image_processor.preprocess(ref, height=size, width=size, resize_mode="crop")
    with torch.no_grad():
        ref_tokens, ref_ids = pipe.prepare_image_latents(
            [proc], batch_size=1, generator=None, device=dev, dtype=torch.float32)
    saved["ref_tokens"], saved["ref_ids"] = ref_tokens.cpu(), ref_ids.cpu()
    g = torch.Generator().manual_seed(SEED)
    h = w = size // 16
    noise = torch.randn(1, 128, h, w, generator=g)  # (B, 128, H/16, W/16), fp32
    saved["noise"] = noise

    # -- sigmas / one velocity (the first loop iteration, by hand)
    n_noisy = h * w
    mu = compute_empirical_mu(image_seq_len=n_noisy, num_steps=STEPS)
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
    sch.set_timesteps(sigmas=np.linspace(1.0, 1 / STEPS, STEPS), device=dev, mu=mu)
    saved["sigmas"], saved["mu"] = sch.sigmas.cpu(), mu
    packed = noise.reshape(1, 128, n_noisy).permute(0, 2, 1).to(dev, torch.float16)
    latent_ids = pipe._prepare_latent_ids(noise).to(dev)
    text_ids = pipe._prepare_text_ids(embeds).to(dev)
    with torch.no_grad():
        v0 = transformer(
            hidden_states=torch.cat([packed, ref_tokens], dim=1).to(torch.float16),
            timestep=(sch.timesteps[0] / 1000).expand(1).to(torch.float16), guidance=None,
            encoder_hidden_states=embeds.to(dev), txt_ids=text_ids,
            img_ids=torch.cat([latent_ids, ref_ids], dim=1), return_dict=False)[0][:, :n_noisy]
    saved["v0"] = v0.float().cpu()

    # -- the full reference pipeline from the same noise
    steps_seen = []

    def on_step(_pipe, i, t, kw):
        steps_seen.append(kw["latents"].float().cpu())
        return kw

    t0 = time.time()
    with torch.no_grad():
        latent = pipe(image=[ref], prompt=None, prompt_embeds=embeds, latents=noise, height=size,
                      width=size, num_inference_steps=STEPS, guidance_scale=1.0,
                      output_type="latent", callback_on_step_end=on_step).images
        img = pipe.image_processor.postprocess(pipe.vae.decode(latent.to(dev, torch.float32), return_dict=False)[0])[0]
    print(f"[diffusers] pipeline {time.time()-t0:.0f}s", flush=True)
    saved["step_latents"], saved["final_latent"] = steps_seen, latent.float().cpu()
    img.save(out / "diffusers.png")
    torch.save(saved, out / "diffusers.pt")
    print("[diffusers] saved", out / "diffusers.pt", flush=True)


# --------------------------------------------------------------------------- #
# diffucore side + comparison
# --------------------------------------------------------------------------- #

def run_diffucore(out: Path, ref_path: str, size: int) -> None:
    from unittest import mock

    from diffucore import TextToImage, load_flux_checkpoint
    from diffucore.pipelines import _flux
    from diffucore.runtime import DevicePolicy, staged

    saved = torch.load(out / "diffusers.pt")
    dev = torch.device("cuda")
    ref = load_ref(ref_path, size)
    policy = DevicePolicy(device=dev, compute_dtype=torch.float16, vae_dtype=torch.float32,
                          offload="stream", vae_tile=False, stream_prefetch=True)
    t0 = time.time()
    model = load_flux_checkpoint(
        transformer_path=str(DIT), vae_path=str(VAE), t5_path=str(TE), mistral_path=str(TE),
        policy=policy)
    print(f"[diffucore] loaded {time.time()-t0:.0f}s", flush=True)
    rows = []

    with torch.no_grad():
        # text context (diffucore encodes with its own tokenizer + LM)
        with staged([model.text_encoder], dev, policy.offload_idle):
            ctx, _ = _flux._encode_text(model, PROMPT, dev, torch.float16)
        n = min(ctx.shape[1], saved["prompt_embeds"].shape[1])
        nreal = int((model.tokenizer(PROMPT) != model.tokenizer.pad_id).sum())
        # diffusers masks the pad keys of the LM; diffucore runs it unmasked, so
        # only the real-token rows are expected to agree
        rows.append(("text context (all 512 tokens)", rel_l2(ctx[:, :n], saved["prompt_embeds"][:, :n])))
        rows.append((f"text context ({nreal} real tokens)", rel_l2(ctx[:, :nreal], saved["prompt_embeds"][:, :nreal])))
        rows.append(("text context (pad tokens)", rel_l2(ctx[:, nreal:n], saved["prompt_embeds"][:, nreal:n])))

        # reference tokens + ids
        tokens, ids = _flux._flux2_encode_references(model, [ref], dev, torch.float16)
        rows.append(("reference tokens", rel_l2(tokens, saved["ref_tokens"])))
        ids_equal = torch.equal(ids.cpu(), saved["ref_ids"].float())
        # sigmas
        seq = (size // 16) ** 2
        sig = _flux.flow_matching_schedule(STEPS, shift=float(np.exp(_flux._flux2_empirical_mu(seq, STEPS))))
        rows.append(("sigmas", rel_l2(sig, saved["sigmas"])))

        # one velocity from identical inputs (saved noise / context / ref tokens)
        h = w = size // 16
        noisy = saved["noise"].reshape(1, 128, h * w).permute(0, 2, 1).to(dev, torch.float16)
        img_ids = _flux._img_ids(h, w, 1, 4, dev)
        txt_ids = _flux._txt_ids(saved["prompt_embeds"].shape[1], 4, 3, dev)
        all_ids = torch.cat([img_ids, saved["ref_ids"].to(dev, torch.float32)], dim=1)
        t = torch.full((1,), float(saved["sigmas"][0]), device=dev, dtype=torch.float16)
        v = model.backbone(torch.cat([noisy, saved["ref_tokens"].to(dev, torch.float16)], dim=1),
                           all_ids, saved["prompt_embeds"].to(dev), txt_ids, t, None, None)[:, :h * w]
        rows.append(("velocity (same inputs)", rel_l2(v, saved["v0"])))

    # full pipeline from the saved noise; capture the latent handed to the VAE.
    # Run twice: with diffucore's own text context, and with diffusers' injected
    # (isolating the DiT / reference / schedule path from the text-pad difference).
    real_randn = torch.randn
    h = w = size // 16

    def fake_randn(*shape, **kw):
        if tuple(shape) == (1, 128, h, w):
            return saved["noise"].to(kw.get("device", "cpu"), kw.get("dtype", torch.float32))
        return real_randn(*shape, **kw)

    def pipeline(tag, inject_text):
        captured = {}
        real_decode = model.vae.decode

        def spy_decode(z):
            captured.setdefault("z", z.float().cpu())
            return real_decode(z)

        patches = [mock.patch.object(torch, "randn", fake_randn), mock.patch.object(model.vae, "decode", spy_decode)]
        if inject_text:
            emb = saved["prompt_embeds"].to(dev)
            patches.append(mock.patch.object(_flux, "_encode_text", lambda *a, **k: (emb, None)))
        t0 = time.time()
        with patches[0], patches[1], (patches[2] if inject_text else contextlib.nullcontext()):
            image = TextToImage(model)(PROMPT, steps=STEPS, cfg_scale=1.0, width=size, height=size, seed=SEED,
                                       sampler="euler", scheduler="flux2", reference_images=[ref])
        print(f"[diffucore] pipeline ({tag}) {time.time()-t0:.0f}s", flush=True)
        image.save(out / f"diffucore_{tag}.png")
        rows.append((f"final latent ({tag})", rel_l2(captured["z"], saved["final_latent"])))
        a = np.asarray(image, dtype=np.float32)
        b = np.asarray(Image.open(out / "diffusers.png").convert("RGB"), dtype=np.float32)
        rows.append((f"final image pixels ({tag})", float(np.linalg.norm(a - b) / np.linalg.norm(b))))

    pipeline("own text", False)
    pipeline("diffusers text", True)

    print("\nquantity                           rel-L2")
    for name, val in rows:
        print(f"{name:<34} {val:.3e}")
    print(f"reference ids identical     {ids_equal}")
    print(f"text tokens compared        {n} (diffucore {ctx.shape[1]}, diffusers {saved['prompt_embeds'].shape[1]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["diffusers", "diffucore"])
    ap.add_argument("out_dir")
    ap.add_argument("ref")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        (run_diffusers if a.stage == "diffusers" else run_diffucore)(out, a.ref, a.size)
