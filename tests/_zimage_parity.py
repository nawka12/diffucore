"""Z-Image Gate 1 on real weights (manual GPU harness, not collected).

    python tests/_zimage_parity.py te        # Qwen3-4B features (ours, fp16, GPU) -> $ZIMAGE_OUT/te_features.pt
    python tests/_zimage_parity.py teref     # ours vs transformers: 5 layers fp32 CPU, then all 35 on the GPU
    python tests/_zimage_parity.py ref       # reference DiT, fp32 CPU, blocks streamed from the bf16 file
    python tests/_zimage_parity.py dit       # ours (bf16 / INT8 file, fp16) vs the reference
    python tests/_zimage_parity.py headroom  # fp16 stream / activation peaks at 1024^2
    python tests/_zimage_parity.py e2e       # real loader + pipeline: Turbo, Base, img2img, inpaint, LoRA

``ref`` runs the vendored reference ``ZImageTransformer2DModel`` at 512^2 in fp32 on
the CPU, one main block at a time (the fp32 DiT is 25 GB, more than the RAM here);
the fused ``attention.qkv`` rows of the file are split back for it. Run ``te`` first:
its features feed both sides.
"""

from __future__ import annotations

import math
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _zimage_reference as R  # noqa: E402

from diffucore.bundle import (_zimage_text_encoder, load_zimage_checkpoint, load_zimage_dit)  # noqa: E402
from diffucore.conditioning import ZImageTokenizer  # noqa: E402
from diffucore.pipelines import _zimage  # noqa: E402
from diffucore.runtime import ConditioningCache, DevicePolicy, file_backed, fp32_accumulation, stream_layers  # noqa: E402
from diffucore.runtime import on_device  # noqa: E402

MODELS = Path(__file__).resolve().parents[2] / "models"
DIT_BF16 = MODELS / "diffusion-models" / "z_image_turbo_bf16.safetensors"
DIT_TURBO_INT8 = MODELS / "diffusion-models" / "z_image_turbo_int8_convrot.safetensors"
DIT_BASE_INT8 = MODELS / "diffusion-models" / "z_image_int8_convrot.safetensors"
TE = MODELS / "text-encoders" / "qwen_3_4b.safetensors"
VAE = MODELS / "vae" / "ae.safetensors"
OUT = Path(os.environ.get("ZIMAGE_OUT", "/tmp/zimage_gate1"))
DEV = torch.device("cuda")

PROMPTS = {
    "still": "A still life photograph of a blue ceramic bowl with red apples and lemons on a wooden table, soft "
             "window light from the left, shallow depth of field",
    "cafe": "A young woman with short black hair sits by the window of a small cafe, holding a white coffee cup, "
            "rain on the glass behind her, warm lighting, candid photograph",
    "empty": "",
}
E2E = [
    ("landscape", 11, "A wide landscape photograph of a mountain lake at sunrise, pine forest on the shore, mist over "
                      "the water, snow on the distant peaks, golden light"),
    ("still", 12, PROMPTS["still"]),
    ("cafe", 13, PROMPTS["cafe"]),
    ("poster", 14, 'A vintage travel poster with the bold title "VISIT THE COAST" in white capital letters at the '
                   "top, a lighthouse on a cliff above a blue sea, flat illustration style, limited palette"),
]
# (feature set, sigma): the single forwards compared against the reference.
RUNS = [("still", 1.0), ("still", 0.7), ("still", 0.3), ("still", 0.05), ("empty", 0.7)]


def rel_l2(a, b) -> float:
    a, b = a.float().cpu().flatten(), b.float().cpu().flatten()
    return float((a - b).norm() / b.norm())


def cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().cpu().flatten(), b.float().cpu().flatten(), dim=0))


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


class SmiPeak:
    """Polls nvidia-smi for the GPU's used memory (the whole device, other processes included)."""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                v = int(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
                self.peak = max(self.peak, v)
            except Exception:
                pass
            self._stop.wait(0.4)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def stream_policy(**kw) -> DevicePolicy:
    """What the UI builds for Krea 2 / MOSS on this GPU: block streaming with prefetch."""
    dtype = torch.bfloat16 if os.environ.get("ZIMAGE_DTYPE") == "bf16" else torch.float16
    return DevicePolicy(device=DEV, compute_dtype=dtype, offload="stream", stream_prefetch=True, **kw)


def stage_te():
    OUT.mkdir(parents=True, exist_ok=True)
    policy = stream_policy()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    te, _ = _zimage_text_encoder(str(TE))
    te = file_backed(te.eval(), policy.compute_dtype, exclude=(te.model.layers,))
    stream_layers(te.model.layers, DEV, policy.compute_dtype)
    print(f"[te] {len(te.model.layers)} layers, loaded in {time.time() - t0:.1f}s, RSS {rss_gb():.1f} GB", flush=True)
    tok = ZImageTokenizer()
    feats = {}
    for name, prompt in PROMPTS.items():
        ids = tok(prompt).ids.to(DEV)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), fp32_accumulation(), on_device(te, DEV):
            f = te(ids, hidden_layers=[_zimage.TE_LAYER])[0]
        torch.cuda.synchronize()
        dt = time.time() - t0
        feats[name] = f.cpu()
        print(f"[te] {name:6s} tokens {ids.shape[1]:4d}  features {tuple(f.shape)} {f.dtype}  "
              f"finite {bool(torch.isfinite(f).all())}  max|x| {f.float().abs().max().item():.1f}  "
              f"mean|x| {f.float().abs().mean().item():.3f}  {dt:.2f}s", flush=True)
    torch.save(feats, OUT / "te_features.pt")
    print(f"[te] peak VRAM allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB, peak RSS {rss_gb():.1f} GB", flush=True)


def stage_teref():
    """Ours against transformers' ``Qwen3Model``."""
    from dataclasses import replace
    from transformers import Qwen3Config as HFConfig, Qwen3Model

    from diffucore.models import Qwen3TextEncoder

    hf_cfg = HFConfig.from_pretrained("/home/kayfa/diffucore-weights/hf/Tongyi-MAI/Z-Image-Turbo/text_encoder")
    ids = ZImageTokenizer()(PROMPTS["cafe"]).ids

    # (a) the first 5 layers (+ one more for the HF final norm) in fp32 on the CPU
    n = 6
    only = os.environ.get("ZIMAGE_TEREF", "ab")
    if "a" in only:
        cfg = hf_cfg.to_dict()
        cfg["num_hidden_layers"] = n
        cfg["layer_types"] = cfg.get("layer_types", ["full_attention"] * 36)[:n]
        hf = Qwen3Model(HFConfig(**cfg)).float().eval()
        ours_te, _ = _zimage_text_encoder(str(TE))
        ours = Qwen3TextEncoder(replace(ours_te.cfg, num_hidden_layers=n)).float().eval()
        with safe_open(str(TE), "pt") as f:
            sd = {k[len("model."):]: f.get_tensor(k).float() for k in f.keys()
                  if k.startswith("model.") and (not k.startswith("model.layers.")
                                                  or int(k.split(".")[2]) < n)}
        print(hf.load_state_dict(sd, strict=False), flush=True)
        ours.load_state_dict({"model." + k: v for k, v in sd.items()}, strict=False)
        del ours_te
        with torch.no_grad():
            want = hf(input_ids=ids, output_hidden_states=True).hidden_states
            got = ours(ids, hidden_layers=list(range(1, n)))
        for i, g in enumerate(got, start=1):
            print(f"[teref] fp32 CPU layer {i}: rel L2 {rel_l2(g, want[i]):.2e}  max abs {float((g - want[i]).abs().max()):.2e}",
                  flush=True)
        del hf, ours, sd

    # (b) all 36 layers: transformers in bf16 on the GPU against our fp16 streamed features
    if "b" not in only:
        return
    feats = torch.load(OUT / "te_features.pt")
    cfg = hf_cfg.to_dict()
    with torch.device("meta"):
        hf = Qwen3Model(HFConfig(**cfg)).to(torch.bfloat16).eval()
    with safe_open(str(TE), "pt", device="cuda") as f:
        sd = {k[len("model."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.")}
    print(hf.load_state_dict(sd, strict=False, assign=True), flush=True)
    hf.rotary_emb = type(hf.rotary_emb)(config=hf.config, device=DEV)
    del sd
    for name in ("still", "cafe"):
        tid = ZImageTokenizer()(PROMPTS[name]).ids.to(DEV)
        with torch.no_grad():
            hs = hf(input_ids=tid, output_hidden_states=True).hidden_states
        want = hs[-2].float().cpu()
        got = feats[name].float()
        print(f"[teref] bf16 GPU, all layers, {name}: rel L2 {rel_l2(got, want):.4f} cos {cos(got, want):.6f}  "
              f"max|hf| {float(want.abs().max()):.1f} max|ours| {float(got.abs().max()):.1f}", flush=True)


# --- reference (fp32 CPU, blocks streamed from the bf16 file) ---------------------------------

_RENAME = [
    (".x_embedder.", ".all_x_embedder.2-1."), (".final_layer.", ".all_final_layer.2-1."),
    (".attention.out.", ".attention.to_out.0."), (".attention.q_norm.", ".attention.norm_q."),
    (".attention.k_norm.", ".attention.norm_k."),
]


def _to_reference_names(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if ("." + k).endswith(".attention.qkv.weight"):
            base = k[: -len("qkv.weight")]
            for n, part in zip(("to_q", "to_k", "to_v"), v.chunk(3, 0)):
                out[base + n + ".weight"] = part
            continue
        k = "." + k
        for old, new in _RENAME:
            k = k.replace(old, new, 1)
        out[k[1:]] = v
    return out


def _load_ref_block(f, i: int, blk) -> None:
    prefix = f"layers.{i}."
    sd = {k[len(prefix):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(prefix)}
    blk.load_state_dict(_to_reference_names(sd), strict=True)


def stage_ref():
    feats = torch.load(OUT / "te_features.pt")
    torch.manual_seed(0)
    x = torch.randn(16, 1, 64, 64)                    # 512^2
    torch.save(x, OUT / "ref_latent.pt")
    ref = R.ZImageTransformer2DModel(n_layers=0).float().eval()
    t0 = time.time()
    with safe_open(str(DIT_BF16), "pt") as f:
        non_block = {k: f.get_tensor(k).float() for k in f.keys() if not k.startswith("layers.")}
        print(ref.load_state_dict(_to_reference_names(non_block), strict=True), flush=True)
        states = []
        with torch.no_grad():
            for name, sigma in RUNS:
                t = torch.tensor([1.0 - sigma])
                unified, mask, freqs, adaln, sizes = ref.embed([x], t, [feats[name][0].float()])
                states.append(dict(x=unified, mask=mask, freqs=freqs, adaln=adaln, sizes=sizes))
            blk = R.ZImageTransformerBlock(3840, 30, 1e-5, True).float().eval()
            for i in range(30):
                _load_ref_block(f, i, blk)
                for s in states:
                    s["x"] = blk(s["x"], s["mask"], s["freqs"], s["adaln"])
                if i % 5 == 4:
                    print(f"[ref] block {i + 1}/30  {time.time() - t0:.0f}s  RSS {rss_gb():.1f} GB", flush=True)
            outs = [ref.finish(s["x"], s["adaln"], s["sizes"])[0] for s in states]
    out = torch.stack(outs)[:, :, 0]                  # (runs, 16, 64, 64)
    torch.save({"runs": RUNS, "out": out}, OUT / "ref_out.pt")
    print(f"[ref] done in {time.time() - t0:.0f}s, outputs finite {bool(torch.isfinite(out).all())}", flush=True)


def stage_dit():
    feats = torch.load(OUT / "te_features.pt")
    x = torch.load(OUT / "ref_latent.pt")[None].squeeze(2).to(DEV)
    ref = torch.load(OUT / "ref_out.pt")
    assert ref["runs"] == RUNS
    variants = [("a bf16 file, fp16", DIT_BF16, None),
                ("b INT8 file (turbo), fp16", DIT_TURBO_INT8, None),
                ("c bf16 file, int8 on load", DIT_BF16, "int8")]
    only = os.environ.get("ZIMAGE_VARIANTS")
    for label, path, mode in variants:
        if only and label[0] not in only:
            continue
        policy = stream_policy(int8=(mode == "int8"))
        t0 = time.time()
        model = load_zimage_dit(str(path), policy, masters="file")
        print(f"[dit] === {label}: loaded in {time.time() - t0:.0f}s, attn/ffn scale {model.attn_scale} {model.ffn_scale}", flush=True)
        ctx_cache = {}
        with torch.inference_mode():
            for idx, (name, sigma) in enumerate(RUNS):
                if name not in ctx_cache:
                    with fp32_accumulation():
                        ctx_cache[name] = model.encode_text(feats[name].to(DEV))
                t = torch.tensor([1.0 - sigma], device=DEV)
                torch.cuda.synchronize()
                t1 = time.time()
                out = model(x, t, ctx_cache[name]).float().cpu()
                torch.cuda.synchronize()
                want = ref["out"][idx:idx + 1]
                print(f"[dit] {label[0]} {name:6s} sigma {sigma:<5} rel_l2 {rel_l2(out, want):.5f} "
                      f"cos {cos(out, want):.6f} finite {bool(torch.isfinite(out).all())} {time.time() - t1:.1f}s", flush=True)
        del model, ctx_cache
        torch.cuda.empty_cache()


def _stat(t) -> tuple[float, int]:
    """``(max |x| over the finite elements, number of non-finite elements)``."""
    t = t.detach().float()
    bad = ~torch.isfinite(t)
    return float(t.masked_fill(bad, 0).abs().max()), int(bad.sum())


def stage_headroom():
    feats = torch.load(OUT / "te_features.pt")
    clean = torch.load(OUT / "final_latent_landscape.pt") if (OUT / "final_latent_landscape.pt").exists() else None
    g = torch.Generator().manual_seed(0)
    side = int(os.environ.get("ZIMAGE_RES", 1024)) // 8
    eps = torch.randn(1, 16, side, side, generator=g)
    scales = [tuple(float(v) for v in s.split(":")) for s in os.environ.get("ZIMAGE_SCALES", "1:1").split(",")]
    paths = [("bf16 file", DIT_BF16), ("INT8 file", DIT_TURBO_INT8)]
    if os.environ.get("ZIMAGE_FILES"):
        paths = [p for p in paths if p[0][0] in os.environ["ZIMAGE_FILES"]]
    for label, path in paths:
        policy = stream_policy()
        model = load_zimage_dit(str(path), policy, masters="file")
        for scale in scales:
            model.attn_scale, model.ffn_scale = scale
            events = []
            hooks = []
            for gi, blocks in (("noise_refiner", model.noise_refiner), ("context_refiner", model.context_refiner),
                               ("layers", model.layers)):
                for i, blk in enumerate(blocks):
                    n = f"{gi}.{i}"
                    hooks.append(blk.attention.qkv.register_forward_hook(
                        lambda m, a, o, n=n: events.append((n, "qkv", *_stat(o)))))
                    hooks.append(blk.attention.out.register_forward_pre_hook(
                        lambda m, a, n=n: events.append((n, "attn_out", *_stat(a[0])))))
                    hooks.append(blk.attention.out.register_forward_hook(
                        lambda m, a, o, n=n: events.append((n, "attn_proj", *_stat(o)))))
                    hooks.append(blk.feed_forward.w2.register_forward_pre_hook(
                        lambda m, a, n=n: events.append((n, "ffn_hidden", *_stat(a[0])))))
                    hooks.append(blk.feed_forward.w2.register_forward_hook(
                        lambda m, a, o, n=n: events.append((n, "ffn_out", *_stat(o)))))
                    hooks.append(blk.register_forward_hook(
                        lambda m, a, o, n=n: events.append((n, "stream", *_stat(o)))))
            with torch.inference_mode():
                with fp32_accumulation():
                    ctx = model.encode_text(feats["cafe"].to(DEV))
                cases = [("noise", 1.0, eps)]
                if clean is not None:
                    for s in (1.0, 0.7, 0.3, 0.05):
                        cases.append((f"x0+noise@{s}", s, ((1 - s) * clean + s * eps)))
                for tag, sigma, xin in cases:
                    events.clear()
                    t = torch.tensor([1.0 - sigma], device=DEV)
                    out = model(xin.to(DEV), t, ctx).float()
                    first = next(((n, k) for n, k, _, bad in events if bad), None)
                    parts = []
                    for kind in ("stream", "ffn_hidden", "ffn_out", "qkv", "attn_out", "attn_proj"):
                        n, _, v, _ = max((e for e in events if e[1] == kind), key=lambda e: e[2])
                        parts.append(f"{kind} {v:10.1f}@{n}")
                    for kind in ("attn_proj", "ffn_out", "ffn_hidden", "stream"):
                        top = sorted((e for e in events if e[1] == kind), key=lambda e: -e[2])[:5]
                        print(f"[headroom]   top {kind}: " + ", ".join(f"{e[0]}={e[2]:.0f}" for e in top), flush=True)
                    print(f"[headroom] {label} scales={scale} {tag:16s} first non-finite {first}  "
                          + "  ".join(parts) + f"  output finite {bool(torch.isfinite(out).all())}", flush=True)
            for h in hooks:
                h.remove()
        del model
        torch.cuda.empty_cache()


def stage_trace():
    """Euler loop at 1024^2 from noise (Turbo: 8 steps, shift 3, no CFG; ZIMAGE_FILES=B: Base int8 file,
    50 steps, shift 6, CFG 5 with the empty prompt as negative); reports the first non-finite value."""
    from diffucore.sampling import flow_matching_schedule

    base = os.environ.get("ZIMAGE_FILES", "I") == "B"
    feats = torch.load(OUT / "te_features.pt")
    path = DIT_BASE_INT8 if base else DIT_TURBO_INT8
    steps, shift, cfg = (50, 6.0, 5.0) if base else (8, 3.0, 1.0)
    model = load_zimage_dit(str(path), stream_policy(), masters="file")
    events = []
    for gi, blocks in (("noise_refiner", model.noise_refiner), ("context_refiner", model.context_refiner),
                       ("layers", model.layers)):
        for i, blk in enumerate(blocks):
            n = f"{gi}.{i}"
            blk.feed_forward.w2.register_forward_pre_hook(lambda m, a, n=n: events.append((n, "ffn_hidden", *_stat(a[0]))))
            for kind, mod in (("qkv", blk.attention.qkv), ("attn_proj", blk.attention.out),
                              ("ffn_out", blk.feed_forward.w2), ("block", blk)):
                mod.register_forward_hook(lambda m, a, o, n=n, kind=kind: events.append((n, kind, *_stat(o))))
    sigmas = flow_matching_schedule(steps, shift=shift)
    g = torch.Generator().manual_seed(int(os.environ.get("ZIMAGE_SEED", 0)))
    side = int(os.environ.get("ZIMAGE_RES", 1024)) // 8
    x = torch.randn(1, 16, side, side, generator=g).to(DEV)
    with torch.inference_mode():
        with fp32_accumulation():
            ctx = model.encode_text(feats[os.environ.get("ZIMAGE_PROMPT", "still")].to(DEV))
            unc = model.encode_text(feats["empty"].to(DEV))
        print("[trace] ctx", _stat(ctx), _stat(unc), flush=True)
        for i in range(steps):
            events.clear()
            sigma = float(sigmas[i])
            t = torch.tensor([1.0 - sigma], device=DEV)
            out = model(x, t, ctx).float()
            if cfg != 1.0:
                out_u = model(x, t, unc).float()
                out = out_u + cfg * (out - out_u)
            first = next(((n, k, v) for n, k, v, bad in events if bad), None)
            peaks = {k: max(e[2] for e in events if e[1] == k) for k in ("block", "ffn_hidden", "ffn_out", "attn_proj", "qkv")}
            if os.environ.get("ZIMAGE_TOP") and i % 10 == 0:
                for kind in ("ffn_out", "ffn_hidden", "attn_proj", "qkv", "block"):
                    top = sorted((e for e in events if e[1] == kind), key=lambda e: -e[2])[:4]
                    print(f"[trace]   top {kind}: " + ", ".join(f"{e[0]}={e[2]:.0f}" for e in top), flush=True)
            if first is not None or i % 5 == 0 or i == steps - 1:
                print(f"[trace] step {i} sigma {sigma:.4f} out {_stat(out)} x {_stat(x)} first bad {first} peaks {peaks}", flush=True)
            if first is not None:
                break
            x = x + (sigmas[i + 1] - sigma) * (-out)


def stage_lora():
    """A real community LoRA (ZIMAGE_LORA) on the Turbo INT8 model: hooked count, effect, exact removal."""
    from diffucore.lora import apply_lora, clear_loras

    OUT.mkdir(parents=True, exist_ok=True)
    path = os.environ["ZIMAGE_LORA"]
    bundle = load_zimage_checkpoint(str(DIT_TURBO_INT8), str(TE), str(VAE), policy=stream_policy(int8=True))
    prompt = "pixel art style, a lighthouse on a cliff above the sea at sunset, seagulls"
    gen = lambda: _zimage.zimage_text_to_image(bundle, prompt, seed=5, width=1024, height=1024)  # noqa: E731
    a = gen()
    a.save(OUT / "lora_off.png")
    t0 = time.time()
    report = apply_lora(bundle, path, 1.0)
    print(f"[lora] applied {report.applied}, unmapped {len(report.unmatched)} in {time.time() - t0:.1f}s", flush=True)
    b = gen()
    b.save(OUT / "lora_on.png")
    clear_loras(bundle)
    c = gen()
    print(f"[lora] on differs from off: {not np.array_equal(np.asarray(a), np.asarray(b))}; "
          f"after clear identical to off: {np.array_equal(np.asarray(a), np.asarray(c))}; "
          f"PSNR on vs off {_psnr(a, b):.1f} dB", flush=True)


def _psnr(a, b) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return 20 * math.log10(255 / math.sqrt((d ** 2).mean()))


def _run_e2e(tag, dit, variant_note, prompts, policy, side=1024, **gen):
    bundle = load_zimage_checkpoint(str(dit), str(TE), str(VAE), policy=policy)
    bundle.cond_cache = ConditioningCache()
    captured = {}
    real_decode = _zimage._decode

    def capture(model, x):
        captured["x"] = x.detach().cpu()
        return real_decode(model, x)

    _zimage._decode = capture
    try:
        for name, seed, prompt in prompts:
            last = [time.time()]

            def progress(i, total, name=name):
                if i in (0, 1) or i == total:
                    print(f"[e2e]   {name} step {i}/{total} +{time.time() - last[0]:.1f}s", flush=True)

            torch.cuda.synchronize()
            t0 = time.time()
            img, info = _zimage.zimage_text_to_image(bundle, prompt, gen.get("negative", ""), seed=seed, width=side,
                                                     height=side, steps=gen.get("steps"), cfg_scale=gen.get("cfg"),
                                                     progress_callback=progress, return_info=True)
            torch.cuda.synchronize()
            dt = time.time() - t0
            path = OUT / f"{tag}_{name}_s{seed}_{side}.png"
            img.save(path)
            torch.save(captured["x"], OUT / f"final_latent_{name}.pt")
            print(f"[e2e] {tag} {name} seed {seed} {side}x{side}: {dt:.1f}s  VRAM alloc peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB  RSS peak {rss_gb():.1f} GB  decode "
                  f"{info.vae_decode_mode}  -> {path}", flush=True)
    finally:
        _zimage._decode = real_decode
    return bundle


def stage_e2e():
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    which = os.environ.get("ZIMAGE_E2E", "turbo,base,i2i,lora").split(",")
    with SmiPeak() as smi:
        if "turbo" in which:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            policy = stream_policy(int8=True)
            print("[e2e] === Turbo, INT8 file, defaults (8 steps, no CFG)", flush=True)
            bundle = _run_e2e("turbo", DIT_TURBO_INT8, "", E2E, policy)
            print(f"[e2e] turbo total (incl. load) {time.time() - t0:.1f}s  smi peak {smi.peak} MiB", flush=True)
            if "i2i" in which:
                src = Image.open(OUT / "turbo_still_s12_1024.png").convert("RGB")
                torch.cuda.synchronize()
                t0 = time.time()
                img = _zimage.zimage_img2img(bundle, "A still life photograph of a blue ceramic bowl with green pears "
                                             "and oranges on a wooden table, soft window light, shallow depth of field",
                                             src, strength=0.6, seed=21, width=1024, height=1024)
                img.save(OUT / "turbo_img2img_s21_1024.png")
                print(f"[e2e] img2img strength 0.6, 8 steps: {time.time() - t0:.1f}s", flush=True)
                mask = Image.new("L", (1024, 1024), 0)
                mask.paste(255, (512, 256, 1024, 768))
                t0 = time.time()
                img = _zimage.zimage_img2img(bundle, "A still life photograph of a blue ceramic bowl with red apples and "
                                             "a silver teapot on a wooden table, soft window light",
                                             src, mask_image=mask, strength=1.0, seed=22, width=1024, height=1024)
                img.save(OUT / "turbo_inpaint_s22_1024.png")
                keep = np.asarray(mask) < 128
                same = np.array_equal(np.asarray(src)[keep], np.asarray(img)[keep])
                print(f"[e2e] inpaint (right half-ish box), 8 steps: {time.time() - t0:.1f}s  keep region identical {same}",
                      flush=True)
            del bundle
            torch.cuda.empty_cache()
        if "base" in which:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            policy = stream_policy(int8=True)
            print("[e2e] === Base, INT8 file, defaults (50 steps, CFG 5, empty negative)", flush=True)
            bundle = _run_e2e("base", DIT_BASE_INT8, "", E2E[:3], policy)
            print(f"[e2e] base total (incl. load) {time.time() - t0:.1f}s  smi peak {smi.peak} MiB", flush=True)
            del bundle
            torch.cuda.empty_cache()
        for tag, int8 in (("bf16", False), ("bf16int8", True)):
            if tag in which:
                torch.cuda.reset_peak_memory_stats()
                print(f"[e2e] === Turbo, bf16 file, fp16 compute, int8 on load: {int8}", flush=True)
                t0 = time.time()
                bundle = _run_e2e(tag, DIT_BF16, "", E2E[:1], stream_policy(int8=int8))
                print(f"[e2e] {tag} total (incl. load) {time.time() - t0:.1f}s  smi peak {smi.peak} MiB", flush=True)
                del bundle
                torch.cuda.empty_cache()
    print(f"[e2e] done. nvidia-smi peak {smi.peak} MiB, peak RSS {rss_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    stages = dict(te=stage_te, teref=stage_teref, ref=stage_ref, dit=stage_dit, headroom=stage_headroom, trace=stage_trace, lora=stage_lora, e2e=stage_e2e)
    if cmd not in stages:
        raise SystemExit(__doc__)
    stages[cmd]()
