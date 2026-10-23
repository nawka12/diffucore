"""MOSS-Image Gate 1 on real weights (manual GPU harness, not collected).

    python tests/_moss_parity.py te        # Qwen3.5-9B INT8 features -> $MOSS_OUT/te_features.pt
    python tests/_moss_parity.py ref       # reference DiT, fp32 CPU, blocks streamed from the bf16 file
    python tests/_moss_parity.py dit       # ours (bf16 / INT8 file, fp16, residual scale) vs the reference
    python tests/_moss_parity.py headroom  # fp16 residual-stream peaks at 1024^2, scale 1 vs 1/16
    python tests/_moss_parity.py e2e [W]   # real loader + pipeline, 3 prompts, small and full VAE

``ref`` runs the reference ``MOSSImageTransformer`` modules at 256^2 in fp32 on the
CPU one block at a time (the fp32 DiT is 20 GB, more than the RAM here); the q/k/v
rows of the file's fused ``proj.qkv`` are split back for it. Run ``te`` first: its
features feed both sides.
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
import _moss_reference as R  # noqa: E402

from diffucore.bundle import (_FLUX_VAE_LEAF, _build_flux_vae, _extract_component, _moss_te_layout,  # noqa: E402
                              _moss_text_encoder, _place, load_moss_checkpoint, load_moss_dit)
from diffucore.conditioning import MossTokenizer  # noqa: E402
from diffucore.loading import load_state_dict  # noqa: E402
from diffucore.pipelines import _moss  # noqa: E402
from diffucore.runtime import ConditioningCache, DevicePolicy, file_backed, fp32_accumulation, stream_layers
from diffucore.runtime import on_device  # noqa: E402

MODELS = Path(__file__).resolve().parents[2] / "models"
DIT_BF16 = MODELS / "diffusion-models" / "moss_image_sft_bf16.safetensors"
DIT_INT8 = MODELS / "diffusion-models" / "moss_image_sft_int8_convrot.safetensors"
TE_INT8 = MODELS / "text-encoders" / "qwen3.5_9b_int8_convrot.safetensors"
VAE_SMALL = MODELS / "vae" / "flux2-vae_small.safetensors"
VAE_FULL = MODELS / "vae" / "flux2-vae.safetensors"
OUT = Path(os.environ.get("MOSS_OUT", "/tmp/moss_gate1"))
DEV = torch.device("cuda")

PROMPTS = {
    "fridge": 'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the '
              'refrigerator door is handwritten in Chinese:\n"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。"',
    "anime": "1girl, silver hair, school uniform, cherry blossoms, looking at viewer, smile, masterpiece",
    "empty": "",
}
E2E = [
    ("fridge", 42, PROMPTS["fridge"]),
    ("bookstore", 1, 'A cozy bookstore window at dusk, a hand-painted wooden sign above the door reads "OPEN LATE", '
                     'warm light, rain on the glass'),
    ("anime", 2, PROMPTS["anime"]),
]
# (feature set, sigma, skipped blocks): the single forwards compared against the reference.
RUNS = [("fridge", 1.0, ()), ("fridge", 0.7, ()), ("fridge", 0.3, ()), ("fridge", 0.05, ()),
        ("fridge", 0.7, (22,)), ("empty", 0.7, ()), ("empty", 0.7, (22,))]


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
    """What the UI builds for Krea 2 / MOSS on this GPU: block streaming with prefetch, INT8 on."""
    return DevicePolicy(device=DEV, compute_dtype=torch.float16, offload="stream", stream_prefetch=True, **kw)


def stage_te():
    OUT.mkdir(parents=True, exist_ok=True)
    policy = stream_policy(int8=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    prefix, hidden = _moss_te_layout(str(TE_INT8))
    te, _ = _moss_text_encoder(str(TE_INT8), prefix, policy)
    te = file_backed(te.eval(), policy.compute_dtype, exclude=(te.layers,))
    stream_layers(te.layers, DEV, policy.compute_dtype)
    print(f"[te] hidden {hidden}, loaded in {time.time() - t0:.1f}s, RSS {rss_gb():.1f} GB", flush=True)
    tok = MossTokenizer()
    feats = {}
    for name, prompt in PROMPTS.items():
        ids = tok(prompt).ids.to(DEV)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), fp32_accumulation(), on_device(te, DEV):
            f = te(ids)
        torch.cuda.synchronize()
        dt = time.time() - t0
        feats[name] = f.cpu()
        print(f"[te] {name:6s} tokens {ids.shape[1]:4d}  features {tuple(f.shape)} {f.dtype}  "
              f"finite {bool(torch.isfinite(f).all())}  max|x| {f.float().abs().max().item():.1f}  "
              f"mean|x| {f.float().abs().mean().item():.3f}  {dt:.2f}s", flush=True)
    torch.save(feats, OUT / "te_features.pt")
    print(f"[te] peak VRAM allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB, peak RSS {rss_gb():.1f} GB", flush=True)


# --- reference (fp32 CPU, blocks streamed from the bf16 file) ---------------------------------

def _load_ref_block(f, i: int, blk) -> None:
    sd = {}
    for k in f.keys():
        if not k.startswith(f"layers.{i}."):
            continue
        name = k[len(f"layers.{i}."):]
        t = f.get_tensor(k).float()
        if name == "proj.qkv.weight":
            for n, part in zip("qkv", t.chunk(3, 0)):
                sd[f"proj.{n}.weight"] = part
        elif name != "comfy_attention.config":
            sd[name] = t
    blk.load_state_dict(sd, strict=True)


def stage_ref():
    feats = torch.load(OUT / "te_features.pt")
    torch.manual_seed(0)
    x = torch.randn(1, 128, 16, 16)
    torch.save(x, OUT / "ref_latent.pt")
    ref = R.MOSSImageTransformer(num_layers=0).float().eval()
    cfg = ref.config
    with safe_open(str(DIT_BF16), "pt") as f:
        non_block = {k: f.get_tensor(k).float() for k in f.keys()
                     if not k.startswith("layers.") and k != "rope.bands"}
        missing, unexpected = ref.load_state_dict(non_block, strict=False)
        assert not unexpected and set(missing) <= {"rope.bands"}, (missing, unexpected)
        h = w = 16
        states = []
        t0 = time.time()
        with torch.no_grad():
            for name, sigma, skip in RUNS:
                ctx = feats[name].float()
                times = torch.tensor([sigma * 1000.0])
                half = cfg.freq_dim // 2
                freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
                phase = times.double()[:, None] * freq
                emb = torch.cat((phase.cos(), phase.sin()), -1).float()
                time_v = ref.time_embedding(emb).float()
                mod = ref.shared_block_modulation(time_v).view(1, 6, -1)
                head = ref.video_head_modulation(time_v).view(1, 2, -1)
                video = ref.video_embedder(x.flatten(2).transpose(1, 2).contiguous())
                rope = ref.rope(h, w, ctx.shape[1], torch.device("cpu")).to(video.dtype)
                text = ref.text_embedder(ctx, rope[:, h * w:])
                states.append(dict(x=torch.cat((video, text), 1), rope=rope, mod=mod, head=head, skip=skip))
            blk = R._Block(cfg.dim, cfg.ffn_dim, cfg.num_heads).float().eval()
            for i in range(42):
                _load_ref_block(f, i, blk)
                for s in states:
                    if i not in s["skip"]:
                        s["x"] = blk(s["x"], s["rope"], s["mod"])
                if i % 6 == 5:
                    print(f"[ref] block {i + 1}/42  {time.time() - t0:.0f}s", flush=True)
            outs = []
            for s in states:
                y = ref.final_norm_video(s["x"][:, :h * w])
                y = y * (1 + s["head"][:, 1, None]) + s["head"][:, 0, None]
                y = ref.final_linear_video(y)
                outs.append(y.transpose(1, 2).reshape(1, -1, h, w))
    torch.save({"runs": RUNS, "out": torch.cat(outs)}, OUT / "ref_out.pt")
    print(f"[ref] done in {time.time() - t0:.0f}s, outputs finite {bool(torch.isfinite(torch.cat(outs)).all())}", flush=True)


def stage_dit():
    feats = torch.load(OUT / "te_features.pt")
    x = torch.load(OUT / "ref_latent.pt").to(DEV)
    ref = torch.load(OUT / "ref_out.pt")
    assert ref["runs"] == RUNS
    variants = [("a bf16 file, fp16, c=1/16", DIT_BF16, None, False),
                ("b INT8 file, fp16, c=1/16", DIT_INT8, None, False),
                ("c bf16 file, fp16, c=1", DIT_BF16, 1.0, False)]
    only = os.environ.get("MOSS_VARIANTS")
    for label, path, scale, _ in variants:
        if only and label[0] not in only:
            continue
        policy = stream_policy()
        t0 = time.time()
        model = load_moss_dit(str(path), policy, masters="file")
        if scale is not None:
            model.residual_scale = scale
        print(f"[dit] === {label}: loaded in {time.time() - t0:.0f}s, residual_scale {model.residual_scale}", flush=True)
        ctx_cache = {}
        with torch.inference_mode():
            for idx, (name, sigma, skip) in enumerate(RUNS):
                if name not in ctx_cache:
                    with fp32_accumulation():
                        ctx_cache[name] = model.encode_text(feats[name].to(DEV))
                mod, head = model.time_vectors(torch.tensor([sigma], device=DEV))
                torch.cuda.synchronize()
                t1 = time.time()
                out = model(x, mod, head, ctx_cache[name], skip).float().cpu()
                torch.cuda.synchronize()
                want = ref["out"][idx:idx + 1]
                print(f"[dit] {label[0]} {name:6s} sigma {sigma:<5} skip {str(skip):6s} rel_l2 {rel_l2(out, want):.5f} "
                      f"cos {cos(out, want):.6f} finite {bool(torch.isfinite(out).all())} {time.time() - t1:.1f}s", flush=True)
        del model, ctx_cache
        torch.cuda.empty_cache()


def stage_headroom():
    feats = torch.load(OUT / "te_features.pt")
    clean = torch.load(OUT / "final_latent_fridge.pt") if (OUT / "final_latent_fridge.pt").exists() else None
    g = torch.Generator().manual_seed(0)
    eps = torch.randn(1, 128, 64, 64, generator=g)
    for label, path in (("bf16 file", DIT_BF16), ("INT8 file", DIT_INT8)):
        policy = stream_policy()
        model = load_moss_dit(str(path), policy, masters="file")
        for scale in (1.0, 0.0625):
            model.residual_scale = scale
            peaks = []
            hooks = [blk.register_forward_hook(lambda m, a, o, i=i: peaks.append((i, float(o.float().abs().max()))))
                     for i, blk in enumerate(model.layers)]
            with torch.inference_mode():
                with fp32_accumulation():
                    ctx = model.encode_text(feats["fridge"].to(DEV))
                cases = [("noise", 1.0, eps)]
                if clean is not None:
                    for s in (1.0, 0.7, 0.3, 0.05):
                        cases.append((f"x0+noise@{s}", s, ((1 - s) * clean + s * eps)))
                for tag, sigma, xin in cases:
                    peaks.clear()
                    mod, head = model.time_vectors(torch.tensor([sigma], device=DEV))
                    out = model(xin.to(DEV), mod, head, ctx).float()
                    i, v = max(peaks, key=lambda p: p[1])
                    print(f"[headroom] {label} c={scale:<7} {tag:16s} peak |stream| {v:10.1f} at block {i:2d} "
                          f"(x{1 / scale:.0f} = {v / scale:9.0f})  output finite {bool(torch.isfinite(out).all())}  "
                          f"first-6 {[round(p[1]) for p in peaks[:6]]}", flush=True)
            for h in hooks:
                h.remove()
        del model
        torch.cuda.empty_cache()


def _psnr(a, b) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return 20 * math.log10(255 / math.sqrt((d ** 2).mean()))


def stage_e2e(side: int = 1024):
    gate = OUT
    gate.mkdir(parents=True, exist_ok=True)
    policy = stream_policy(int8=True)
    torch.cuda.reset_peak_memory_stats()
    with SmiPeak() as smi:
        t0 = time.time()
        bundle = load_moss_checkpoint(str(DIT_INT8), str(TE_INT8), str(VAE_SMALL), policy=policy)
        load_s = time.time() - t0
        bundle.cond_cache = ConditioningCache()
        print(f"[e2e] load {load_s:.1f}s  RSS {rss_gb():.1f} GB  VRAM alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
              flush=True)
        captured = {}
        real_decode = _moss._decode

        def capture(model, x):
            captured["x"] = x.detach().cpu()
            return real_decode(model, x)

        _moss._decode = capture
        for tag, seed, prompt in E2E:
            n = side if tag == "fridge" or not os.environ.get("MOSS_SMALL_REST") else int(os.environ["MOSS_SMALL_REST"])
            last = [time.time()]

            def progress(i, total):
                if i in (0, 1) or i == total:
                    print(f"[e2e]   {tag} step {i}/{total} +{time.time() - last[0]:.1f}s", flush=True)

            torch.cuda.synchronize()
            t0 = time.time()
            img, info = _moss.moss_text_to_image(bundle, prompt, "", seed=seed, width=n, height=n, steps=50,
                                                 cfg_scale=3.0, sampler="euler", scheduler="moss",
                                                 progress_callback=progress, return_info=True)
            torch.cuda.synchronize()
            dt = time.time() - t0
            path = gate / f"{tag}_s{seed}_{n}_small.png"
            img.save(path)
            torch.save(captured["x"], gate / f"final_latent_{tag}.pt")
            print(f"[e2e] {tag} seed {seed} {n}x{n}: {dt:.1f}s  VRAM alloc peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB  "
                  f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB  smi peak {smi.peak} MiB  "
                  f"RSS peak {rss_gb():.1f} GB  decode {info.vae_decode_mode}  -> {path}", flush=True)
            if tag == "fridge":
                # same latent through the full decoder
                small_vae = bundle.vae
                sd = _extract_component(load_state_dict(str(VAE_FULL), device="cpu"), _FLUX_VAE_LEAF)
                vae, sd = _build_flux_vae(sd, bundle.spec)
                bundle.vae = _place(vae, policy.offload_device, policy.vae_dtype, sd).eval()
                full, _ = real_decode(bundle, captured["x"].to(DEV))
                full.save(gate / f"{tag}_s{seed}_{n}_fullvae.png")
                print(f"[e2e] full VAE decode of the same latent: PSNR vs small decoder {_psnr(img, full):.1f} dB -> "
                      f"{gate / f'{tag}_s{seed}_{n}_fullvae.png'}", flush=True)
                bundle.vae = small_vae
                del vae
    print(f"[e2e] done. torch peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB, nvidia-smi peak {smi.peak} MiB, peak RSS {rss_gb():.1f} GB",
          flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "te":
        stage_te()
    elif cmd == "ref":
        stage_ref()
    elif cmd == "dit":
        stage_dit()
    elif cmd == "headroom":
        stage_headroom()
    elif cmd == "e2e":
        stage_e2e(int(sys.argv[2]) if len(sys.argv) > 2 else 1024)
    else:
        raise SystemExit(__doc__)
