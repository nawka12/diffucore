"""Load checkpoints into a ready-to-run :class:`ModelBundle`.

Anima and split-file FLUX arrive as several files (DiT / VAE / text encoders);
their loaders return the same :class:`ModelBundle` so the pipelines can
dispatch on ``spec.architecture``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .conditioning import AnimaTokenizer, CLIPTokenizer, FluxTokenizer, Flux2Tokenizer
from .loading import ModelSpec, detect_architecture, load_state_dict, read_header
from .models import (
    AutoencoderKL, CLIPTextEncoder, OpenCLIPTextEncoder, UNetModel, VAEConfig,
    AnimaDiT, QwenImageVAE, Qwen3TextEncoder, Qwen3Config,
    Qwen35TextEncoder, Qwen35Config,
    Flux, FluxConfig, T5TextEncoder, MistralConfig, MistralTextEncoder,
)
from .models._attention import resolve_attention_backend, set_attention_backend
from .models.unet import sdxl_unet_config
from .runtime import DevicePolicy, maybe_compile_backbone, stream_blocks, to_channels_last
from .sampling import DiscreteSchedule, make_betas

# On-disk prefixes. SDXL keeps CLIP-L under embedders.0 and OpenCLIP bigG under
# embedders.1.
_VAE_PREFIX = "first_stage_model."
_UNET_PREFIX = "model.diffusion_model."
_SD15_CLIP = "cond_stage_model.transformer."
_SDXL_CLIP_L = "conditioner.embedders.0.transformer."
_SDXL_CLIP_G = "conditioner.embedders.1.model."


def _load_sub(module, state_dict, prefix):
    sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    # position_ids is a derived non-persistent buffer; drop it if shipped.
    sub = {k: v for k, v in sub.items() if not k.endswith("position_ids")}
    module.load_state_dict(sub, strict=True)
    return module


@dataclass
class ModelBundle:
    """A loaded model, ready for a pipeline."""

    spec: ModelSpec
    schedule: DiscreteSchedule
    tokenizer: object
    text_encoder: object
    backbone: object
    vae: object
    text_encoder_2: object = None   # SDXL OpenCLIP bigG / FLUX.1 CLIP-L
    policy: DevicePolicy = None      # None -> all-resident
    cond_cache: object = None        # optional runtime.ConditioningCache


def load_checkpoint(
    path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    policy: DevicePolicy | None = None,
) -> ModelBundle:
    """Detect, build and weight-load an SD1.5 / SDXL checkpoint (an all-in-one
    FLUX file is routed to :func:`load_flux_checkpoint`).

    ``policy`` is the placement authority; without one, everything is resident
    in ``dtype`` on ``device``. The VAE runs in ``policy.vae_dtype``. The sigma
    schedule always lives on the compute device.
    """
    if policy is None:
        policy = DevicePolicy(device=torch.device(device), compute_dtype=dtype)

    spec = detect_architecture(read_header(path))
    if spec.architecture in ("flux1", "flux2"):
        return load_flux_checkpoint(path, device=device, dtype=dtype, policy=policy)
    if spec.architecture not in ("sd15", "sdxl"):
        raise NotImplementedError(f"unsupported architecture {spec.architecture!r}")

    # Keep the fp32 sigma table on the compute device, next to the latents.
    schedule = DiscreteSchedule(
        make_betas(spec.beta_schedule, spec.num_train_timesteps),
        zero_terminal_snr=spec.zero_terminal_snr,
    )
    schedule.sigmas = schedule.sigmas.to(policy.device)
    schedule.log_sigmas = schedule.log_sigmas.to(policy.device)

    state_dict = load_state_dict(path, device="cpu")

    # The text encoders + VAE offload in every mode; the UNet only under full.
    idle_target = policy.offload_device if policy.offload_idle else policy.device
    unet_target = policy.offload_device if policy.offload_unet else policy.device

    vae = _load_sub(AutoencoderKL(VAEConfig(scale_factor=spec.latent_scale)), state_dict, _VAE_PREFIX)
    vae = vae.to(idle_target, policy.vae_dtype).eval()

    text_encoder_2 = None
    if spec.architecture == "sd15":
        text_encoder = _load_sub(CLIPTextEncoder(), state_dict, _SD15_CLIP)
        backbone = _load_sub(UNetModel(), state_dict, _UNET_PREFIX)
    else:  # sdxl
        text_encoder = _load_sub(CLIPTextEncoder(), state_dict, _SDXL_CLIP_L)
        text_encoder_2 = _load_sub(OpenCLIPTextEncoder(), state_dict, _SDXL_CLIP_G)
        text_encoder_2 = text_encoder_2.to(idle_target, policy.compute_dtype).eval()
        backbone = _load_sub(UNetModel(sdxl_unet_config()), state_dict, _UNET_PREFIX)

    text_encoder = text_encoder.to(idle_target, policy.compute_dtype).eval()
    if policy.offload_stream:
        # Stream the UNet's blocks (ComfyUI --lowvram analog), so SDXL fits ~4 GB.
        backbone = backbone.to(policy.offload_device, policy.compute_dtype).eval()
        stream_blocks(backbone, ("input_blocks", "middle_block", "output_blocks"),
                      policy.device, policy.offload_device,
                      num_blocks_per_group=policy.stream_blocks_per_group,
                      prefetch=policy.stream_prefetch)
    else:
        backbone = backbone.to(unet_target, policy.compute_dtype).eval()

    # NHWC for the conv backbones when opted in.
    if policy.channels_last:
        backbone = to_channels_last(backbone)
        vae = to_channels_last(vae)

    # Compile after channels_last so Inductor specializes on the final layout.
    backbone = maybe_compile_backbone(backbone, policy)

    return ModelBundle(
        spec=spec,
        schedule=schedule,
        tokenizer=CLIPTokenizer(),
        text_encoder=text_encoder,
        backbone=backbone,
        vae=vae,
        text_encoder_2=text_encoder_2,
        policy=policy,
    )


# Anima DiT keys are bare (``net.*``) in a native export but under
# ``model.diffusion_model.*`` in an all-in-one file; this leaf finds either.
_ANIMA_DIT_LEAF = "llm_adapter.blocks.0.cross_attn.q_proj.weight"


def load_anima_checkpoint(
    dit_path: str,
    vae_path: str,
    te_path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    policy: DevicePolicy | None = None,
) -> ModelBundle:
    """Load Anima's three files (DiT + Qwen-Image VAE + Qwen3 TE) into a
    :class:`ModelBundle`. The VAE runs in ``policy.vae_dtype``, the DiT and text
    encoder in ``dtype``. ``schedule`` is ``None``: flow models build their σ
    schedule at sample time.
    """
    if policy is None:
        policy = DevicePolicy(device=torch.device(device), compute_dtype=dtype)

    # Per-stage progress, so a slow (or stuck) multi-GB load shows where it is.
    _t0 = time.perf_counter()
    def _stage(msg: str) -> None:
        print(f"[load] (+{time.perf_counter() - _t0:.1f}s) {msg}", flush=True)

    # Validate the DiT file is Anima before building the 2B-param module.
    _stage("detecting architecture")
    spec = detect_architecture(read_header(dit_path))
    if spec.architecture != "anima":
        raise ValueError(
            f"{dit_path!r} is not an Anima checkpoint (detected {spec.architecture!r})"
        )

    idle_target = policy.offload_device if policy.offload_idle else policy.device
    unet_target = policy.offload_device if policy.offload_unet else policy.device

    _stage("loading VAE weights")
    vae = QwenImageVAE()
    vae.load_state_dict(load_state_dict(vae_path, device="cpu"), strict=True)
    vae = vae.to(idle_target, policy.vae_dtype).eval()

    # Stock Anima uses Qwen3-0.6B; the experimental cosmos-qwen3.5 swap (4B, or
    # the raw 0.8B base) is a Qwen3.5 hybrid, identified by an SSM
    # ``linear_attn`` block. ``_extract_component`` strips any
    # ``model.language_model.`` prefix and drops the vision / mtp heads. Both
    # emit 1024-d, so the adapter and DiT are unchanged.
    te_sd = load_state_dict(te_path, device="cpu")
    qwen35_sd = _extract_component(te_sd, "embed_tokens.weight")
    is_qwen35 = qwen35_sd is not None and any("linear_attn.A_log" in k for k in qwen35_sd)
    if is_qwen35:
        _stage("loading text encoder (Qwen3.5 hybrid) weights")
        text_encoder = Qwen35TextEncoder(Qwen35Config.from_state_dict(qwen35_sd))
        text_encoder.load_state_dict(qwen35_sd, strict=True)
    else:
        _stage("loading text encoder (Qwen3) weights")
        text_encoder = Qwen3TextEncoder()
        text_encoder.load_state_dict(te_sd, strict=True)
    text_encoder = text_encoder.to(idle_target, policy.compute_dtype).eval()

    # The DiT (with ``llm_adapter``) sits under ``net.*`` or
    # ``model.diffusion_model.*``.
    _stage("loading DiT weights (largest file)")
    sd_dit = _extract_component(load_state_dict(dit_path, device="cpu"), _ANIMA_DIT_LEAF)
    _stage("building DiT backbone (2B params)")
    backbone = AnimaDiT()
    backbone.load_state_dict(sd_dit, strict=True)
    if policy.offload_stream:
        # Keep block 0 resident: TeaCache probes it outside the block's
        # __call__, where the stream hooks don't fire.
        _stage("streaming DiT blocks to GPU (low-VRAM mode)")
        backbone = backbone.to(policy.offload_device, policy.compute_dtype).eval()
        stream_blocks(backbone, ("blocks",), policy.device, policy.offload_device,
                      keep_resident=(backbone.blocks[0],),
                      num_blocks_per_group=policy.stream_blocks_per_group,
                      prefetch=policy.stream_prefetch)
    else:
        backbone = backbone.to(unet_target, policy.compute_dtype).eval()

    # "sdpa" (default) stamps nothing; the modules already use it.
    attn_backend = resolve_attention_backend(policy)
    if attn_backend != "sdpa":
        set_attention_backend(backbone, attn_backend)
        _stage(f"attention backend: {attn_backend}")

    if policy.compile:
        _stage("compiling backbone (torch.compile warmup, may take minutes)")
    backbone = maybe_compile_backbone(backbone, policy)
    _stage("model ready")

    tokenizer = AnimaTokenizer.qwen35() if is_qwen35 else AnimaTokenizer()

    return ModelBundle(
        spec=spec,
        schedule=None,                  # flow: σ schedule built at sample time
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        backbone=backbone,
        vae=vae,
        text_encoder_2=None,
        policy=policy,
    )


# ─── FLUX (FLUX.1 + FLUX.2) ──────────────────────────────────────────────────
# Components come all-in-one or as separate files; each is located by a
# fingerprint leaf key whose prefix is recovered and stripped.

_FLUX_DIT_LEAF = "double_blocks.0.img_attn.qkv.weight"
_FLUX_VAE_LEAF = "decoder.conv_in.weight"
_FLUX_CLIP_LEAF = "text_model.embeddings.token_embedding.weight"
_FLUX_T5_LEAF = "encoder.block.0.layer.0.SelfAttention.q.weight"
_FLUX_MISTRAL_LEAF = "model.layers.0.self_attn.q_proj.weight"


def _flux_arch(architecture: str) -> dict:
    """Per-family FLUX DiT constants the tensor shapes don't reveal. head_dim is
    128 for both, so ``num_heads = hidden // 128``."""
    if architecture == "flux2":
        return dict(
            axes_dim=(32, 32, 32, 32), theta=2000, mlp_ratio=3.0, qkv_bias=False,
            global_modulation=True, mlp_silu_act=True, ops_bias=False,
        )
    return dict(
        axes_dim=(16, 56, 56), theta=10000, mlp_ratio=4.0, qkv_bias=True,
        global_modulation=False, mlp_silu_act=False, ops_bias=True,
    )


def _infer_vae_config(sd, *, scale_factor: float, shift_factor: float) -> VAEConfig:
    """Build a :class:`VAEConfig` from an LDM-style autoencoder state dict
    (FLUX.1: 16-ch / 8×; FLUX.2: 128-ch / 16×), reading depth, widths, res
    blocks, latent channels and quant convs off the weights. Attention is
    assumed mid-only.
    """
    ch = sd["encoder.conv_in.weight"].shape[0]
    num_res = 0
    while f"encoder.down.0.block.{num_res}.norm1.weight" in sd:
        num_res += 1
    n_levels = 0
    while f"encoder.down.{n_levels}.block.0.norm1.weight" in sd:
        n_levels += 1
    mult = []
    for i in range(n_levels):
        j = 0
        while f"encoder.down.{i}.block.{j}.norm1.weight" in sd:
            j += 1
        out_ch = sd[f"encoder.down.{i}.block.{j - 1}.conv2.weight"].shape[0]
        mult.append(out_ch // ch)
    z_channels = sd["encoder.conv_out.weight"].shape[0] // 2  # double_z
    return VAEConfig(
        base_channels=ch,
        channel_mult=tuple(mult),
        num_res_blocks=num_res,
        z_channels=z_channels,
        scale_factor=scale_factor,
        shift_factor=shift_factor,
        # FLUX.2 nests these under encoder./decoder. (see _lift_quant_convs).
        use_quant_conv="quant_conv.weight" in sd or "encoder.quant_conv.weight" in sd,
    )


def _lift_quant_convs(sd):
    """Lift FLUX.2's nested ``encoder.quant_conv.*`` / ``decoder.post_quant_conv.*``
    to the top level :class:`AutoencoderKL` expects. No-op otherwise."""
    moves = (("encoder.quant_conv.", "quant_conv."),
             ("decoder.post_quant_conv.", "post_quant_conv."))
    out = {}
    for k, v in sd.items():
        for old, new in moves:
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def _extract_component(sd, leaf: str):
    """The sub-state-dict of the component whose keys end with ``leaf``, prefix
    stripped, or ``None``."""
    if sd is None:
        return None
    prefix = None
    for k in sd:
        if k.endswith(leaf):
            prefix = k[: -len(leaf)]
            break
    if prefix is None:
        return None
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _load_no_missing(module, sub, drop_suffixes=()):
    """Load ``sub`` into ``module``, tolerating extra keys but rejecting missing
    ones (a real architecture mismatch)."""
    drop = ("position_ids",) + tuple(drop_suffixes)
    sub = {k: v for k, v in sub.items() if not any(k.endswith(s) for s in drop)}
    missing, _ = module.load_state_dict(sub, strict=False)
    if missing:
        raise RuntimeError(
            f"{type(module).__name__}: {len(missing)} missing key(s) "
            f"(e.g. {missing[:5]}), checkpoint/architecture mismatch"
        )
    return module


def _qwen3_config_from_sd(sd) -> Qwen3Config:
    """Derive a Qwen3 config from a checkpoint (head_dim from the q-norm)."""
    emb = sd["model.embed_tokens.weight"]
    n = 0
    while f"model.layers.{n}.self_attn.q_proj.weight" in sd:
        n += 1
    head_dim = sd["model.layers.0.self_attn.q_norm.weight"].shape[0]
    q = sd["model.layers.0.self_attn.q_proj.weight"].shape[0]
    kv = sd["model.layers.0.self_attn.k_proj.weight"].shape[0]
    ffn = sd["model.layers.0.mlp.gate_proj.weight"].shape[0]
    return Qwen3Config(
        vocab_size=emb.shape[0], hidden_size=emb.shape[1], intermediate_size=ffn,
        num_hidden_layers=n, num_attention_heads=q // head_dim,
        num_key_value_heads=kv // head_dim, head_dim=head_dim,
    )


def _build_flux2_text_encoder(lm_sub):
    """Build FLUX.2's text encoder: ``(encoder, tokenizer_kind,
    needs_mistral_tokenizer)``. Qwen3 (Klein) has a per-head ``q_norm``;
    Mistral (Dev) doesn't."""
    if "model.layers.0.self_attn.q_norm.weight" in lm_sub:   # Qwen3 (Klein)
        cfg = _qwen3_config_from_sd(lm_sub)
        encoder = Qwen3TextEncoder(cfg)
        _load_no_missing(encoder, lm_sub, drop_suffixes=("lm_head.weight",))
        kind = "qwen3_8b" if cfg.hidden_size >= 4096 else "qwen3_4b"
        return encoder, kind, False
    cfg = MistralConfig.from_state_dict(lm_sub)              # Mistral (Dev)
    encoder = MistralTextEncoder(cfg)
    _load_no_missing(encoder, lm_sub, drop_suffixes=("lm_head.weight",))
    return encoder, "mistral3_24b", True


def load_flux_checkpoint(
    path: str | None = None,
    *,
    transformer_path: str | None = None,
    vae_path: str | None = None,
    t5_path: str | None = None,
    clip_path: str | None = None,
    mistral_path: str | None = None,
    mistral_tokenizer_path: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    policy: DevicePolicy | None = None,
) -> ModelBundle:
    """Load FLUX.1 or FLUX.2 into a :class:`ModelBundle`.

    Each component comes from its own file when given, else from the all-in-one
    ``path``. FLUX.1 needs transformer + VAE + T5-XXL + CLIP-L; FLUX.2 needs
    transformer + VAE + Qwen3 or Mistral-3 (Mistral also needs a
    ``tokenizer.json``: ``mistral_tokenizer_path`` or a sidecar). The VAE runs
    in ``policy.vae_dtype``; ``schedule`` is ``None``.
    """
    if path is None and transformer_path is None:
        raise ValueError("provide an all-in-one `path` or at least `transformer_path`")
    if policy is None:
        policy = DevicePolicy(device=torch.device(device), compute_dtype=dtype)

    cache: dict[str, dict] = {}

    def source(p):
        p = p or path
        if p is None:
            return None
        if p not in cache:
            cache[p] = load_state_dict(p, device="cpu")
        return cache[p]

    dit_sd_full = source(transformer_path)
    if dit_sd_full is None:
        raise ValueError("no transformer found (need `path` or `transformer_path`)")
    spec = detect_architecture({k: tuple(v.shape) for k, v in dit_sd_full.items()})
    if spec.architecture not in ("flux1", "flux2"):
        raise ValueError(f"not a FLUX checkpoint (detected {spec.architecture!r})")

    idle_target = policy.offload_device if policy.offload_idle else policy.device
    unet_target = policy.offload_device if policy.offload_unet else policy.device

    # ---- transformer: family constants from _flux_arch, widths from shapes
    dit_sub = _extract_component(dit_sd_full, _FLUX_DIT_LEAF)
    hidden = dit_sub["img_in.weight"].shape[0]
    arch_params = _flux_arch(spec.architecture)
    flux_cfg = FluxConfig.from_state_dict(
        dit_sub, num_heads=hidden // sum(arch_params["axes_dim"]), **arch_params
    )
    backbone = Flux(flux_cfg)
    _load_no_missing(backbone, dit_sub)
    if policy.offload_stream:
        # Keep the small modules resident and stream the blocks.
        backbone = backbone.to(policy.offload_device, policy.compute_dtype).eval()
        stream_blocks(backbone, ("double_blocks", "single_blocks"),
                      policy.device, policy.offload_device,
                      num_blocks_per_group=policy.stream_blocks_per_group,
                      prefetch=policy.stream_prefetch)
    else:
        backbone = backbone.to(unet_target, policy.compute_dtype).eval()
    # "sdpa" stamps nothing; the modules already use it.
    attn_backend = resolve_attention_backend(policy)
    if attn_backend != "sdpa":
        set_attention_backend(backbone, attn_backend)
    backbone = maybe_compile_backbone(backbone, policy)

    # ---- VAE: config inferred from the weights, scale/shift from the spec.
    vae_sub = _extract_component(source(vae_path), _FLUX_VAE_LEAF)
    if vae_sub is None:
        raise ValueError("no VAE found (need it in `path` or `vae_path`)")
    vae_sub = _lift_quant_convs(vae_sub)
    vae = AutoencoderKL(_infer_vae_config(
        vae_sub, scale_factor=spec.latent_scale, shift_factor=spec.latent_shift
    ))
    _load_no_missing(vae, vae_sub)
    # FLUX.2 bridges its 32-ch VAE latent to the 128-ch DiT space with a 2×2
    # pixel-shuffle and a non-affine batch norm; keep the bn stats so the
    # pipeline can invert them before decode.
    if spec.architecture == "flux2" and "bn.running_mean" in vae_sub:
        eps = 1e-4
        vae.register_buffer(
            "flux2_latent_mean", vae_sub["bn.running_mean"].float().view(1, -1, 1, 1),
            persistent=False)
        vae.register_buffer(
            "flux2_latent_std", (vae_sub["bn.running_var"].float() + eps).sqrt().view(1, -1, 1, 1),
            persistent=False)
    vae = vae.to(idle_target, policy.vae_dtype).eval()

    # ---- text encoder(s)
    if spec.architecture == "flux1":
        t5_sub = _extract_component(source(t5_path), _FLUX_T5_LEAF)
        if t5_sub is None:
            raise ValueError("FLUX.1 needs a T5-XXL encoder (in `path` or `t5_path`)")
        if "shared.weight" not in t5_sub and "encoder.embed_tokens.weight" in t5_sub:
            t5_sub["shared.weight"] = t5_sub["encoder.embed_tokens.weight"]
        text_encoder = T5TextEncoder()
        _load_no_missing(text_encoder, t5_sub, drop_suffixes=("encoder.embed_tokens.weight",))
        text_encoder = text_encoder.to(idle_target, policy.compute_dtype).eval()

        clip_sub = _extract_component(source(clip_path), _FLUX_CLIP_LEAF)
        if clip_sub is None:
            raise ValueError("FLUX.1 needs a CLIP-L encoder (in `path` or `clip_path`)")
        text_encoder_2 = CLIPTextEncoder()
        _load_no_missing(
            text_encoder_2, clip_sub, drop_suffixes=("text_projection.weight", "logit_scale")
        )
        text_encoder_2 = text_encoder_2.to(idle_target, policy.compute_dtype).eval()
        tokenizer = FluxTokenizer()
    else:  # flux2: one decoder LM (Klein/Qwen3 or Dev/Mistral)
        lm_sub = _extract_component(source(mistral_path), _FLUX_MISTRAL_LEAF)
        if lm_sub is None:
            raise ValueError("FLUX.2 needs a Qwen3 (Klein) or Mistral-3 (Dev) encoder")
        text_encoder, kind, mistral_kind = _build_flux2_text_encoder(lm_sub)
        text_encoder = text_encoder.to(idle_target, policy.compute_dtype).eval()
        text_encoder_2 = None
        if mistral_kind:
            tok_path = mistral_tokenizer_path
            if tok_path is None:
                sidecar = mistral_path or path
                tok_path = str(Path(sidecar).with_name("tokenizer.json")) if sidecar else None
            tokenizer = Flux2Tokenizer(kind=kind, tokenizer_path=tok_path)
        else:
            tokenizer = Flux2Tokenizer(kind=kind)

    return ModelBundle(
        spec=spec,
        schedule=None,                  # flow: σ schedule built at sample time
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        backbone=backbone,
        vae=vae,
        text_encoder_2=text_encoder_2,
        policy=policy,
    )


__all__ = [
    "ModelBundle", "load_checkpoint", "load_anima_checkpoint", "load_flux_checkpoint",
]
