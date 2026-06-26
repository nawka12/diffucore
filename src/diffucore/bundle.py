"""Loading a checkpoint into a ready-to-run :class:`ModelBundle`.

Architecture detection (implemented, M3) runs here; building the modules and
loading their weights is the M4–M6 work described in
``docs/IMPLEMENTATION_SPEC.md``.

Anima checkpoints ship as three separate files (DiT / VAE / Qwen3 TE) rather
than one bundled safetensors; the :func:`load_anima_checkpoint` entrypoint
takes all three paths and returns the same :class:`ModelBundle` type as
:func:`load_checkpoint` so the pipelines can dispatch by ``spec.architecture``.
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
from .models.unet import sdxl_unet_config
from .runtime import DevicePolicy, maybe_compile_backbone, stream_blocks, to_channels_last
from .sampling import DiscreteSchedule, make_betas

# On-disk prefixes (minus the top-level architecture prefix). SDXL keeps CLIP-L
# under embedders.0 and adds OpenCLIP bigG under embedders.1.
_VAE_PREFIX = "first_stage_model."
_UNET_PREFIX = "model.diffusion_model."
_SD15_CLIP = "cond_stage_model.transformer."
_SDXL_CLIP_L = "conditioner.embedders.0.transformer."
_SDXL_CLIP_G = "conditioner.embedders.1.model."


def _load_sub(module, state_dict, prefix):
    sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    # position_ids is a derived constant (non-persistent buffer); drop it if a
    # checkpoint ships one so the strict load neither misses nor rejects it.
    sub = {k: v for k, v in sub.items() if not k.endswith("position_ids")}
    module.load_state_dict(sub, strict=True)
    return module


@dataclass
class ModelBundle:
    """A loaded model, ready for a pipeline."""

    spec: ModelSpec
    schedule: DiscreteSchedule
    tokenizer: object               # conditioning.CLIPTokenizer
    text_encoder: object            # CLIPTextEncoder (CLIP-L)
    backbone: object                # models.UNetModel
    vae: object                     # models.AutoencoderKL
    text_encoder_2: object = None   # SDXL only: OpenCLIPTextEncoder (bigG)
    policy: DevicePolicy = None      # placement authority; None -> all-resident


def load_checkpoint(
    path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    policy: DevicePolicy | None = None,
) -> ModelBundle:
    """Detect, build, and weight-load a checkpoint into a :class:`ModelBundle`.

    Supports SD1.5 and SDXL. Text encoder(s) and UNet run in ``dtype`` (fp16 on
    CUDA); the VAE stays fp32 (fp16 decode produces artifacts/NaNs).

    ``policy`` is the single placement authority. When omitted, one is built from
    ``device``/``dtype`` with offload off (current all-resident behavior). When
    ``policy.offload`` is set, modules are left on CPU; the pipeline shuttles each
    onto the GPU around its stage. The sigma schedule always lives on the compute
    device — it is never offloaded.
    """
    if policy is None:
        policy = DevicePolicy(device=torch.device(device), compute_dtype=dtype)

    spec = detect_architecture(read_header(path))
    # An all-in-one FLUX checkpoint (transformer + CLIP/T5/Mistral + VAE in one
    # file) lands here too; route it to the FLUX loader.
    if spec.architecture in ("flux1", "flux2"):
        return load_flux_checkpoint(path, device=device, dtype=dtype, policy=policy)
    if spec.architecture not in ("sd15", "sdxl"):
        raise NotImplementedError(f"unsupported architecture {spec.architecture!r}")

    # The training schedule is fully determined by the spec; keep its sigma table
    # (fp32) on the compute device so sigma<->t stays with the latents.
    schedule = DiscreteSchedule(
        make_betas(spec.beta_schedule, spec.num_train_timesteps),
        zero_terminal_snr=spec.zero_terminal_snr,
    )
    schedule.sigmas = schedule.sigmas.to(policy.device)
    schedule.log_sigmas = schedule.log_sigmas.to(policy.device)

    state_dict = load_state_dict(path, device="cpu")

    # With offload, modules wait on CPU and the pipeline moves them per stage. The
    # "idle" group (text encoders + VAE) offloads in every mode; the UNet only in
    # full offload ("encoders" mode keeps it resident — see RUNTIME_SPEC.md R4).
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
        # Park the whole UNet on CPU and stream its blocks (down / mid / up) onto
        # the GPU one at a time per forward, keeping only the small modules
        # (time/label embed, final conv) resident — the SD/SDXL analog of ComfyUI's
        # --lowvram. Fits SDXL's ~2.6 GB UNet on a 4 GB card, where whole-module
        # staging ("full") OOMs once 1024² activations land on top of it.
        backbone = backbone.to(policy.offload_device, policy.compute_dtype).eval()
        stream_blocks(backbone, ("input_blocks", "middle_block", "output_blocks"),
                      policy.device, policy.offload_device)
    else:
        backbone = backbone.to(unet_target, policy.compute_dtype).eval()

    # NHWC for the conv backbones when opted in. cuDNN picks faster channels-last
    # kernels on Ampere+ fp16. Text encoders and 1D-attention modules stay default.
    if policy.channels_last:
        backbone = to_channels_last(backbone)
        vae = to_channels_last(vae)

    # torch.compile wraps the backbone after channels_last so Inductor's first
    # specialize sees the final layout. Raises if the policy offloads the UNet.
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


_ANIMA_DIT_PREFIX = "net."


def load_anima_checkpoint(
    dit_path: str,
    vae_path: str,
    te_path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    policy: DevicePolicy | None = None,
) -> ModelBundle:
    """Load Anima's three-file split (DiT + Qwen-Image VAE + Qwen3 TE) into a
    :class:`ModelBundle`. The VAE stays fp32 (per-channel latent stats blow up
    in fp16); the DiT and Qwen3 encoder run in ``dtype``.

    ``schedule`` is left ``None`` — flow-matching models drive sampling from a
    σ table built at pipeline time by :func:`diffucore.sampling.flow_matching_schedule`,
    not from a discrete training schedule.
    """
    if policy is None:
        policy = DevicePolicy(device=torch.device(device), compute_dtype=dtype)

    # Progress markers to the server's stdout: the load reads three multi-GB files
    # and builds a 2B-param module with no other output, so the last line printed
    # tells you which stage is slow (or stuck).
    _t0 = time.perf_counter()
    def _stage(msg: str) -> None:
        print(f"[load] (+{time.perf_counter() - _t0:.1f}s) {msg}", flush=True)

    # Detection runs on the DiT file (the VAE/TE have their own keys and are
    # already known by name). It validates the file is Anima before we build
    # the 2B-param module.
    _stage("detecting architecture")
    spec = detect_architecture(read_header(dit_path))
    if spec.architecture != "anima":
        raise ValueError(
            f"{dit_path!r} is not an Anima checkpoint (detected {spec.architecture!r})"
        )

    idle_target = policy.offload_device if policy.offload_idle else policy.device
    unet_target = policy.offload_device if policy.offload_unet else policy.device

    # VAE: independent file with no key prefix.
    _stage("loading VAE weights")
    vae = QwenImageVAE()
    vae.load_state_dict(load_state_dict(vae_path, device="cpu"), strict=True)
    vae = vae.to(idle_target, policy.vae_dtype).eval()

    # Text encoder: independent file. Anima's stock encoder is Qwen3-0.6B; the
    # experimental cosmos-qwen3.5 swap ships a Qwen3.5 hybrid (SSM+attention)
    # encoder instead — both the Anima-packaged 4B and the raw 0.8B base. They're
    # identified by an SSM ``linear_attn`` block; the backbone is extracted by its
    # ``embed_tokens.weight`` leaf, which strips any ``model.language_model.``
    # prefix and drops the base model's ``model.visual.*`` / ``mtp.*`` heads. Both
    # expose a 1024-d output, so the LLM-Adapter and DiT are unchanged; only the
    # encoder + its BPE vocab differ.
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

    # DiT (incl. the LLM-Adapter under ``llm_adapter``): keys live under ``net.*``.
    _stage("loading DiT weights (largest file)")
    sd_dit = load_state_dict(dit_path, device="cpu")
    sd_dit = {k[len(_ANIMA_DIT_PREFIX):]: v for k, v in sd_dit.items()
              if k.startswith(_ANIMA_DIT_PREFIX)}
    _stage("building DiT backbone (2B params)")
    backbone = AnimaDiT()
    backbone.load_state_dict(sd_dit, strict=True)
    if policy.offload_stream:
        # Stream the DiT blocks (the SD/FLUX --lowvram analog) so Anima's ~4 GB
        # DiT fits a small card. Keep block 0 resident: TeaCache probes its
        # modulated_self_attn_input directly (outside the block's __call__, so the
        # stream hooks don't fire), and that probe must find its weights on the GPU.
        _stage("streaming DiT blocks to GPU (low-VRAM mode)")
        backbone = backbone.to(policy.offload_device, policy.compute_dtype).eval()
        stream_blocks(backbone, ("blocks",), policy.device, policy.offload_device,
                      keep_resident=(backbone.blocks[0],))
    else:
        backbone = backbone.to(unet_target, policy.compute_dtype).eval()

    if policy.compile:
        _stage("compiling backbone (torch.compile warmup — may take minutes)")
    backbone = maybe_compile_backbone(backbone, policy)
    _stage("model ready")

    tokenizer = AnimaTokenizer.qwen35() if is_qwen35 else AnimaTokenizer()

    return ModelBundle(
        spec=spec,
        schedule=None,                  # flow-matching: σ-schedule built at sample time
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        backbone=backbone,
        vae=vae,
        text_encoder_2=None,
        policy=policy,
    )


# ─── FLUX (FLUX.1 + FLUX.2) ──────────────────────────────────────────────────
#
# FLUX components can arrive bundled in one all-in-one checkpoint or as separate
# files (transformer / VAE / text-encoder(s)). Rather than enumerate the many
# on-disk prefix conventions, each component is located by a fingerprint leaf key
# and its prefix is recovered and stripped — so bare BFL files and nested
# ``model.diffusion_model.`` / ``text_encoders.*`` all-in-one layouts both work.

_FLUX_DIT_LEAF = "double_blocks.0.img_attn.qkv.weight"
_FLUX_VAE_LEAF = "decoder.conv_in.weight"
_FLUX_CLIP_LEAF = "text_model.embeddings.token_embedding.weight"
_FLUX_T5_LEAF = "encoder.block.0.layer.0.SelfAttention.q.weight"
_FLUX_MISTRAL_LEAF = "model.layers.0.self_attn.q_proj.weight"


def _flux_arch(architecture: str) -> dict:
    """Per-family FLUX DiT constants the tensor shapes don't reveal. Both families
    keep head_dim 128 (axes sum to 128), so ``num_heads = hidden // 128``."""
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
    """Build a :class:`VAEConfig` from an LDM-style autoencoder state dict.

    FLUX.1's VAE is 16-channel / 8× (channel_mult ``(1,2,4,4)``) and FLUX.2's is
    128-channel / 16× (one extra downsample stage) — both follow the LDM
    ``encoder.down.{i}.block.{j}`` layout, so the depth, base width, per-level
    multipliers, res-block count, latent channels and quant-conv presence are all
    read off the weights rather than hardcoded. (Attention is assumed mid-only, as
    in the FLUX autoencoders.)
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
        # SD/LDM keeps them at the top level; FLUX.2 nests them under
        # encoder./decoder. (lifted by _lift_quant_convs before this runs).
        use_quant_conv="quant_conv.weight" in sd or "encoder.quant_conv.weight" in sd,
    )


def _lift_quant_convs(sd):
    """Lift FLUX.2's nested ``encoder.quant_conv.*`` / ``decoder.post_quant_conv.*``
    to the top-level ``quant_conv.*`` / ``post_quant_conv.*`` that
    :class:`AutoencoderKL` expects. A no-op for SD/LDM and FLUX.1 (which either
    keep them top-level or have none)."""
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
    """Locate the component whose keys end with ``leaf``, strip its on-disk prefix,
    and return the sub-state-dict (or ``None`` if the component isn't present)."""
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
    """Load ``sub`` into ``module``, tolerating benign *extra* keys (duplicated
    embeddings, projection heads, ``position_ids``) but rejecting any *missing*
    key — a missing key is a real architecture mismatch, the correctness check."""
    drop = ("position_ids",) + tuple(drop_suffixes)
    sub = {k: v for k, v in sub.items() if not any(k.endswith(s) for s in drop)}
    missing, _ = module.load_state_dict(sub, strict=False)
    if missing:
        raise RuntimeError(
            f"{type(module).__name__}: {len(missing)} missing key(s) "
            f"(e.g. {missing[:5]}) — checkpoint/architecture mismatch"
        )
    return module


def _qwen3_config_from_sd(sd) -> Qwen3Config:
    """Derive a Qwen3 config from a checkpoint (head_dim from the per-head q-norm)."""
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
    """Build FLUX.2's text encoder from its LM state dict. Returns
    ``(encoder, tokenizer_kind, needs_mistral_tokenizer)``. Qwen3 (Klein) is told
    apart from Mistral (Dev) by its per-head ``q_norm`` weight."""
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

    Components come from their dedicated file when given, else from the all-in-one
    ``path``; pass either (or a mix). FLUX.1 needs transformer + VAE + T5-XXL +
    CLIP-L; FLUX.2 needs transformer + VAE + Mistral-3 (+ its ``tokenizer.json``,
    ``mistral_tokenizer_path`` or a sidecar next to the encoder file).

    Like Anima, ``schedule`` is left ``None`` — FLUX is flow-matching and builds
    its σ schedule (a resolution-dependent shift) at pipeline time. The VAE runs
    fp32; the transformer and text encoder(s) run in ``dtype``.
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

    # ---- transformer (DiT). Family constants from _flux_arch; widths from shapes.
    dit_sub = _extract_component(dit_sd_full, _FLUX_DIT_LEAF)
    hidden = dit_sub["img_in.weight"].shape[0]
    arch_params = _flux_arch(spec.architecture)
    flux_cfg = FluxConfig.from_state_dict(
        dit_sub, num_heads=hidden // sum(arch_params["axes_dim"]), **arch_params
    )
    backbone = Flux(flux_cfg)
    _load_no_missing(backbone, dit_sub)
    if policy.offload_stream:
        # Park the whole DiT on CPU at the right dtype, then keep the small
        # modules resident and stream the blocks (fits FLUX.1 on a 24 GB card).
        backbone = backbone.to(policy.offload_device, policy.compute_dtype).eval()
        stream_blocks(backbone, ("double_blocks", "single_blocks"),
                      policy.device, policy.offload_device)
    else:
        backbone = backbone.to(unet_target, policy.compute_dtype).eval()
    backbone = maybe_compile_backbone(backbone, policy)

    # ---- VAE (config inferred from the weights: 16-ch/8× for FLUX.1, 128-ch/16×
    # for FLUX.2; scale/shift from the spec — FLUX.2 uses none).
    vae_sub = _extract_component(source(vae_path), _FLUX_VAE_LEAF)
    if vae_sub is None:
        raise ValueError("no VAE found (need it in `path` or `vae_path`)")
    vae_sub = _lift_quant_convs(vae_sub)   # FLUX.2 nests the quant convs
    vae = AutoencoderKL(_infer_vae_config(
        vae_sub, scale_factor=spec.latent_scale, shift_factor=spec.latent_shift
    ))
    _load_no_missing(vae, vae_sub)
    # FLUX.2 bridges its 32-ch VAE latent to the 128-ch DiT space with a 2×2
    # pixel-shuffle + a (non-affine) batch-norm latent normalisation. The conv
    # stack lives in AutoencoderKL; stash the bn stats so the pipeline can invert
    # them before decode (see _flux.flux_text_to_image). eps = config batch_norm_eps.
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
    else:  # flux2 — a single decoder LM (Klein/Qwen3 or Dev/Mistral)
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
        schedule=None,                  # flow-matching: σ-schedule built at sample time
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
