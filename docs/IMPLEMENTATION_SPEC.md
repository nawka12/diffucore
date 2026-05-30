# Implementation Spec — M4–M7 (SD1.5 model components + pipeline)

> **Status: implemented and verified** (see [`ROADMAP.md`](ROADMAP.md)). This
> remains the authoritative description of *what was built*; the code in
> `src/diffucore/` follows it. Implementation learnings/deviations are recorded
> in [`HANDOFF.md`](HANDOFF.md) "Gotchas".

This is the build sheet for the parts that can only be verified on the RTX 2060
with real weights. The sampling core they plug into (schedules, samplers, the
denoising loop, CFG, eps/v parameterization, sigma⇄t) is **already implemented
and tested** — do not reimplement it; import from `diffucore.sampling`.

Each module below has a skeleton in `src/diffucore/` with the exact forward
signature. Fill in the bodies, then run the per-module verification at the end.

## Guiding strategy: mirror the on-disk names

SD1.5 checkpoints use the original LDM/CompVis naming. **Name your module
submodules and parameters identically to the on-disk keys (minus the top-level
prefix).** Then weight loading is just:

```python
sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
module.load_state_dict(sub, strict=True)   # strict=True is your correctness check
```

A clean `strict=True` load is the single best signal that your architecture
matches the checkpoint. Prefixes:

| Module | Prefix |
|---|---|
| UNet | `model.diffusion_model.` |
| VAE | `first_stage_model.` |
| CLIP text encoder | `cond_stage_model.transformer.` (HF CLIP names under `text_model.`) |

Reference checkpoint for development: `v1-5-pruned-emaonly.safetensors`
(SD1.5, ~4 GB, eps-prediction). Inspect real keys first:
`python -c "from diffucore.loading import read_header; import pprint; pprint.pprint(sorted(read_header('PATH'))[:40])"`.

---

## §Conditioning  (`conditioning/__init__.py`)

**CLIPTokenizer** — standard CLIP BPE (vocab 49408). Use HF `tokenizers` or the
CLIP `bpe_simple_vocab_16e6` merges. Special tokens: BOS `49406`
(`<|startoftext|>`), EOS `49407` (`<|endoftext|>`). Build a sequence
`[BOS, *tokens, EOS]`, truncate to 77, then **pad to 77 with EOS (49407)**.
Output `LongTensor[77]`.

**Conditioner** — `__call__(prompt, batch=1) -> [batch, 77, 768]`: tokenize →
`text_encoder(ids, clip_skip)` → expand to batch. For CFG, the pipeline calls
this twice: once with the prompt (cond) and once with the negative/empty prompt
(uncond). `clip_skip=1` for stock SD1.5.

---

## §CLIP  (`models/clip_text.py`)  — CLIP ViT-L/14 text transformer

Config (`CLIPTextConfig`): hidden 768, layers 12, heads 12 (head dim 64), mlp
3072, max positions 77, vocab 49408, `layer_norm_eps=1e-5`.

Structure:
- `token_embedding` `[49408, 768]` + `position_embedding` `[77, 768]` (added).
- 12 × encoder layer, pre-norm:
  - `x = x + self_attn(layer_norm1(x), causal_mask)`
  - `x = x + mlp(layer_norm2(x))`, mlp = `fc1(768→3072) → quick_gelu → fc2(3072→768)`.
  - Attention is multi-head (12 heads) with a **causal** mask (lower-triangular),
    `q/k/v/out` = Linear(768→768) with bias.
  - `quick_gelu(x) = x * sigmoid(1.702 * x)`.
- `final_layer_norm`.

Output (`forward(token_ids, clip_skip=1)`):
- `clip_skip=1` → return `final_layer_norm(last_hidden_state)`.
- `clip_skip=k>1` → return the hidden state from `k` layers before the end,
  **without** `final_layer_norm`.

On-disk key names (HF CLIP) to mirror, e.g.
`text_model.embeddings.token_embedding.weight`,
`text_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`,
`text_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}`,
`text_model.encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}`,
`text_model.final_layer_norm.{weight,bias}`.

---

## §VAE  (`models/vae.py`)  — AutoencoderKL

Config: base 128, `channel_mult=(1,2,4,4)` → channels `[128,256,512,512]`,
`num_res_blocks=2`, `z_channels=4`, `scale_factor=0.18215`. GroupNorm uses 32
groups; nonlinearity is SiLU/swish.

**Encoder**: `conv_in(3→128)` → for each level: `num_res_blocks` ResnetBlocks
then a strided-conv `Downsample` (except the last level) → mid (`ResnetBlock`,
spatial self-`Attention`, `ResnetBlock`) → GroupNorm → SiLU →
`conv_out(512 → 2*z=8)`. Then `quant_conv(8→8)`. Output parameterizes a
`DiagonalGaussian` (`mean, logvar = chunk(h, 2)`); `encode(sample=True)` draws
`mean + exp(0.5*logvar)*eps`, else returns `mean`.

**Decoder**: `post_quant_conv(4→4)` → `conv_in(4→512)` → mid (same as encoder) →
for each level reversed: `num_res_blocks+1` ResnetBlocks then nearest-neighbor
`Upsample`+conv (except the last) → GroupNorm → SiLU → `conv_out(128→3)`.

**Resnet/attention** of the VAE: ResnetBlock = GroupNorm→SiLU→conv3x3,
GroupNorm→SiLU→conv3x3, + (1x1 shortcut if channels change). Attention =
GroupNorm → q,k,v 1x1 → scaled dot-product over `H·W` → proj 1x1 → residual.

**Scale factor** (apply at the boundary, not inside the nets):
`encode` returns `0.18215 * z`; `decode` consumes `z / 0.18215`.

On-disk keys mirror `encoder.*`, `decoder.*`, `quant_conv.*`, `post_quant_conv.*`.

---

## §UNet  (`models/unet.py`)  — SD1.5 epsilon UNet

Config: in/out 4, `model_channels=320`, `channel_mult=(1,2,4,4)` →
`[320,640,1280,1280]`, `num_res_blocks=2`, `num_heads=8`, `context_dim=768`,
`transformer_depth=1`. `time_embed_dim = 4*320 = 1280`. Attention is present at
levels with downsample factor ∈ `{1,2,4}` (i.e. levels 0,1,2 — **not** the
deepest 1280/level-3 blocks).

`forward(x[B,4,h,w], timesteps[B], context[B,77,768]) -> eps[B,4,h,w]`:

- **Time embedding**: sinusoidal `timestep_embedding(timesteps, 320)` →
  `Linear(320→1280) → SiLU → Linear(1280→1280)`. `timesteps` here are the
  continuous indices from `DiscreteSchedule.sigma_to_t` (floats are fine).
- **input_blocks** (12): `[0]`=conv_in(4→320); then per level i: `num_res_blocks`
  × `(ResBlock [+ SpatialTransformer if level has attn])`, then a `Downsample`
  (strided conv) after levels 0–2. Stash every block output for skips.
- **middle_block**: `ResBlock, SpatialTransformer, ResBlock`.
- **output_blocks** (12): mirror; each block concatenates the matching skip on
  the channel dim before its ResBlock; `Upsample` (nearest+conv) ends levels.
- **out**: GroupNorm32 → SiLU → `conv3x3(320→4)` (zero-initialized).

**ResBlock**: `GroupNorm32→SiLU→conv3x3`; add `SiLU→Linear(1280→out_ch)` of the
time embedding (broadcast over H,W); `GroupNorm32→SiLU→dropout→conv3x3`
(zero-init last conv); skip = identity or 1x1 conv if channels change.

**SpatialTransformer**: `GroupNorm32 → proj_in(1x1) → rearrange b c h w → b (h w)
c → transformer_depth × BasicTransformerBlock → rearrange back → proj_out(1x1,
zero-init) → + residual`.

**BasicTransformerBlock** (pre-norm, residual on each):
- `x += attn1(LN(x))` — self-attention.
- `x += attn2(LN(x), context)` — cross-attention to the 768-d text context.
- `x += ff(LN(x))` — `ff = GEGLU(dim, 4*dim) → Linear(4*dim, dim)`.

**Attention** (`CrossAttention`): heads 8, `dim_head = channels//8`,
`scale = dim_head**-0.5`; `to_q(dim→inner, bias=False)`,
`to_k/to_v(context_dim→inner, bias=False)`, `to_out = Linear(inner→dim)`.
**GEGLU**: `Linear(dim, 2*inner)`, split → `a, gate`; output `a * gelu(gate)`.

On-disk keys mirror the LDM names: `time_embed.{0,2}.*`, `input_blocks.{n}.{m}.*`,
`middle_block.{0,1,2}.*`, `output_blocks.{n}.{m}.*`, `out.{0,2}.*`. (Within a
block, index `0` is the ResBlock and `1` is the SpatialTransformer.)

---

## §Pipeline  (`pipelines/text_to_image.py`)  — reference wiring

Everything below the model forwards already exists and is tested. Wiring:

```python
from diffucore.sampling import (EpsScaling, ModelDenoiser, CFGDenoiser,
                                 karras_schedule, get_sampler)

scaling  = EpsScaling()                       # spec.prediction == "eps"
denoiser = ModelDenoiser(model.backbone, scaling, model.schedule)

cond   = {"context": conditioner(prompt,           batch)}
uncond = {"context": conditioner(negative_prompt,  batch)}
cfg    = CFGDenoiser(denoiser, cond, uncond, scale=cfg_scale)

sigmas = karras_schedule(steps,
                         model.schedule.sigma_min.item(),
                         model.schedule.sigma_max.item(),
                         device=device, dtype=compute_dtype)

g = torch.Generator(device).manual_seed(seed)
x = torch.randn(batch, 4, height//8, width//8, generator=g,
                device=device, dtype=compute_dtype) * sigmas[0]

x0  = get_sampler(sampler)(cfg, x, sigmas)    # the verified loop
img = model.vae.decode(x0)                    # then [-1,1] -> uint8 -> PIL
```

Note: `ModelDenoiser` calls `backbone(model_input, t, **cond)`, so the UNet's
forward must accept the context as the `context=` kwarg (i.e. call with
`context=...`). Keep the kwarg name consistent between `Conditioner` output dict,
`CFGDenoiser` cond/uncond dicts, and `UNetModel.forward`.

---

## §Runtime  (`runtime/__init__.py`)

`DevicePolicy.auto()` picks CUDA+fp16 on the 2060. Run the UNet/CLIP in fp16; run
the **VAE in fp32** (fp16 VAE decode produces artifacts/NaNs on many SD1.5
weights) and keep all sigma math fp32. With 12 GB, 512² fits resident; add
optional CPU offload + tiled VAE only if you push to high resolutions.

---

## Verification plan (the M4–M7 success criteria)

Use HF `diffusers`/`transformers` purely as a **numerical oracle** for the same
inputs — compare outputs, don't copy code.

- **M4 CLIP**: `strict=True` load succeeds; `encode("a photo of an astronaut
  riding a horse")` → `[1,77,768]`; values match a `transformers.CLIPTextModel`
  reference within `atol≈1e-3` (fp32).
- **M5 VAE**: `strict=True` load; `decode(encode(img))` reaches **PSNR > 25 dB**
  on a natural image; latent channel stats are O(1) after the 0.18215 scaling.
- **M6 UNet**: `strict=True` load; one `forward([1,4,64,64], [1], [1,77,768])`
  → `[1,4,64,64]`, runs in fp16 under ~6 GB; a single denoise step reduces the
  residual toward a reference eps within tolerance.
- **M7 pipeline**: produces a coherent 512² image from the reference checkpoint;
  **same seed → identical image** (determinism); a known prompt looks right.
