"""LPW (long prompt weighting) for the SDXL conditioner.

Covers the A1111 attention parser, the 75-token chunker, and the end-to-end
``SDXLConditioner`` behavior with tiny random encoders (no checkpoint needed):
backward-compat for short/unweighted prompts, that weighting changes the
context, and that long prompts produce a >77-token context.
"""

import torch

from diffucore.conditioning import (
    EOS_TOKEN,
    MAX_LENGTH,
    CLIPTokenizer,
    SDXLConditioner,
    _chunk,
    _tokenize_with_weights,
    parse_prompt_attention,
)
from diffucore.models.clip_text import CLIPTextConfig, CLIPTextEncoder
from diffucore.models.open_clip_text import OpenCLIPTextConfig, OpenCLIPTextEncoder


def test_parse_attention_syntax():
    assert parse_prompt_attention("a (cat:1.3) dog") == [["a ", 1.0], ["cat", 1.3], [" dog", 1.0]]
    assert parse_prompt_attention("a (cat) dog") == [["a ", 1.0], ["cat", 1.1], [" dog", 1.0]]
    assert parse_prompt_attention("a [cat] dog")[1][1] == 1.0 / 1.1
    assert abs(parse_prompt_attention("((cat))")[0][1] - 1.1 * 1.1) < 1e-9
    assert parse_prompt_attention(r"a \(cat\)") == [["a (cat)", 1.0]]
    assert parse_prompt_attention("") == [["", 1.0]]


def test_chunker_short_prompt_matches_plain_encode():
    tok = CLIPTokenizer()
    tokens, weights = _tokenize_with_weights(tok, "a photo of a cat")
    ids, _ = _chunk(tokens, weights, pad_token=EOS_TOKEN)
    assert len(ids) == 1
    assert ids[0] == tok.encode("a photo of a cat", pad_token=EOS_TOKEN).tolist()


def test_chunker_empty_and_long():
    tok = CLIPTokenizer()
    # empty prompt -> exactly one [BOS, EOS, pad...] chunk, all weights 1.0
    ids, ws = _chunk(*_tokenize_with_weights(tok, ""), pad_token=0)
    assert len(ids) == 1 and ids[0][:2] == [49406, EOS_TOKEN] and len(ids[0]) == MAX_LENGTH
    assert all(w == 1.0 for w in ws[0])
    # long prompt -> several chunks, each a full 77-token window
    ids, _ = _chunk(*_tokenize_with_weights(tok, "cat " * 200), pad_token=0)
    assert len(ids) > 1 and all(len(c) == MAX_LENGTH for c in ids)


def _tiny_conditioner():
    torch.manual_seed(0)
    clip_l = CLIPTextEncoder(
        CLIPTextConfig(hidden_size=64, num_layers=3, num_heads=4, intermediate_size=128)
    ).eval()
    clip_g = OpenCLIPTextEncoder(
        OpenCLIPTextConfig(width=80, num_layers=3, num_heads=4, mlp_dim=160)
    ).eval()
    with torch.no_grad():  # the bigG Parameters are created empty
        clip_g.positional_embedding.normal_()
        clip_g.text_projection.normal_()
        for blk in clip_g.transformer.resblocks:
            blk.attn.in_proj_weight.normal_(std=0.02)
            blk.attn.in_proj_bias.zero_()
    return SDXLConditioner(CLIPTokenizer(), clip_l, clip_g), clip_l, clip_g


def test_unweighted_short_prompt_matches_plain_path():
    cond, clip_l, clip_g = _tiny_conditioner()
    tok = cond.tokenizer
    with torch.no_grad():
        ctx, pooled = cond("a photo of a cat")
        h_l = clip_l(tok.encode("a photo of a cat", pad_token=EOS_TOKEN).unsqueeze(0), clip_skip=2)
        h_g, p = clip_g(tok.encode("a photo of a cat", pad_token=0).unsqueeze(0), clip_skip=2)
        ref = torch.cat([h_l, h_g], dim=-1)
    assert torch.allclose(ctx, ref, atol=1e-5)
    assert torch.allclose(pooled, p, atol=1e-5)


def test_weighting_changes_context():
    cond, _, _ = _tiny_conditioner()
    with torch.no_grad():
        weighted, _ = cond("a photo of a (cat:1.5)")
        base, _ = cond("a photo of a cat")
    assert not torch.allclose(weighted, base)


def test_long_prompt_exceeds_77_tokens():
    cond, _, _ = _tiny_conditioner()
    with torch.no_grad():
        ctx, pooled = cond("cat " * 200, batch=2)
    assert ctx.shape == (2, MAX_LENGTH * 3, 80 + 64)
    assert pooled.shape == (2, 80)


def test_mismatched_chunks_padded_for_cfg_batching():
    # A long positive prompt and a short negative prompt split into different
    # chunk counts; CFG batches them along dim 0, so their sequence lengths must
    # match. _match_context_chunks pads the shorter with empty-prompt chunks.
    from diffucore.pipelines._base import _match_context_chunks

    cond, _, _ = _tiny_conditioner()
    with torch.no_grad():
        ctx_c, _ = cond("cat " * 200)  # several chunks
        ctx_u, _ = cond("")            # single chunk
        assert ctx_c.shape[1] != ctx_u.shape[1]

        padded_c, padded_u = _match_context_chunks(ctx_c, ctx_u, cond)
        assert padded_c.shape[1] == padded_u.shape[1]
        # batching cond+uncond along dim 0 (what CFGDenoiser does) now works
        torch.cat([padded_c, padded_u])

        # the longer context is untouched; the shorter is extended with the
        # empty-prompt chunk embedding
        assert torch.equal(padded_c, ctx_c)
        empty = cond("", batch=1)[0]
        assert torch.allclose(padded_u[:, :MAX_LENGTH], empty)
        assert torch.allclose(padded_u[:, MAX_LENGTH : 2 * MAX_LENGTH], empty)
