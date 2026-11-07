"""Reference Z-Image transformer for the oracle tests.

``ZImageTransformer2DModel`` from diffusers 0.42 (Apache-2.0, Alibaba Z-Image
Team and The HuggingFace Team), the standard (non-Omni) path with the diffusers
``Attention`` and ``RMSNorm`` replaced by minimal equivalents and the
gradient-checkpoint, ControlNet and SigLIP branches dropped. State-dict names
are the diffusers ones, so it doubles as the key-name oracle for
``from_diffusers``. Each batch item keeps its own padding, as in the original.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class Attention(nn.Module):
    def __init__(self, dim, heads, eps=1e-5):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.norm_q = RMSNorm(dim // heads, eps)
        self.norm_k = RMSNorm(dim // heads, eps)

    def forward(self, hidden_states, attention_mask, freqs_cis):
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        query, key = self.norm_q(query), self.norm_k(key)

        def apply_rotary_emb(x_in, freqs_cis):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            return torch.view_as_real(x * freqs_cis).flatten(3).type_as(x_in)

        query, key = apply_rotary_emb(query, freqs_cis), apply_rotary_emb(key, freqs_cis)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask)
        out = out.transpose(1, 2).flatten(2, 3).to(query.dtype)
        return self.to_out[0](out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps, modulation=True):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = RMSNorm(dim, norm_eps)
        self.ffn_norm1 = RMSNorm(dim, norm_eps)
        self.attention_norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    def forward(self, x, attn_mask, freqs_cis, adaln_input=None):
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attn_mask, freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)
            return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        attn_out = self.attention(self.attention_norm1(x), attn_mask, freqs_cis)
        x = x + self.attention_norm2(attn_out)
        return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True))

    def forward(self, x, c):
        scale = (1.0 + self.adaLN_modulation(c)).unsqueeze(1)
        return self.linear(self.norm_final(x) * scale)


class RopeEmbedder:
    def __init__(self, theta, axes_dims, axes_lens):
        self.theta, self.axes_dims, self.axes_lens = theta, axes_dims, axes_lens
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim, end, theta):
        freqs_cis = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            timestep = torch.arange(e, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return freqs_cis

    def __call__(self, ids):
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, self.theta)
        return torch.cat([self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


class ZImageTransformer2DModel(nn.Module):
    def __init__(self, patch_size=2, f_patch_size=1, in_channels=16, dim=3840, n_layers=30, n_refiner_layers=2,
                 n_heads=30, norm_eps=1e-5, cap_feat_dim=2560, rope_theta=256.0, t_scale=1000.0,
                 axes_dims=(32, 48, 48), axes_lens=(1536, 512, 512)):
        super().__init__()
        self.in_channels = self.out_channels = in_channels
        self.patch_size, self.f_patch_size = patch_size, f_patch_size
        self.t_scale = t_scale
        key = f"{patch_size}-{f_patch_size}"
        self.all_x_embedder = nn.ModuleDict(
            {key: nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)})
        self.all_final_layer = nn.ModuleDict(
            {key: FinalLayer(dim, patch_size * patch_size * f_patch_size * in_channels)})
        self.noise_refiner = nn.ModuleList(
            [ZImageTransformerBlock(dim, n_heads, norm_eps, True) for _ in range(n_refiner_layers)])
        self.context_refiner = nn.ModuleList(
            [ZImageTransformerBlock(dim, n_heads, norm_eps, False) for _ in range(n_refiner_layers)])
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))
        self.x_pad_token = nn.Parameter(torch.zeros((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.zeros((1, dim)))
        self.layers = nn.ModuleList([ZImageTransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)])
        self.rope_embedder = RopeEmbedder(rope_theta, list(axes_dims), list(axes_lens))

    @staticmethod
    def create_coordinate_grid(size, start=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32) for x0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def _patchify_image(self, image, patch_size, f_patch_size):
        pH, pW, pF = patch_size, patch_size, f_patch_size
        C, F_, H, W = image.size()
        F_t, H_t, W_t = F_ // pF, H // pH, W // pW
        image = image.view(C, F_t, pF, H_t, pH, W_t, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_t * H_t * W_t, pF * pH * pW * C)
        return image, (F_, H, W), (F_t, H_t, W_t)

    def _pad_with_ids(self, feat, pos_grid_size, pos_start):
        ori_len = len(feat)
        pad_len = (-ori_len) % SEQ_MULTI_OF
        total_len = ori_len + pad_len
        ori_pos_ids = self.create_coordinate_grid(size=pos_grid_size, start=pos_start).flatten(0, 2)
        if pad_len > 0:
            pad_pos_ids = self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0)).flatten(0, 2).repeat(pad_len, 1)
            pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat([torch.zeros(ori_len, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)])
        else:
            pos_ids, padded_feat = ori_pos_ids, feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool)
        return padded_feat, pos_ids, pad_mask, total_len

    def _prepare_sequence(self, feats, pos_ids, inner_pad_mask, pad_token):
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)
        feats_cat = torch.cat(feats, dim=0)
        mask = torch.cat(inner_pad_mask).unsqueeze(-1)
        feats_cat = torch.where(mask, pad_token.to(feats_cat.dtype), feats_cat)
        feats = list(feats_cat.split(item_seqlens, dim=0))
        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))
        feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]
        if all(seq == max_seqlen for seq in item_seqlens):
            attn_mask = None
        else:
            attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool)
            for i, seq_len in enumerate(item_seqlens):
                attn_mask[i, :seq_len] = 1
        return feats, freqs_cis, attn_mask, item_seqlens

    def embed(self, x, t, cap_feats):
        """Everything before the main layers: ``(unified, mask, freqs, adaln, sizes)``."""
        patch_size, f_patch_size = self.patch_size, self.f_patch_size
        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])
        all_img, all_size, all_img_pos, all_img_mask = [], [], [], []
        all_cap, all_cap_pos, all_cap_mask = [], [], []
        for image, cap_feat in zip(x, cap_feats):
            cap_out, cap_pos, cap_mask, cap_len = self._pad_with_ids(
                cap_feat, (len(cap_feat) + (-len(cap_feat)) % SEQ_MULTI_OF, 1, 1), (1, 0, 0))
            all_cap.append(cap_out), all_cap_pos.append(cap_pos), all_cap_mask.append(cap_mask)
            patches, size, (F_t, H_t, W_t) = self._patchify_image(image, patch_size, f_patch_size)
            img_out, img_pos, img_mask, _ = self._pad_with_ids(patches, (F_t, H_t, W_t), (cap_len + 1, 0, 0))
            all_img.append(img_out), all_size.append(size), all_img_pos.append(img_pos), all_img_mask.append(img_mask)

        x_seqlens = [len(xi) for xi in all_img]
        emb = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(all_img, dim=0))
        img, x_freqs, x_mask, _ = self._prepare_sequence(
            list(emb.split(x_seqlens, dim=0)), all_img_pos, all_img_mask, self.x_pad_token)
        for layer in self.noise_refiner:
            img = layer(img, x_mask, x_freqs, adaln_input)

        cap_seqlens = [len(ci) for ci in all_cap]
        cap_emb = self.cap_embedder(torch.cat(all_cap, dim=0))
        cap, cap_freqs, cap_mask, _ = self._prepare_sequence(
            list(cap_emb.split(cap_seqlens, dim=0)), all_cap_pos, all_cap_mask, self.cap_pad_token)
        for layer in self.context_refiner:
            cap = layer(cap, cap_mask, cap_freqs)

        unified, unified_freqs = [], []
        for i in range(len(x)):
            unified.append(torch.cat([img[i][: x_seqlens[i]], cap[i][: cap_seqlens[i]]]))
            unified_freqs.append(torch.cat([x_freqs[i][: x_seqlens[i]], cap_freqs[i][: cap_seqlens[i]]]))
        unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]
        max_seqlen = max(unified_seqlens)
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)
        if all(seq == max_seqlen for seq in unified_seqlens):
            unified_mask = None
        else:
            unified_mask = torch.zeros((len(x), max_seqlen), dtype=torch.bool)
            for i, seq_len in enumerate(unified_seqlens):
                unified_mask[i, :seq_len] = 1
        return unified, unified_mask, unified_freqs, adaln_input, all_size

    def forward(self, x, t, cap_feats):
        """``x``: list of ``(C, 1, H, W)`` latents, ``cap_feats``: list of ``(L, cap_feat_dim)``,
        ``t``: ``(B,)`` model timestep. Returns a list of ``(C, 1, H, W)``."""
        unified, unified_mask, unified_freqs, adaln_input, all_size = self.embed(x, t, cap_feats)
        for layer in self.layers:
            unified = layer(unified, unified_mask, unified_freqs, adaln_input)
        return self.finish(unified, adaln_input, all_size)

    def finish(self, unified, adaln_input, all_size):
        patch_size, f_patch_size = self.patch_size, self.f_patch_size
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)

        pH = pW = patch_size
        pF = f_patch_size
        out = []
        for i, item in enumerate(unified.unbind(dim=0)):
            F_, H, W = all_size[i]
            ori_len = (F_ // pF) * (H // pH) * (W // pW)
            out.append(item[:ori_len].view(F_ // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                       .permute(6, 0, 3, 1, 4, 2, 5).reshape(self.out_channels, F_, H, W))
        return out
