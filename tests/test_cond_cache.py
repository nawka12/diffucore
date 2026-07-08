"""Conditioning cache: the LRU itself and its wiring into the shared SD/SDXL
``_Pipeline._encode_prompts``.

Pure-CPU, no model files: the encoder half (``_encode_conditioning``) is stubbed
so these exercise the cache *logic* — hit/miss keying, the resolution-independent
split (SDXL's ``y`` rebuilt per call, context/pooled cached) — not the encoders.
The Anima/FLUX pipeline-level hits (zero re-encode, bit-identity) are covered in
``test_anima_pipeline.py`` against the real fixture.
"""

from __future__ import annotations

import types

import torch

from diffucore import ConditioningCache
from diffucore.pipelines._base import _Pipeline
from diffucore.runtime import DevicePolicy


def test_lru_get_put_clear():
    c = ConditioningCache(max_entries=2)
    assert c.get(("a",)) is None
    c.put(("a",), {"v": 1})
    c.put(("b",), {"v": 2})
    assert c.get(("a",))["v"] == 1        # touch 'a' -> most-recently-used
    c.put(("c",), {"v": 3})               # over capacity: evicts LRU ('b'), keeps 'a'
    assert c.get(("b",)) is None
    assert c.get(("a",))["v"] == 1 and c.get(("c",))["v"] == 3
    c.clear()
    assert c.get(("a",)) is None and c.get(("c",)) is None


def _stub_pipeline(arch, cache):
    """A ``_Pipeline`` with no real weights: offload off (so ``staged`` is a no-op
    and never touches the encoder), and a counting ``_encode_conditioning`` that
    returns deterministic per-encode sentinel tensors."""
    pipe = _Pipeline.__new__(_Pipeline)
    pipe.model = types.SimpleNamespace(
        spec=types.SimpleNamespace(architecture=arch),
        text_encoder=object(),
        text_encoder_2=None,
        cond_cache=cache,
    )
    calls: list[tuple[str, str]] = []

    def fake_encode(prompt, negative_prompt):
        calls.append((prompt, negative_prompt))
        h = float(len(calls))                       # distinct per miss, frozen once cached
        out = {"ctx_c": torch.full((1, 4, 8), h), "ctx_u": torch.full((1, 4, 8), -h)}
        if arch == "sdxl":
            out["pooled_c"] = torch.full((1, 1280), h)
            out["pooled_u"] = torch.full((1, 1280), -h)
        return out

    pipe._encode_conditioning = fake_encode
    return pipe, calls


_POLICY = DevicePolicy(device=torch.device("cpu"), compute_dtype=torch.float32)


def test_encode_prompts_hits_cache_sd15():
    pipe, calls = _stub_pipeline("sd15", ConditioningCache())
    cond1, _ = pipe._encode_prompts("a cat", "blurry", 128, 128, _POLICY)
    cond2, _ = pipe._encode_prompts("a cat", "blurry", 256, 256, _POLICY)   # same key -> hit
    assert len(calls) == 1                                   # encoder ran once
    assert torch.equal(cond1["context"], cond2["context"])  # cached context reused
    pipe._encode_prompts("a dog", "blurry", 128, 128, _POLICY)              # new key -> miss
    assert len(calls) == 2
    pipe._encode_prompts("a cat", "different neg", 128, 128, _POLICY)       # new negative -> miss
    assert len(calls) == 3


def test_encode_prompts_sdxl_reassembles_y_per_resolution():
    pipe, calls = _stub_pipeline("sdxl", ConditioningCache())
    cond1, _ = pipe._encode_prompts("a cat", "blurry", 128, 256, _POLICY)
    cond2, _ = pipe._encode_prompts("a cat", "blurry", 512, 768, _POLICY)   # hit, new resolution
    assert len(calls) == 1                                   # encoder ran once
    assert torch.equal(cond1["context"], cond2["context"])  # context cached
    assert not torch.equal(cond1["y"], cond2["y"])           # y is resolution-dependent -> rebuilt
    cond3, _ = pipe._encode_prompts("a cat", "blurry", 128, 256, _POLICY)  # same size again
    assert torch.equal(cond1["y"], cond3["y"])               # same pooled + size -> identical y


def test_encode_prompts_no_cache_reencodes_each_call():
    """``cond_cache=None`` (the default / direct-library path) never consults a
    cache and re-encodes every call — exactly today's behavior."""
    pipe, calls = _stub_pipeline("sd15", None)
    pipe._encode_prompts("a cat", "blurry", 128, 128, _POLICY)
    pipe._encode_prompts("a cat", "blurry", 128, 128, _POLICY)
    assert len(calls) == 2
