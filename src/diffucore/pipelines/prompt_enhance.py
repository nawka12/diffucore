"""Prompt enhancement: a Qwen3 / Qwen3-VL language model (Krea 2's text
encoder, Qwen3-VL-4B-Instruct, by default) rewriting a short prompt into a
detailed one. Text only; a VL file's vision tower is never read.

Two prompt styles. :meth:`PromptEnhancer.enhance` asks for a rewrite in the
chat template (Krea 2: one descriptive paragraph). :meth:`PromptEnhancer.expand`
is a few-shot completion whose answer starts with the user's prompt verbatim,
so tags (Anima's ``@artist``, character, ``1girl``) can't be dropped; the model
only writes the sentences after them.

The Comfy-Org TE file holds all 36 layers and the final norm but no lm_head:
Qwen3-VL-4B (like Qwen3-0.6B/4B) ties it to the token embedding. Sampling
defaults are Qwen's generation config (temperature 0.7, top-p 0.8, top-k 20).
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from ..bundle import _build, _extract_component, _qwen3_config_from_sd
from ..conditioning.krea2_tokenizer import _QWEN_VOCAB
from ..loading import load_state_dict
from ..models._int8 import Int8Linear, int8_fast_available
from ..models.qwen3_text import Qwen3TextEncoder, _apply_rope
from ..runtime import (DevicePolicy, _readahead, _stream_tensors, _upload, fp32_accumulation,
                       parked_on_cpu)

DEFAULT_INSTRUCTION = (
    "You rewrite prompts for a text-to-image model. Expand the user's prompt into one detailed "
    "paragraph describing the image: the main subject and what it is doing, then the colors, shapes, "
    "sizes, textures, materials and quantities, any text the user asked for (quoted exactly), where "
    "things sit relative to each other, the background and setting, the lighting, the camera framing, "
    "and the medium or art style. Keep every detail the user gave and never contradict it; only fill in "
    "what is missing, with concrete visual details. If the prompt is a list of tags, turn it into natural "
    "sentences. Describe only what is in the image. Write in English. Reply with the rewritten prompt "
    "only: no preamble, headings, lists or quotes, under 200 words."
)

# (short prompt, sentences that follow it) in Anima's tag order: quality/meta/
# year/safety, 1girl/1boy, character, series, @artist, general tags.
ANIMA_EXAMPLES = (
    ("1boy, black hair, hoodie, rain, city at night",
     "A young man with messy black hair stands under a flickering streetlight, hands in the pocket of "
     "his grey hoodie as rain streaks past. Neon signs blur across the wet street behind him, and cold "
     "blue light catches the water on his shoulders."),
    ("masterpiece, best quality, newest, safe, 1girl, witch, long purple hair, library",
     "A young witch with long purple hair sits cross-legged on the floor of a candlelit library, a "
     "heavy spellbook open across her lap. Her wide-brimmed hat tilts back as she reads, and dust "
     "drifts through the warm light between towering shelves."),
    ("a fox in a snowy forest",
     "A red fox pads through deep snow between tall pines, its breath fogging in the cold air. Soft "
     "morning light filters through the branches and glitters on the frost."),
)

_QWEN3_LEAF = "layers.0.self_attn.q_norm.weight"
_QWEN3_VL_ROPE_THETA = 5_000_000.0
_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
_VRAM_MARGIN = 512 * 2**20


class PromptEnhancer:
    """A Qwen3 LM left on its checkpoint's mmap, uploaded for each call and
    dropped afterward. ``int8`` quantizes the layers' linears on the GPU at
    upload (about half the VRAM and bytes read per token); the embedding,
    which is also the LM head, stays in the compute dtype."""

    def __init__(self, lm: Qwen3TextEncoder, policy: DevicePolicy, int8: bool = False):
        self.lm = lm
        self.policy = policy
        self.int8 = int8
        self.tok = Tokenizer.from_file(str(_QWEN_VOCAB))
        self._stop = {self.tok.token_to_id(t) for t in _STOP_TOKENS}
        self._newlines: set[int] | None = None

    def enhance(self, prompt: str, *, instruction: str | None = None, max_new_tokens: int = 400,
                **kwargs) -> str:
        """The prompt rewritten per ``instruction`` (chat template). Keyword
        arguments as :meth:`_run`."""
        text = (f"<|im_start|>system\n{instruction or DEFAULT_INSTRUCTION}<|im_end|>\n"
                f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n")
        out = self._run(self.tok.encode(text).ids, max_new_tokens, self._stop, **kwargs)
        return self.tok.decode(out, skip_special_tokens=True).strip()

    def expand(self, prompt: str, *, examples: Sequence[tuple[str, str]] = ANIMA_EXAMPLES,
               max_new_tokens: int = 200, **kwargs) -> str:
        """``prompt`` verbatim, then the sentences the model adds after it in
        the style of ``examples`` (a few-shot completion; works on base models
        too). Keyword arguments as :meth:`_run`."""
        head = " ".join(prompt.split()).rstrip(" ,.;")
        shots = "".join(f"Short prompt: {s}\nDetailed prompt: {s}. {d}\n\n" for s, d in examples)
        text = f"{shots}Short prompt: {head}\nDetailed prompt: {head}."
        out = self._run(self.tok.encode(text).ids, max_new_tokens, self._stop | self._newline_ids(),
                        **kwargs)
        tail = self.tok.decode(out, skip_special_tokens=True).split("\n")[0].strip()
        return f"{head}. {tail}" if tail else f"{head}."

    def _newline_ids(self) -> set[int]:
        """Tokens containing a newline (``Ċ`` in byte-level BPE): the end of a
        few-shot answer."""
        if self._newlines is None:
            self._newlines = {i for i in range(self.tok.get_vocab_size())
                              if "Ċ" in (self.tok.id_to_token(i) or "")}
        return self._newlines

    def _run(self, ids: list[int], max_new_tokens: int, stop: set[int], *, temperature: float = 0.7,
             top_p: float = 0.8, top_k: int = 20, seed: int | None = None,
             make_room: Sequence[nn.Module] = (),
             progress_callback: Callable[[int, int], None] | None = None) -> list[int]:
        """Generated token ids. ``temperature`` 0 is greedy. ``make_room`` are
        modules moved to CPU for the call when the LM wouldn't fit beside them
        (the loaded model's GPU-resident blocks)."""
        gen = torch.Generator()
        if seed is None:
            gen.seed()
        else:
            gen.manual_seed(seed)
        device = self.policy.device
        with ExitStack() as stack:
            if device.type == "cuda" and _free_vram(device) < self._need_bytes(len(ids) + max_new_tokens):
                stack.enter_context(parked_on_cpu(make_room))
            stack.enter_context(self._resident(device))
            stack.enter_context(torch.no_grad())
            stack.enter_context(fp32_accumulation())
            return self._generate(ids, max_new_tokens, temperature, top_p, top_k, gen, progress_callback,
                                  stop)

    def _linears(self) -> list[tuple[nn.Module, str, nn.Linear]]:
        return [(parent, name, child) for layer in self.lm.model.layers
                for parent in (layer.self_attn, layer.mlp)
                for name, child in parent.named_children() if isinstance(child, nn.Linear)]

    def _need_bytes(self, total: int) -> int:
        cfg = self.lm.cfg
        size = torch.finfo(self.policy.compute_dtype).bits // 8
        params = sum(p.numel() for p in self.lm.parameters())
        lin = sum(c.weight.numel() for _, _, c in self._linears()) if self.int8 else 0
        kv = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * total * size
        return (params - lin) * size + lin + kv + _VRAM_MARGIN

    @contextmanager
    def _resident(self, device):
        """The LM on ``device`` for one call, read from the mmap one tensor at
        a time (a single up-front readahead of 8 GB would be evicted before
        use on a low-RAM box), restored to its mmap tensors afterward."""
        dtype = self.policy.compute_dtype
        linears = self._linears() if self.int8 else []
        skip = {id(c.weight) for _, _, c in linears}
        uploaded, swapped = [], []
        try:
            for t in _stream_tensors(self.lm):
                if id(t) not in skip:
                    _readahead([t])
                    uploaded.append((t, t.data))
                    t.data = _upload(t.data, device, dtype)
            for parent, name, child in linears:
                _readahead([child.weight])
                setattr(parent, name, Int8Linear.from_linear(child, device=device))
                swapped.append((parent, name, child))
            yield
        finally:
            for parent, name, child in swapped:
                setattr(parent, name, child)
            for t, master in uploaded:
                t.data = master
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def _generate(self, ids, max_new_tokens, temperature, top_p, top_k, gen, progress_callback,
                  stop=None) -> list[int]:
        stop = self._stop if stop is None else stop
        cfg = self.lm.cfg
        emb = self.lm.model.embed_tokens.weight
        device, dtype = emb.device, emb.dtype
        total = len(ids) + max_new_tokens
        # RoPE tables built in fp64 here: the module's inv_freq buffer is cast to
        # the compute dtype with the weights, too coarse for positions in the hundreds.
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64,
                                                      device=device) / cfg.head_dim))
        ang = torch.arange(total, dtype=torch.float64, device=device)[:, None] * inv
        ang = torch.cat([ang, ang], dim=-1)
        cos, sin = ang.cos().to(dtype), ang.sin().to(dtype)
        kv = torch.empty(cfg.num_hidden_layers, 2, 1, cfg.num_key_value_heads, total, cfg.head_dim,
                         device=device, dtype=dtype)
        step_ids = torch.tensor([ids], device=device)
        pos, out = 0, []
        for step in range(max_new_tokens):
            logits = self._forward(step_ids, pos, cos, sin, kv)
            pos += step_ids.shape[1]
            nxt = _sample(logits, temperature, top_p, top_k, gen)
            out.append(nxt)   # kept: a newline stop can carry text (".\n" is one token)
            if nxt in stop:
                break
            step_ids = torch.tensor([[nxt]], device=device)
            if progress_callback is not None:
                progress_callback(step + 1, max_new_tokens)
        return out

    def _forward(self, ids: torch.Tensor, pos: int, cos, sin, kv) -> torch.Tensor:
        """Next-token logits (fp32) after ``ids`` at positions ``pos..``, with
        K/V appended to ``kv``. A multi-token call is the prefill (``pos`` 0)."""
        m = self.lm.model
        T = ids.shape[1]
        end = pos + T
        x = m.embed_tokens(ids)
        c, s = cos[None, pos:end], sin[None, pos:end]
        for i, layer in enumerate(m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(1, T, a.num_heads, a.head_dim)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(1, T, a.num_kv_heads, a.head_dim)).transpose(1, 2)
            v = a.v_proj(h).view(1, T, a.num_kv_heads, a.head_dim).transpose(1, 2)
            q, k = _apply_rope(q, k, c, s)
            kv[i, 0, :, :, pos:end] = k
            kv[i, 1, :, :, pos:end] = v
            o = F.scaled_dot_product_attention(q, kv[i, 0, :, :, :end], kv[i, 1, :, :, :end],
                                               is_causal=T > 1, enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, T, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return F.linear(m.norm(x[0, -1]), m.embed_tokens.weight).float()


def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int,
            gen: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        vals, idx = logits.topk(min(top_k, logits.numel()))
    else:
        vals, idx = logits.sort(descending=True)
    probs = vals.softmax(-1)
    if top_p < 1.0:
        # Keep the smallest head whose mass reaches top_p (always the top token).
        probs = probs * ((probs.cumsum(-1) - probs) < top_p)
    choice = torch.multinomial(probs.cpu(), 1, generator=gen)
    return int(idx[choice.item()])


def _free_vram(device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def load_prompt_enhancer(path: str, policy: DevicePolicy, int8: bool = False) -> PromptEnhancer:
    """A Qwen3 or Qwen3-VL language-model file (Krea 2's TE, Anima's
    Qwen3-0.6B, ...) as a :class:`PromptEnhancer`. The weights stay on the
    file's mmap until a call uploads them. ``int8`` applies only where the
    fast int8 path runs (CUDA sm75+ with Triton)."""
    sd = load_state_dict(path, device="cpu")
    lm = _extract_component(sd, _QWEN3_LEAF)
    if lm is None:
        raise ValueError(f"{Path(path).name} is not a Qwen3 or Qwen3-VL language model")
    if any(k.endswith("lm_head.weight") for k in sd):
        raise ValueError(f"{Path(path).name} has an untied LM head, which the prompt enhancer doesn't read")
    lm = {"model." + k: v for k, v in lm.items()}
    cfg = _qwen3_config_from_sd(lm)
    if any("visual." in k or "language_model." in k for k in sd):
        cfg = replace(cfg, rope_theta=_QWEN3_VL_ROPE_THETA)
    model = _build(Qwen3TextEncoder, cfg)
    model.load_state_dict(lm, strict=True, assign=True)
    return PromptEnhancer(model.eval(), policy, int8=int8 and int8_fast_available(policy.device))
