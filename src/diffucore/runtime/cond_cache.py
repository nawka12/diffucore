"""A small LRU cache of prompt conditioning tensors.

Every generation re-tokenizes and re-encodes the prompt (and, for CFG families,
the negative prompt), even when both are unchanged — the dominant workflow (seed
hunting, X/Y/Z sweeps, batch runs) repeats the same conditioning dozens of times.
With any offload mode the cost is not just the encoder forward: ``staged()``
shuttles the text encoder(s) over PCIe both ways per image. Caching the encoded
result lets a repeat-prompt generation skip the whole conditioning stage,
including the encoder staging itself.

Mechanism only — the *policy* (when to create it, when to clear it) lives in the
backend engine, which owns model lifecycle and LoRA state. A ``ModelBundle``
carries an optional instance; the pipelines consult it if present, else behave
exactly as before.
"""

from __future__ import annotations

from collections import OrderedDict


class ConditioningCache:
    """LRU of prompt-key → conditioning tensors, stored on CPU.

    Values are plain dicts of **CPU** tensors: a few MB each (Anima ~1 MB,
    FLUX T5 context ~40 MB), so the cache is VRAM-neutral and the caller moves a
    hit's tensors onto the compute device itself. Keys are hashable tuples the
    pipeline builds (prompt/negative plus any family-specific fields). ``get``
    returns ``None`` on a miss; the pipeline then encodes and ``put``s.

    Not thread-safe by design: the backend drives it from a single FIFO job
    worker, so no locking is needed.
    """

    def __init__(self, max_entries: int = 16):
        self.max_entries = max_entries
        self._store: "OrderedDict[tuple, dict]" = OrderedDict()

    def get(self, key: tuple) -> dict | None:
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)          # mark most-recently-used
        return value

    def put(self, key: tuple, value: dict) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)       # evict least-recently-used

    def clear(self) -> None:
        self._store.clear()
