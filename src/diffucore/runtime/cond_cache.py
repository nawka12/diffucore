"""A small LRU cache of prompt conditioning tensors, so a repeated prompt skips
encoding and, under offload, the text encoder's PCIe round trip. The engine
decides when to create and clear it; pipelines use it when a ``ModelBundle``
carries one.
"""

from __future__ import annotations

from collections import OrderedDict


class ConditioningCache:
    """LRU of prompt key → dict of CPU tensors (VRAM-neutral; the caller moves a
    hit to the device). Not thread-safe; the backend uses one worker thread.
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
