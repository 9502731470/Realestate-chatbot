"""Thread-safe TTL + LRU cache (saves scarce free-model requests on repeats)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import TypeVar

V = TypeVar("V")


class TTLCache[V]:
    def __init__(self, max_entries: int = 512, ttl_s: float = 6 * 3600) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> V | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] < time.monotonic():
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value: V) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
