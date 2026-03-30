"""Simple in-memory LRU cache for RAG responses.

Key = SHA256(question + server + model)
TTL = 300 seconds
Max 100 entries
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any


class RAGCache:
    """Thread-safe LRU cache with TTL expiration."""

    def __init__(self, max_size: int = 100, ttl: int = 300) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    @staticmethod
    def make_key(question: str, server: str, model: str) -> str:
        raw = f"{question}:{server}:{model}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing/expired."""
        if key not in self._store:
            return None
        value, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        # Move to end to mark as recently used
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        """Insert or update a cache entry, evicting LRU if at capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time() + self._ttl)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)
