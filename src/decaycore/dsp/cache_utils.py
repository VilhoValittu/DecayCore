# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0
"""Yhtenainen rajattu LRU-valimuisti DSP-moduulien moduulitason cacheille.

Korvaa aiemmat erilliset toteutukset (plain dict + clear-all, dict + FIFO,
OrderedDict + move_to_end), joilla oli keskenaan eri evict-politiikat.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Hashable

__all__ = ["BoundedLruCache"]


def _retained_nbytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate bytes retained by NumPy-like arrays inside a cache value."""
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return max(0, int(nbytes))
        except (TypeError, ValueError, OverflowError):
            return 0
    if isinstance(value, dict):
        return sum(_retained_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(_retained_nbytes(item, seen) for item in value)
    return 0


class BoundedLruCache:
    """Thread-safe LRU bounded by item count and optionally retained bytes."""

    def __init__(self, max_items: int, *, max_bytes: int | None = None) -> None:
        self._max_items = max(1, int(max_items))
        self._max_bytes = None if max_bytes is None else max(0, int(max_bytes))
        self._data: OrderedDict = OrderedDict()
        self._weights: dict[Hashable, int] = {}
        self._retained_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            try:
                value = self._data[key]
            except KeyError:
                self._misses += 1
                return default
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            weight = _retained_nbytes(value)
            if self._max_bytes is not None and weight > self._max_bytes:
                if key in self._data:
                    self._data.pop(key)
                    self._retained_bytes -= self._weights.pop(key, 0)
                return
            if key in self._data:
                self._retained_bytes -= self._weights.pop(key, 0)
                self._data.move_to_end(key)
            self._data[key] = value
            self._weights[key] = weight
            self._retained_bytes += weight
            while len(self._data) > self._max_items or (
                self._max_bytes is not None and self._retained_bytes > self._max_bytes
            ):
                old_key, _old_value = self._data.popitem(last=False)
                self._retained_bytes -= self._weights.pop(old_key, 0)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._weights.clear()
            self._retained_bytes = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": int(self._hits),
                "misses": int(self._misses),
                "size": len(self._data),
                "max_items": int(self._max_items),
                "retained_bytes": int(self._retained_bytes),
                "max_bytes": self._max_bytes,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._data

    # Dict-yhteensopiva rajapinta, jotta luokka toimii drop-in-korvaajana
    # aiemmille moduulitason dict/OrderedDict-cacheille.

    def __getitem__(self, key: Hashable) -> Any:
        with self._lock:
            value = self._data[key]
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.put(key, value)

    def __iter__(self):
        with self._lock:
            return iter(list(self._data))

    def __eq__(self, other: object) -> bool:
        with self._lock:
            if isinstance(other, BoundedLruCache):
                return dict(self._data) == dict(other._data)
            if isinstance(other, dict):
                return dict(self._data) == other
            return NotImplemented

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def values(self):
        with self._lock:
            return list(self._data.values())

    def items(self):
        with self._lock:
            return list(self._data.items())
