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


class BoundedLruCache:
    """Saikeisturvallinen LRU-valimuisti kiintealla maksimikoolla."""

    def __init__(self, max_items: int) -> None:
        self._max_items = max(1, int(max_items))
        self._data: OrderedDict = OrderedDict()
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
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": int(self._hits),
                "misses": int(self._misses),
                "size": len(self._data),
                "max_items": int(self._max_items),
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
