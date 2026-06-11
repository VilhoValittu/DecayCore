"""Testit jaetulle BoundedLruCache-valimuistille."""

from decaycore.dsp.cache_utils import BoundedLruCache


def test_bounded_lru_cache_basic_roundtrip():
    cache = BoundedLruCache(4)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None
    assert cache.get("missing", "fallback") == "fallback"
    assert "a" in cache
    assert len(cache) == 1


def test_bounded_lru_cache_evicts_oldest_and_respects_recent_use():
    cache = BoundedLruCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # "a" tuoreutuu, "b" on nyt vanhin
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_bounded_lru_cache_overwrite_does_not_grow():
    cache = BoundedLruCache(2)
    cache.put("a", 1)
    cache.put("a", 2)
    assert cache.get("a") == 2
    assert len(cache) == 1


def test_bounded_lru_cache_clear_and_stats():
    cache = BoundedLruCache(8)
    cache.put("a", 1)
    cache.get("a")
    cache.get("nope")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["max_items"] == 8
    cache.clear()
    assert len(cache) == 0
