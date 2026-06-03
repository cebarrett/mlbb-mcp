"""
Disk-based cache with TTL and stale-serve support.

Design
------
Backend: diskcache.Cache (file-backed key/value store, no server required).

Each entry stores the raw data plus a `stored_at` timestamp. The cache
itself uses a long absolute eviction TTL (ABSOLUTE_TTL) so entries stick
around for stale-serving even after their freshness window expires.

Freshness is checked by the caller, not the cache:

    entry = cache.get(key)
    if entry and cache.is_fresh(entry, ttl=FRESH_TTL_SHORT):
        return entry["data"], "fresh"
    try:
        data = await fetch_upstream(...)
        cache.set(key, data)
        return data, "fresh"
    except UpstreamError:
        if entry:                        # stale but better than nothing
            return entry["data"], "stale"
        raise

This keeps the stale-serve logic visible in the client rather than hidden
inside the cache, making it easier to reason about and test.

Note on async: diskcache is synchronous. We call it directly from async
code because individual cache ops are fast (sub-millisecond for hits).
Wrap in asyncio.to_thread() if this ever becomes a bottleneck.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, TypedDict

import diskcache


# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------

FRESH_TTL_SHORT: int = 3_600       # 1 hour  — dynamic: rank stats, win rates
FRESH_TTL_LONG: int  = 86_400      # 24 hours — static: hero profiles, builds
ABSOLUTE_TTL: int    = 7 * 86_400  # 7 days   — eviction horizon for stale fallback


# ---------------------------------------------------------------------------
# Entry type
# ---------------------------------------------------------------------------

class CacheEntry(TypedDict):
    data: Any
    stored_at: str  # ISO 8601 UTC string, e.g. "2026-06-03T03:09:30.700178+00:00"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class Cache:
    """
    Thin wrapper around diskcache.Cache that adds stored_at tracking and
    structured key construction.
    """

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self._dc = diskcache.Cache(str(cache_dir))

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    def make_key(self, path: str, params: dict[str, Any] | None = None) -> str:
        """
        Build a deterministic cache key from an API path + query params.

        Uses JSON serialisation so the key is unambiguous regardless of what
        characters appear in keys or values (avoids separator-collision bugs
        that arise with pipe-delimited strings).

        >>> cache.make_key("heroes/rank", {"rank": "mythic", "days": 7})
        '{"path": "heroes/rank", "params": {"days": 7, "rank": "mythic"}}'
        """
        import json
        return json.dumps(
            {"path": path, "params": dict(sorted((params or {}).items()))},
            separators=(",", ":"),
        )

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str) -> CacheEntry | None:
        """
        Return the cached entry for key, or None if absent/evicted.

        Returns entries regardless of freshness — the caller decides
        whether a stale entry is acceptable via is_fresh().
        """
        return self._dc.get(key)  # type: ignore[return-value]

    def set(self, key: str, data: Any) -> None:
        """
        Store data under key with an ABSOLUTE_TTL eviction window.
        Only call this with a successful upstream response.
        """
        entry: CacheEntry = {
            "data": data,
            "stored_at": _utcnow().isoformat(),
        }
        self._dc.set(key, entry, expire=ABSOLUTE_TTL)

    # ------------------------------------------------------------------
    # Freshness helpers
    # ------------------------------------------------------------------

    def is_fresh(self, entry: CacheEntry, max_age_seconds: int) -> bool:
        """True if the entry was stored within max_age_seconds ago."""
        age = _utcnow() - datetime.datetime.fromisoformat(entry["stored_at"])
        return age.total_seconds() < max_age_seconds

    def stored_at(self, entry: CacheEntry) -> datetime.datetime:
        """Parse stored_at into a datetime (for use in Citation.retrieved_at)."""
        return datetime.datetime.fromisoformat(entry["stored_at"])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._dc.close()

    def __len__(self) -> int:
        return len(self._dc)

    def __repr__(self) -> str:
        return f"Cache(dir={self._dc.directory!r}, entries={len(self)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)
