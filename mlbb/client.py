"""
Async HTTP client for mlbb.rone.dev with cache + stale-serve.

Usage pattern (endpoint modules follow this template):

    async with MLBBClient() as client:
        data, citation = await client.fetch(
            "api/heroes/rank",
            params={"days": "7", "rank": "mythic"},
            fresh_ttl=FRESH_TTL_SHORT,
        )
        # augment citation with query-specific fields
        citation = citation.model_copy(update={
            "time_window_days": 7,
            "rank_tier": "mythic",
        })

Error handling
--------------
All errors raised to callers subclass MLBBError, so endpoint wrappers can
catch a single type:

  - UpstreamError   — upstream unusable (unreachable, timeout, 5xx, non-JSON
                      body, or an error envelope with code != 0) AND no cached
                      data to fall back on. When a cached entry *does* exist,
                      these conditions stale-serve instead of raising.
  - BadRequestError — upstream returned 4xx. Indicates a bug in our request
                      construction, not an outage; stale data is irrelevant,
                      so this always raises.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import httpx

from mlbb.cache import Cache, CacheEntry, FRESH_TTL_SHORT, FRESH_TTL_LONG
from mlbb.models import Citation

log = logging.getLogger(__name__)

_UTC = datetime.timezone.utc

# Re-export so callers can import TTL constants from one place.
__all__ = [
    "MLBBClient",
    "MLBBError",
    "UpstreamError",
    "BadRequestError",
    "FRESH_TTL_SHORT",
    "FRESH_TTL_LONG",
]


class MLBBError(Exception):
    """
    Base class for all errors raised by MLBBClient.

    Endpoint wrappers catch this single type and convert it to a ToolError,
    rather than juggling httpx exceptions and library exceptions separately.
    """


class UpstreamError(MLBBError):
    """
    Upstream is unusable — unreachable, timed out, returned a 5xx, served a
    non-JSON body, or returned an error envelope (code != 0) — AND there is no
    cached data to fall back on.

    When a cached entry exists, these conditions serve stale data instead of
    raising.
    """


class BadRequestError(MLBBError):
    """
    Upstream returned a 4xx status, meaning our request was malformed (bad
    params, unknown path, etc.). This is a bug on our side, not an outage, so
    stale data is not served.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _BadResponse(Exception):
    """
    Internal signal: upstream returned 2xx but the body is unusable (non-JSON
    or an error envelope). Never escapes fetch() — it is converted to a
    stale-serve result or an UpstreamError.
    """


class MLBBClient:
    """
    Async client for mlbb.rone.dev.

    Owns an httpx.AsyncClient (connection pooling) and a Cache instance.
    Use as an async context manager; or call aclose() when done.

    Parameters
    ----------
    base_url:
        API base. Switch to https://openmlbb.fastapicloud.dev for >500 req/day.
    cache_dir:
        Directory for the disk cache. Relative paths are resolved from the
        working directory. Defaults to ".cache" (gitignored).
    timeout:
        Per-request timeout in seconds.
    """

    DEFAULT_BASE_URL = "https://mlbb.rone.dev"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: str = ".cache",
        timeout: float = 10.0,
    ) -> None:
        self._cache = Cache(cache_dir)
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"User-Agent": "mlbb-mcp/0.1"},
            follow_redirects=False,
        )

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    async def fetch(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        fresh_ttl: int = FRESH_TTL_SHORT,
    ) -> tuple[Any, Citation]:
        """
        Fetch `path` with caching and stale-serve fallback.

        Returns (data, citation). The citation has source/attribution/
        retrieved_at/data_freshness filled in. Callers should augment it
        with query-specific fields:

            citation = citation.model_copy(update={
                "time_window_days": 7, "rank_tier": "mythic"
            })

        Parameters
        ----------
        path:
            API path without leading slash, e.g. "api/heroes/rank".
        params:
            Query parameters. All values are coerced to strings for
            cache-key stability (so `{"days": 7}` and `{"days": "7"}` hit
            the same cache entry).
        fresh_ttl:
            How long (seconds) a cached entry is considered fresh. Use
            FRESH_TTL_SHORT (1h) for dynamic data, FRESH_TTL_LONG (24h)
            for static data.
        """
        norm = _normalize_params(params)
        key = self._cache.make_key(path, norm)
        entry = self._cache.get(key)

        # --- Cache hit (fresh) ---
        if entry and self._cache.is_fresh(entry, fresh_ttl):
            log.debug("cache hit (fresh): %s", key)
            return entry["data"], _citation(entry, self._cache, "fresh")

        # --- Upstream fetch ---
        try:
            data = await self._get(path, norm)
            self._cache.set(key, data)
            log.debug("upstream fetch ok: %s", key)
            return data, _fresh_citation_now()

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # 4xx = bad request (our bug); don't stale-serve. Surface it as a
            # typed error so endpoints catch one exception family (MLBBError).
            if status < 500:
                raise BadRequestError(
                    f"Upstream rejected request for '{key}' (HTTP {status})",
                    status_code=status,
                ) from exc
            # 5xx = server-side problem; fall through to stale-serve.
            log.warning("upstream 5xx for %s: %s", key, exc)
            return _stale_or_raise(entry, self._cache, key, exc)

        except _BadResponse as exc:
            # 2xx but the body is unusable (non-JSON or code != 0). Treat like a
            # 5xx: serve stale if we can, otherwise raise UpstreamError. Note the
            # cache was NOT written, so a bad body never poisons the cache.
            log.warning("unusable upstream response for %s: %s", key, exc)
            return _stale_or_raise(entry, self._cache, key, exc)

        except (httpx.TransportError, httpx.TimeoutException) as exc:
            log.warning("upstream unreachable for %s: %s", key, exc)
            return _stale_or_raise(entry, self._cache, key, exc)

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict[str, str]) -> Any:
        """
        GET, parse JSON, and validate the response envelope.

        Returns the validated envelope dict (guaranteed ``code == 0``).

        Raises
        ------
        httpx.HTTPStatusError
            Non-2xx status.
        _BadResponse
            2xx but the body is non-JSON or carries an error envelope
            (``code != 0``). Routed to stale-serve by the caller.
        """
        resp = await self._http.get(f"/{path}", params=params)
        resp.raise_for_status()

        try:
            body = resp.json()
        except ValueError as exc:
            # A 200 with a non-JSON body — e.g. a CDN/proxy error page in front
            # of a struggling backend. Unusable; signal for stale-serve.
            raise _BadResponse(f"non-JSON body (HTTP {resp.status_code})") from exc

        # Success envelope is code == 0. Anything else (including a non-dict
        # body) is an application-level error we must not cache or return.
        if not isinstance(body, dict) or body.get("code") != 0:
            code = body.get("code") if isinstance(body, dict) else None
            message = body.get("message") if isinstance(body, dict) else None
            raise _BadResponse(f"error envelope (code={code!r}, message={message!r})")

        return body

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        await self._http.aclose()
        self._cache.close()

    async def __aenter__(self) -> "MLBBClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return (
            f"MLBBClient(base_url={self._http.base_url!r}, "
            f"cache={self._cache!r})"
        )


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _normalize_params(params: dict[str, Any] | None) -> dict[str, str]:
    """Coerce all param values to strings for cache-key stability."""
    if not params:
        return {}
    return {k: str(v) for k, v in params.items()}


def _citation(
    entry: CacheEntry,
    cache: Cache,
    freshness: str,
) -> Citation:
    return Citation(
        retrieved_at=cache.stored_at(entry),
        data_freshness=freshness,  # type: ignore[arg-type]
    )


def _fresh_citation_now() -> Citation:
    return Citation(
        retrieved_at=datetime.datetime.now(_UTC),
        data_freshness="fresh",
    )


def _stale_or_raise(
    entry: CacheEntry | None,
    cache: Cache,
    key: str,
    exc: Exception,
) -> tuple[Any, Citation]:
    if entry:
        log.warning("serving stale cache entry for %s", key)
        return entry["data"], _citation(entry, cache, "stale")
    raise UpstreamError(
        f"Upstream unavailable and no cached data for key '{key}'"
    ) from exc
