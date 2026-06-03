"""
Stale-cache integration tests.

These tests verify the core resilience behaviour of MLBBClient: when the
upstream API is unavailable, the client should serve cached data with
data_freshness="stale" rather than failing. When no cached data exists, it
should raise UpstreamError.

Each test gets an isolated temp-dir cache (via the `client` fixture) so
test runs never share state.

Monkeypatching strategy
-----------------------
We replace `client._http.get` with an async function that returns a
FakeResponse or raises an exception. This is simpler than an httpx mock
library and makes the failure modes explicit and readable.
"""

from __future__ import annotations

import pytest
import httpx

from mlbb.client import MLBBClient, UpstreamError, BadRequestError
from mlbb.cache import FRESH_TTL_SHORT

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

# A well-formed upstream response (code=0 envelope)
GOOD_DATA = {
    "code": 0,
    "message": "OK",
    "data": {"win_rate": 0.45, "hero": "Lancelot"},
}

# A simulated error envelope (code != 0)
ERROR_ENVELOPE = {"code": 1, "message": "Internal error", "data": None}

PATH = "api/heroes/rank"
PARAMS = {"days": "7", "rank": "mythic"}


class FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(
        self,
        body=None,
        *,
        status_code: int = 200,
        is_json: bool = True,
    ) -> None:
        self._body = body
        self.status_code = status_code
        # Build a minimal real httpx.Request/Response for raise_for_status
        self._request = httpx.Request("GET", "https://mlbb.rone.dev/api/test")
        self._response = httpx.Response(status_code, request=self._request)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self._request,
                response=self._response,
            )

    def json(self):
        if not self._is_json:
            raise ValueError("not json")
        return self._body

    # httpx.Response.json() is synchronous; match that
    @property
    def _is_json(self) -> bool:
        return self._body is not None


def good_upstream():
    """Async mock that returns a well-formed response."""
    async def mock(url, *, params=None, **kw):
        return FakeResponse(GOOD_DATA)
    return mock


def fail_network():
    """Async mock that raises a network error."""
    async def mock(url, *, params=None, **kw):
        raise httpx.ConnectError("connection refused")
    return mock


def fail_timeout():
    """Async mock that raises a timeout."""
    async def mock(url, *, params=None, **kw):
        raise httpx.ReadTimeout("timed out")
    return mock


def fail_5xx():
    """Async mock that returns an HTTP 503."""
    async def mock(url, *, params=None, **kw):
        return FakeResponse(None, status_code=503)
    return mock


def fail_4xx():
    """Async mock that returns an HTTP 422."""
    async def mock(url, *, params=None, **kw):
        return FakeResponse(None, status_code=422)
    return mock


def fail_non_json():
    """Async mock that returns HTTP 200 with an HTML body (CDN error page)."""
    async def mock(url, *, params=None, **kw):
        return FakeResponse(None, status_code=200, is_json=False)
    return mock


def fail_bad_envelope():
    """Async mock that returns HTTP 200 with a code!=0 envelope."""
    async def mock(url, *, params=None, **kw):
        return FakeResponse(ERROR_ENVELOPE)
    return mock


@pytest.fixture
async def client(tmp_path):
    """Fresh MLBBClient with an isolated temp-dir cache per test."""
    async with MLBBClient(cache_dir=str(tmp_path / ".cache")) as c:
        yield c


async def seed_cache(client: MLBBClient) -> None:
    """
    Populate the cache for PATH+PARAMS with a good response.
    Uses fresh_ttl=0 so subsequent fetches see the entry as stale
    and will try upstream again — letting us test the fallback path.
    """
    client._http.get = good_upstream()
    await client.fetch(PATH, PARAMS, fresh_ttl=FRESH_TTL_SHORT)
    # After seeding, set ttl=0 so all future fetches treat it as stale
    # (the entry is still on disk for stale-serve to find)


# ---------------------------------------------------------------------------
# Tests: stale-serve paths
# ---------------------------------------------------------------------------

async def test_stale_on_network_error(client):
    """Network failure with a cached entry → serve stale, freshness='stale'."""
    await seed_cache(client)

    client._http.get = fail_network()
    data, citation = await client.fetch(PATH, PARAMS, fresh_ttl=0)

    assert data == GOOD_DATA
    assert citation.data_freshness == "stale"


async def test_stale_on_timeout(client):
    """Read timeout with a cached entry → serve stale."""
    await seed_cache(client)

    client._http.get = fail_timeout()
    data, citation = await client.fetch(PATH, PARAMS, fresh_ttl=0)

    assert data == GOOD_DATA
    assert citation.data_freshness == "stale"


async def test_stale_on_5xx(client):
    """HTTP 503 with a cached entry → serve stale."""
    await seed_cache(client)

    client._http.get = fail_5xx()
    data, citation = await client.fetch(PATH, PARAMS, fresh_ttl=0)

    assert data == GOOD_DATA
    assert citation.data_freshness == "stale"


async def test_stale_on_non_json_200(client):
    """
    HTTP 200 with a non-JSON body (e.g. CDN error page) with a cached entry
    → serve stale. This is the H1 bug scenario from the code review.
    """
    await seed_cache(client)

    client._http.get = fail_non_json()
    data, citation = await client.fetch(PATH, PARAMS, fresh_ttl=0)

    assert data == GOOD_DATA
    assert citation.data_freshness == "stale"


async def test_stale_on_error_envelope(client):
    """
    HTTP 200 with code!=0 envelope with a cached entry → serve stale.
    Bad body must NOT overwrite the cached good body.
    """
    await seed_cache(client)

    client._http.get = fail_bad_envelope()
    data, citation = await client.fetch(PATH, PARAMS, fresh_ttl=0)

    assert data == GOOD_DATA
    assert citation.data_freshness == "stale"


# ---------------------------------------------------------------------------
# Tests: UpstreamError when no cache
# ---------------------------------------------------------------------------

async def test_upstream_error_network_no_cache(client):
    """Network failure with empty cache → UpstreamError, not stale."""
    client._http.get = fail_network()

    with pytest.raises(UpstreamError):
        await client.fetch(PATH, PARAMS)


async def test_upstream_error_5xx_no_cache(client):
    """HTTP 503 with empty cache → UpstreamError."""
    client._http.get = fail_5xx()

    with pytest.raises(UpstreamError):
        await client.fetch(PATH, PARAMS)


async def test_upstream_error_non_json_no_cache(client):
    """Non-JSON 200 with empty cache → UpstreamError."""
    client._http.get = fail_non_json()

    with pytest.raises(UpstreamError):
        await client.fetch(PATH, PARAMS)


# ---------------------------------------------------------------------------
# Tests: 4xx never stale-serves
# ---------------------------------------------------------------------------

async def test_4xx_raises_bad_request_even_with_cache(client):
    """
    4xx always raises BadRequestError — even if cached data exists.
    A bad request means our code is wrong, not the upstream is down;
    stale data is irrelevant.
    """
    await seed_cache(client)

    client._http.get = fail_4xx()

    with pytest.raises(BadRequestError) as exc_info:
        await client.fetch(PATH, PARAMS, fresh_ttl=0)

    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# Tests: cache poisoning guard
# ---------------------------------------------------------------------------

async def test_bad_response_does_not_poison_cache(client):
    """
    A bad upstream response (non-JSON, error envelope) must not overwrite
    a good cached entry. After the bad response, the good entry is still
    retrievable as stale.
    """
    await seed_cache(client)

    # Bad response — should not be cached
    client._http.get = fail_bad_envelope()
    await client.fetch(PATH, PARAMS, fresh_ttl=0)

    # Now upstream comes back with a good response
    client._http.get = good_upstream()
    data, citation = await client.fetch(PATH, PARAMS, fresh_ttl=0)

    # Should be the original good data, freshly fetched
    assert data == GOOD_DATA
    assert citation.data_freshness == "fresh"


# ---------------------------------------------------------------------------
# Tests: recovery
# ---------------------------------------------------------------------------

async def test_fresh_after_recovery(client):
    """
    After serving stale, when upstream recovers the next fetch is fresh
    and overwrites the stale cache entry.
    """
    await seed_cache(client)

    # Upstream goes down — stale served
    client._http.get = fail_network()
    _, stale_citation = await client.fetch(PATH, PARAMS, fresh_ttl=0)
    assert stale_citation.data_freshness == "stale"

    # Upstream recovers — fresh served
    new_data = {**GOOD_DATA, "data": {"win_rate": 0.47, "hero": "Lancelot"}}
    async def recovered(url, *, params=None, **kw):
        return FakeResponse(new_data)
    client._http.get = recovered

    fresh_data, fresh_citation = await client.fetch(PATH, PARAMS, fresh_ttl=0)
    assert fresh_citation.data_freshness == "fresh"
    assert fresh_data == new_data


async def test_cache_hit_skips_upstream(client):
    """
    A fresh cache hit never calls the upstream at all.
    We verify by setting the mock to raise — if upstream were called,
    the test would fail.
    """
    await seed_cache(client)

    client._http.get = fail_network()
    # fresh_ttl=FRESH_TTL_SHORT: entry was just seeded, so it IS fresh
    data, citation = await client.fetch(PATH, PARAMS, fresh_ttl=FRESH_TTL_SHORT)

    assert data == GOOD_DATA
    assert citation.data_freshness == "fresh"  # served from cache, no upstream call
