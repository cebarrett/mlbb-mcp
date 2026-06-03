"""
Endpoint wrapper for GET /api/heroes/rank.

Provides two functions consumed by MCP tools:

  fetch_hero_rank_stats  — win/pick/ban rate for one hero  (-> get_hero_winrate)
  fetch_top_heroes       — ranked list of heroes by a stat  (-> get_top_heroes)

Cache strategy
--------------
Both functions share a single underlying fetch: all 132 heroes, sorted by
win_rate desc, for the requested (days, rank) combination. This means one
cache entry covers both use cases — important given the 500 req/day limit.
Sorting for `fetch_top_heroes` is done client-side after the fetch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mlbb.cache import FRESH_TTL_SHORT
from mlbb.client import MLBBError
from mlbb.models import Citation, HeroRef, RankTier, TimeWindow, ToolError

if TYPE_CHECKING:
    from mlbb.client import MLBBClient
    from mlbb.heroes import HeroRoster

log = logging.getLogger(__name__)

# Valid sort fields for fetch_top_heroes
SORT_FIELDS = {"win_rate", "pick_rate", "ban_rate"}


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class HeroRankResult(BaseModel):
    """Win/pick/ban rate for a single hero at a given rank tier and time window."""

    hero: str
    hero_id: int
    win_rate: str   # e.g. "44.94%"
    pick_rate: str  # e.g. "0.68%"
    ban_rate: str   # e.g. "1.54%"
    citation: Citation


class TopHeroEntry(BaseModel):
    position: int   # 1-based rank in the sorted list
    hero: str
    hero_id: int
    win_rate: str
    pick_rate: str
    ban_rate: str


class TopHeroesResult(BaseModel):
    """Ranked list of heroes sorted by a chosen stat."""

    sort_by: str
    rank_tier: str
    time_window_days: int
    heroes: list[TopHeroEntry]
    citation: Citation


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


async def fetch_hero_rank_stats(
    client: "MLBBClient",
    roster: "HeroRoster",
    hero: str | int,
    days: TimeWindow = 7,
    rank: RankTier = RankTier.MYTHIC,
) -> HeroRankResult | ToolError:
    """
    Return win/pick/ban rate for a single hero.

    Parameters
    ----------
    hero:
        Hero name (full or partial) or numeric ID.
    days:
        Time window in days. One of 1, 3, 7, 15, 30.
    rank:
        Rank tier filter.
    """
    # Resolve hero name/ID -> HeroRef (validates input before hitting the API)
    try:
        from mlbb.heroes import AmbiguousHeroError, HeroNotFoundError
        hero_ref = await roster.resolve(hero)
    except HeroNotFoundError as e:
        return ToolError(
            error="hero_not_found",
            message=str(e),
            details={"identifier": str(hero)},
        )
    except AmbiguousHeroError as e:
        return ToolError(
            error="ambiguous_hero",
            message=str(e),
            details={"query": e.query, "candidates": e.candidates},
        )

    # Fetch the full ranked list (shared cache entry for this days+rank pair)
    try:
        raw, citation = await _fetch_all_rank_stats(client, days, rank)
    except MLBBError as e:
        return ToolError(
            error="upstream_unavailable",
            message=f"Could not retrieve rank stats: {e}",
        )

    # Find this hero in the records
    record = _find_hero_record(raw["data"]["records"], hero_ref.id)
    if record is None:
        return ToolError(
            error="hero_not_in_results",
            message=(
                f"{hero_ref.name} was not found in rank stats for "
                f"{rank} over {days} days. "
                "The hero may have too few games to appear in ranked data."
            ),
        )

    d = record["data"]
    citation = citation.model_copy(update={
        "time_window_days": int(days),
        "rank_tier": rank.value,
    })

    return HeroRankResult(
        hero=hero_ref.name,
        hero_id=hero_ref.id,
        win_rate=_pct(d["main_hero_win_rate"]),
        pick_rate=_pct(d["main_hero_appearance_rate"]),
        ban_rate=_pct(d["main_hero_ban_rate"]),
        citation=citation,
    )


async def fetch_top_heroes(
    client: "MLBBClient",
    days: TimeWindow = 7,
    rank: RankTier = RankTier.MYTHIC,
    sort_by: str = "win_rate",
    limit: int = 10,
) -> TopHeroesResult | ToolError:
    """
    Return the top `limit` heroes sorted by a stat.

    Parameters
    ----------
    sort_by:
        One of "win_rate", "pick_rate", "ban_rate".
    limit:
        Number of heroes to return (max 30).
    """
    if sort_by not in SORT_FIELDS:
        return ToolError(
            error="invalid_parameter",
            message=f"sort_by must be one of: {', '.join(sorted(SORT_FIELDS))}",
        )

    limit = max(1, min(limit, 30))

    try:
        raw, citation = await _fetch_all_rank_stats(client, days, rank)
    except MLBBError as e:
        return ToolError(
            error="upstream_unavailable",
            message=f"Could not retrieve rank stats: {e}",
        )

    # Sort client-side (we fetched all heroes, so no extra API call needed)
    stat_key = _stat_key(sort_by)
    records = sorted(
        raw["data"]["records"],
        key=lambda r: r["data"].get(stat_key, 0),
        reverse=True,
    )[:limit]

    heroes = [
        TopHeroEntry(
            position=i + 1,
            hero=r["data"]["main_hero"]["data"]["name"],
            hero_id=r["data"]["main_heroid"],
            win_rate=_pct(r["data"]["main_hero_win_rate"]),
            pick_rate=_pct(r["data"]["main_hero_appearance_rate"]),
            ban_rate=_pct(r["data"]["main_hero_ban_rate"]),
        )
        for i, r in enumerate(records)
    ]

    citation = citation.model_copy(update={
        "time_window_days": int(days),
        "rank_tier": rank.value,
    })

    return TopHeroesResult(
        sort_by=sort_by,
        rank_tier=rank.value,
        time_window_days=int(days),
        heroes=heroes,
        citation=citation,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _fetch_all_rank_stats(
    client: "MLBBClient",
    days: TimeWindow,
    rank: RankTier,
) -> tuple:
    """
    Fetch all 132 heroes from /api/heroes/rank for the given (days, rank).
    This is the shared cache entry — both fetch_hero_rank_stats and
    fetch_top_heroes call this, so one upstream request serves both tools.

    Note: we use rank.value explicitly rather than str(rank). In Python 3.11,
    str(StrEnum.MEMBER) returns "ClassName.MEMBER" rather than the value —
    behaviour that changed from 3.10. Using .value is unambiguous.
    """
    return await client.fetch(
        "api/heroes/rank",
        params={
            "days": str(int(days)),
            "rank": rank.value,
            "size": "132",
            "sort_field": "win_rate",
            "sort_order": "desc",
        },
        fresh_ttl=FRESH_TTL_SHORT,
    )


def _find_hero_record(records: list, hero_id: int) -> dict | None:
    for r in records:
        if r["data"]["main_heroid"] == hero_id:
            return r
    return None


def _pct(value: float) -> str:
    """Format a decimal rate as a percentage string: 0.449381 -> '44.94%'"""
    return f"{value * 100:.2f}%"


def _stat_key(sort_by: str) -> str:
    return {
        "win_rate":  "main_hero_win_rate",
        "pick_rate": "main_hero_appearance_rate",
        "ban_rate":  "main_hero_ban_rate",
    }[sort_by]
