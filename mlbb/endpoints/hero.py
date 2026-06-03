"""
Endpoint wrappers for hero-specific dynamic stats.

  /api/heroes/{id}/counters  ->  fetch_hero_counters, fetch_hero_synergies
  /api/heroes/{id}/trends    ->  fetch_hero_trends
  /api/heroes/{id}           ->  fetch_hero_profile

Counter vs synergy semantics
-----------------------------
The /counters endpoint returns two sub-arrays in one response:
  sub_hero       — allies that boost this hero's win rate (synergies)
  sub_hero_last  — enemies that reduce this hero's win rate (counters)

`increase_win_rate` is always from the queried hero's perspective:
  +0.033 in sub_hero      = "Lancelot wins 3.3% more often when Atlas is an ally"
  -0.035 in sub_hero_last = "Lancelot wins 3.5% less often when Phoveus is an enemy"

Because both tools hit the same endpoint, they share one cache entry per
(hero, days, rank) triple — two tool calls for the same hero don't double-fetch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

import re

from mlbb.cache import FRESH_TTL_LONG, FRESH_TTL_SHORT
from mlbb.client import MLBBError
from mlbb.models import Citation, RankTier, TimeWindow, ToolError

if TYPE_CHECKING:
    from mlbb.client import MLBBClient
    from mlbb.heroes import HeroRoster

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class HeroMatchupEntry(BaseModel):
    hero: str
    hero_id: int
    win_rate_delta: str   # e.g. "+3.35%" or "-3.46%" (effect on queried hero)
    their_win_rate: str   # this hero's overall win rate in the dataset


class HeroCountersResult(BaseModel):
    """Heroes that counter the queried hero (reduce its win rate)."""

    hero: str
    hero_id: int
    hero_win_rate: str
    counters: list[HeroMatchupEntry]
    citation: Citation


class HeroSynergiesResult(BaseModel):
    """Heroes that synergize with the queried hero (boost its win rate)."""

    hero: str
    hero_id: int
    hero_win_rate: str
    synergies: list[HeroMatchupEntry]
    citation: Citation


class TrendEntry(BaseModel):
    date: str        # "YYYY-MM-DD"
    win_rate: str    # e.g. "44.80%"
    pick_rate: str
    ban_rate: str


class HeroTrendsResult(BaseModel):
    """Day-by-day win/pick/ban rate trend for a hero."""

    hero: str
    hero_id: int
    days: int
    rank_tier: str
    trend: list[TrendEntry]   # newest date first
    citation: Citation


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


async def fetch_hero_counters(
    client: "MLBBClient",
    roster: "HeroRoster",
    hero: str | int,
    days: TimeWindow = 7,
    rank: RankTier = RankTier.MYTHIC,
) -> HeroCountersResult | ToolError:
    """
    Return the heroes that most reduce a hero's win rate (i.e. counter it).

    These come from the `sub_hero_last` array in /api/heroes/{id}/counters,
    where increase_win_rate is negative — meaning the queried hero wins less
    often when facing them.
    """
    hero_ref, error = await _resolve(roster, hero)
    if error:
        return error

    try:
        raw, citation = await _fetch_counters_raw(client, hero_ref.id, days, rank)
    except MLBBError as e:
        return ToolError(error="upstream_unavailable", message=str(e))

    d = raw["data"]["records"][0]["data"]
    entries = await _build_matchup_entries(roster, d["sub_hero_last"])

    return HeroCountersResult(
        hero=hero_ref.name,
        hero_id=hero_ref.id,
        hero_win_rate=_pct(d["main_hero_win_rate"]),
        counters=entries,
        citation=_augment(citation, days, rank),
    )


async def fetch_hero_synergies(
    client: "MLBBClient",
    roster: "HeroRoster",
    hero: str | int,
    days: TimeWindow = 7,
    rank: RankTier = RankTier.MYTHIC,
) -> HeroSynergiesResult | ToolError:
    """
    Return the heroes that most increase a hero's win rate (i.e. synergize).

    These come from the `sub_hero` array in /api/heroes/{id}/counters,
    where increase_win_rate is positive — meaning the queried hero wins more
    often when they are allies.
    """
    hero_ref, error = await _resolve(roster, hero)
    if error:
        return error

    try:
        raw, citation = await _fetch_counters_raw(client, hero_ref.id, days, rank)
    except MLBBError as e:
        return ToolError(error="upstream_unavailable", message=str(e))

    d = raw["data"]["records"][0]["data"]
    entries = await _build_matchup_entries(roster, d["sub_hero"])

    return HeroSynergiesResult(
        hero=hero_ref.name,
        hero_id=hero_ref.id,
        hero_win_rate=_pct(d["main_hero_win_rate"]),
        synergies=entries,
        citation=_augment(citation, days, rank),
    )


async def fetch_hero_trends(
    client: "MLBBClient",
    roster: "HeroRoster",
    hero: str | int,
    days: TimeWindow = 7,
    rank: RankTier = RankTier.MYTHIC,
) -> HeroTrendsResult | ToolError:
    """
    Return day-by-day win/pick/ban rate for a hero over the given window.

    Useful for "is this hero trending up or down?" questions. The trend array
    is ordered newest-date-first.
    """
    hero_ref, error = await _resolve(roster, hero)
    if error:
        return error

    try:
        raw, citation = await client.fetch(
            f"api/heroes/{hero_ref.id}/trends",
            params={"days": str(int(days)), "rank": rank.value},
            fresh_ttl=FRESH_TTL_SHORT,
        )
    except MLBBError as e:
        return ToolError(error="upstream_unavailable", message=str(e))

    records = raw["data"]["records"]
    if not records:
        return ToolError(
            error="no_data",
            message=f"No trend data found for {hero_ref.name} at {rank.value} rank.",
        )

    trend_data = records[0]["data"]["win_rate"]  # already newest-first from API
    trend = [
        TrendEntry(
            date=entry["date"],
            win_rate=_pct(entry["win_rate"]),
            pick_rate=_pct(entry["app_rate"]),
            ban_rate=_pct(entry["ban_rate"]),
        )
        for entry in trend_data
    ]

    return HeroTrendsResult(
        hero=hero_ref.name,
        hero_id=hero_ref.id,
        days=int(days),
        rank_tier=rank.value,
        trend=trend,
        citation=_augment(citation, days, rank),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _fetch_counters_raw(
    client: "MLBBClient",
    hero_id: int,
    days: TimeWindow,
    rank: RankTier,
) -> tuple:
    """
    Fetch /api/heroes/{id}/counters. Shared by fetch_hero_counters and
    fetch_hero_synergies so both tools hit the same cache entry.
    """
    return await client.fetch(
        f"api/heroes/{hero_id}/counters",
        params={"days": str(int(days)), "rank": rank.value},
        fresh_ttl=FRESH_TTL_SHORT,
    )


async def _resolve(
    roster: "HeroRoster",
    hero: str | int,
) -> tuple:
    """
    Resolve hero identifier to a HeroRef.
    Returns (hero_ref, None) on success, (None, ToolError) on failure.
    """
    from mlbb.heroes import AmbiguousHeroError, HeroNotFoundError
    try:
        return await roster.resolve(hero), None
    except HeroNotFoundError as e:
        return None, ToolError(
            error="hero_not_found",
            message=str(e),
            details={"identifier": str(hero)},
        )
    except AmbiguousHeroError as e:
        return None, ToolError(
            error="ambiguous_hero",
            message=str(e),
            details={"query": e.query, "candidates": e.candidates},
        )


async def _build_matchup_entries(
    roster: "HeroRoster",
    sub_heroes: list[dict],
) -> list[HeroMatchupEntry]:
    """
    Convert sub_hero / sub_hero_last dicts into typed HeroMatchupEntry objects,
    resolving hero IDs to names via the roster.
    """
    entries = []
    for s in sub_heroes:
        hero_id = s["heroid"]
        name = await roster.name_for_id(hero_id)
        delta = s["increase_win_rate"]
        entries.append(HeroMatchupEntry(
            hero=name,
            hero_id=hero_id,
            win_rate_delta=f"{delta:+.2%}",   # e.g. "+3.35%" or "-3.46%"
            their_win_rate=_pct(s["hero_win_rate"]),
        ))
    return entries


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _augment(citation: Citation, days: TimeWindow, rank: RankTier) -> Citation:
    return citation.model_copy(update={
        "time_window_days": int(days),
        "rank_tier": rank.value,
    })


# ---------------------------------------------------------------------------
# Hero profile — output models + fetch function
# ---------------------------------------------------------------------------


class SkillInfo(BaseModel):
    name: str
    description: str    # HTML tags stripped
    cooldown_cost: str  # e.g. "CD: 14   Mana Cost: 40", empty string for passives


class HeroProfileResult(BaseModel):
    """Static hero data: role, lane, skills, lore. Long cache TTL (24h)."""

    hero: str
    hero_id: int
    role: list[str]         # e.g. ["Assassin"]
    lane: list[str]         # e.g. ["Jungle"]
    specialties: list[str]  # e.g. ["Chase", "Burst"]
    difficulty: str         # "Low" | "Medium" | "High"
    story: str
    skills: list[SkillInfo]
    citation: Citation


async def fetch_hero_profile(
    client: "MLBBClient",
    roster: "HeroRoster",
    hero: str | int,
) -> HeroProfileResult | ToolError:
    """
    Return static profile data for a hero: role, lane, skills, lore.

    Uses a 24h cache TTL — hero stats and skills only change on patch day.
    """
    hero_ref, error = await _resolve(roster, hero)
    if error:
        return error

    try:
        raw, citation = await client.fetch(
            f"api/heroes/{hero_ref.id}",
            fresh_ttl=FRESH_TTL_LONG,
        )
    except MLBBError as e:
        return ToolError(error="upstream_unavailable", message=str(e))

    records = raw["data"]["records"]
    if not records:
        return ToolError(
            error="no_data",
            message=f"No profile data found for {hero_ref.name}.",
        )

    data = records[0]["data"]["hero"]["data"]

    # Role and lane: filter out empty strings the API sometimes includes
    role = [r for r in data.get("sortlabel", []) if r]
    lane = [ln for ln in data.get("roadsortlabel", []) if ln]
    specialties = [s for s in data.get("speciality", []) if s]

    # Skills
    skill_lists = data.get("heroskilllist", [])
    raw_skills = skill_lists[0]["skilllist"] if skill_lists else []
    skills = [
        SkillInfo(
            name=s["skillname"][:100],
            description=_strip_html(s.get("skilldesc", ""))[:500],
            cooldown_cost=s.get("skillcd&cost", "")[:80],
        )
        for s in raw_skills
    ]

    return HeroProfileResult(
        hero=hero_ref.name,
        hero_id=hero_ref.id,
        role=role,
        lane=lane,
        specialties=specialties,
        difficulty=_difficulty_label(data.get("difficulty", "0")),
        story=data.get("story", "")[:300],
        skills=skills,
        citation=citation.model_copy(update={"time_window_days": None, "rank_tier": None}),
    )


def _strip_html(text: str) -> str:
    """Remove HTML font/color tags from skill descriptions."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _difficulty_label(raw: str) -> str:
    """Map 0-100 difficulty score to a human label."""
    try:
        score = int(raw)
    except (ValueError, TypeError):
        return "Unknown"
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Medium"
    return "High"
