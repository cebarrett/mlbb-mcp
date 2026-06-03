"""
Endpoint wrapper for GET /api/academy/heroes/{id}/builds.

Lane inference
--------------
The builds endpoint requires a `lane` param. When the caller omits it, we
fetch the hero profile to read `roadsortlabel` and map it to the API's enum
(exp | mid | roam | jungle | gold). The `lane_inferred` field in the result
flags this so the LLM can mention it in its answer.

Equipment lookup
----------------
The API returns build items as integer IDs. EquipmentLookup resolves them to
names by fetching /api/academy/equipment once (24h cache). Pass one instance
per server lifetime — don't construct it per-request.
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mlbb.cache import FRESH_TTL_LONG, FRESH_TTL_SHORT
from mlbb.client import MLBBError
from mlbb.models import Citation, RankTier, ToolError

if TYPE_CHECKING:
    from mlbb.client import MLBBClient
    from mlbb.heroes import HeroRoster

log = logging.getLogger(__name__)

# Valid lane values accepted by the builds endpoint
VALID_LANES = ("exp", "mid", "roam", "jungle", "gold")

# Map from hero profile roadsortlabel strings -> API lane param
_LANE_LABEL_MAP: dict[str, str] = {
    "jungle":    "jungle",
    "gold lane": "gold",
    "exp lane":  "exp",
    "mid lane":  "mid",
    "roam":      "roam",
}


# ---------------------------------------------------------------------------
# Equipment lookup
# ---------------------------------------------------------------------------


class EquipmentLookup:
    """
    Lazy-loaded mapping of equipment IDs to names.

    Constructed once at server startup and reused across all tool calls.
    Uses a 24h cache TTL — item names only change on patch day.
    """

    def __init__(self, client: "MLBBClient") -> None:
        self._client = client
        self._map: dict[int, str] = {}
        self._loaded = False

    async def name_for_id(self, equip_id: int) -> str:
        await self._ensure_loaded()
        return self._map.get(equip_id, f"Item#{equip_id}")

    async def names_for_ids(self, equip_ids: list[int]) -> list[str]:
        await self._ensure_loaded()
        return [self._map.get(eid, f"Item#{eid}") for eid in equip_ids]

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self._load()

    async def _load(self) -> None:
        log.debug("loading equipment lookup")
        data, _ = await self._client.fetch(
            "api/academy/equipment",
            params={"size": "200"},
            fresh_ttl=FRESH_TTL_LONG,
        )
        for record in data["data"]["records"]:
            d = record["data"]
            self._map[d["equipid"]] = d["equipname"]
        self._loaded = True
        log.debug("equipment lookup loaded: %d items", len(self._map))


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class BuildEntry(BaseModel):
    items: list[str]    # item names, as returned by the API (typically 3 core items)
    spell: str          # battle spell name, e.g. "Retribution"
    emblem: str         # emblem name, e.g. "Assassin"
    emblem_stats: str   # e.g. "+14 Adaptive Penetration / +10 Adaptive Attack"
    win_rate: str       # e.g. "49.20%"
    pick_rate: str      # e.g. "4.42%"


class HeroBuildResult(BaseModel):
    hero: str
    hero_id: int
    lane: str
    lane_inferred: bool   # True if lane was inferred from hero profile, not user-supplied
    builds: list[BuildEntry]  # top 3 builds by pick rate
    citation: Citation


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


async def fetch_hero_build(
    client: "MLBBClient",
    roster: "HeroRoster",
    equipment: EquipmentLookup,
    hero: str | int,
    lane: str | None = None,
) -> HeroBuildResult | ToolError:
    """
    Return the top builds for a hero on a given lane.

    If lane is omitted, it is inferred from the hero's primary lane in their
    profile. The `lane_inferred` flag in the result indicates this so the LLM
    can mention it.

    Parameters
    ----------
    hero:
        Hero name (full or partial) or numeric ID.
    lane:
        One of: exp, mid, roam, jungle, gold. If None, inferred from profile.
    """
    from mlbb.heroes import AmbiguousHeroError, HeroNotFoundError

    # Resolve hero
    try:
        hero_ref = await roster.resolve(hero)
    except HeroNotFoundError as e:
        return ToolError(error="hero_not_found", message=str(e),
                         details={"identifier": str(hero)})
    except AmbiguousHeroError as e:
        return ToolError(error="ambiguous_hero", message=str(e),
                         details={"query": e.query, "candidates": e.candidates})

    # Validate or infer lane
    lane_inferred = False
    if lane is not None:
        lane = lane.lower().strip()
        if lane not in VALID_LANES:
            return ToolError(
                error="invalid_parameter",
                message=f"lane must be one of: {', '.join(VALID_LANES)} — got {lane!r}",
            )
    else:
        lane, error = await _infer_lane(client, hero_ref.id)
        if error:
            return error
        lane_inferred = True

    # Fetch builds
    try:
        raw, citation = await client.fetch(
            f"api/academy/heroes/{hero_ref.id}/builds",
            params={"lane": lane},
            fresh_ttl=FRESH_TTL_LONG,
        )
    except MLBBError as e:
        return ToolError(error="upstream_unavailable", message=str(e))

    records = raw["data"]["records"]
    if not records:
        return ToolError(
            error="no_data",
            message=f"No build data found for {hero_ref.name} on {lane} lane.",
        )

    # The builds are in records[0].data.build[]
    builds_raw = records[0]["data"]["build"]

    # Sort by pick rate (most popular first) and take top 3
    builds_raw = sorted(builds_raw, key=lambda b: b["build_pick_rate"], reverse=True)[:3]

    builds = []
    for b in builds_raw:
        items = await equipment.names_for_ids(b["equipid"])
        emblem_stats = _clean_emblem_stats(
            b["emblem"]["data"]["emblemattr"]["emblemattr"]
        )
        builds.append(BuildEntry(
            items=items,
            spell=b["battleskill"]["data"]["__data"]["skillname"],
            emblem=b["emblem"]["data"]["emblemname"],
            emblem_stats=emblem_stats,
            win_rate=_pct(b["build_win_rate"]),
            pick_rate=_pct(b["build_pick_rate"]),
        ))

    citation = citation.model_copy(update={"rank_tier": None, "time_window_days": None})
    return HeroBuildResult(
        hero=hero_ref.name,
        hero_id=hero_ref.id,
        lane=lane,
        lane_inferred=lane_inferred,
        builds=builds,
        citation=citation,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _infer_lane(
    client: "MLBBClient",
    hero_id: int,
) -> tuple[str | None, ToolError | None]:
    """
    Fetch hero profile and extract primary lane as an API lane param string.
    Returns (lane_str, None) on success, (None, ToolError) on failure.
    """
    try:
        raw, _ = await client.fetch(
            f"api/heroes/{hero_id}",
            fresh_ttl=FRESH_TTL_LONG,
        )
    except MLBBError as e:
        return None, ToolError(error="upstream_unavailable", message=str(e))

    records = raw["data"]["records"]
    if not records:
        return None, ToolError(
            error="no_data",
            message="Could not fetch hero profile to infer lane.",
        )

    lane_labels: list[str] = records[0]["data"]["hero"]["data"]["roadsortlabel"]
    # Take first non-empty label and map to API param
    for label in lane_labels:
        if label:
            mapped = _LANE_LABEL_MAP.get(label.lower())
            if mapped:
                log.debug("inferred lane %r -> %r for hero_id %d", label, mapped, hero_id)
                return mapped, None

    return None, ToolError(
        error="lane_unknown",
        message=(
            f"Could not infer lane for this hero (labels: {lane_labels}). "
            f"Please specify lane explicitly: {', '.join(VALID_LANES)}."
        ),
    )


def _clean_emblem_stats(raw: str) -> str:
    """Convert multiline emblem attr string to a compact slash-separated line."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return " / ".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"
