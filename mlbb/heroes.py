"""
Hero roster: name/ID resolution and reverse lookup.

Two main use cases:

1. Forward resolution — accept messy user input, return a clean HeroRef:
       roster.resolve("lancelot")  ->  HeroRef(id=47, name="Lancelot")
       roster.resolve("lance")     ->  HeroRef(id=47, name="Lancelot")  # prefix
       roster.resolve(47)          ->  HeroRef(id=47, name="Lancelot")  # by ID

2. Reverse lookup — enrich API responses that return hero IDs without names:
       roster.name_for_id(47)  ->  "Lancelot"

The roster is fetched from /api/heroes with a 24h cache TTL (hero list only
changes on patch day). Loading is lazy: the first call to resolve() or
name_for_id() triggers a fetch if the roster isn't populated yet.

Note: the upstream API also accepts hero names directly as path parameters
(e.g. /api/heroes/Lancelot), so forward resolution is mainly for input
validation. Reverse lookup is essential for enriching counter/synergy lists,
which return only hero IDs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mlbb.cache import FRESH_TTL_LONG
from mlbb.models import HeroRef

if TYPE_CHECKING:
    from mlbb.client import MLBBClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HeroNotFoundError(Exception):
    """No hero matched the given name or ID."""

    def __init__(self, identifier: str | int) -> None:
        self.identifier = identifier
        super().__init__(f"No hero found matching {identifier!r}")


class AmbiguousHeroError(Exception):
    """A name matched more than one hero; user needs to be more specific."""

    def __init__(self, query: str, candidates: list[str]) -> None:
        self.query = query
        self.candidates = candidates
        joined = ", ".join(candidates)
        super().__init__(
            f"{query!r} is ambiguous — did you mean one of: {joined}?"
        )


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------


class HeroRoster:
    """
    In-memory index of all MLBB heroes, populated from /api/heroes.

    Construct once per MLBBClient and reuse. All public methods are async
    because the first call may need to fetch from the network.
    """

    def __init__(self, client: "MLBBClient") -> None:
        self._client = client
        self._by_id: dict[int, HeroRef] = {}
        self._by_name_lower: dict[str, HeroRef] = {}  # lowercase name -> HeroRef
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(self, identifier: str | int) -> HeroRef:
        """
        Resolve a hero name or ID to a HeroRef.

        Accepts:
          - int: exact ID match (47 -> Lancelot)
          - str of digits: treated as ID ("47" -> Lancelot)
          - str name: case-insensitive exact match first, then substring match

        Raises HeroNotFoundError if nothing matches.
        Raises AmbiguousHeroError if a substring matches multiple heroes.
        """
        await self._ensure_loaded()

        # --- Numeric ID ---
        if isinstance(identifier, int):
            return self._lookup_by_id(identifier)

        stripped = identifier.strip()

        if stripped.isdigit():
            return self._lookup_by_id(int(stripped))

        # --- Name: exact match (case-insensitive) ---
        key = stripped.lower()
        if key in self._by_name_lower:
            return self._by_name_lower[key]

        # --- Name: substring match ---
        matches = [
            hero
            for name, hero in self._by_name_lower.items()
            if key in name
        ]
        if len(matches) == 1:
            log.debug("resolved %r via substring match -> %s", identifier, matches[0].name)
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousHeroError(stripped, [h.name for h in matches])

        raise HeroNotFoundError(stripped)

    async def name_for_id(self, hero_id: int) -> str:
        """
        Return the hero name for a given ID.

        Returns a placeholder string ("Hero#<id>") for unknown IDs rather
        than raising — counter and synergy lists may include recently-added
        heroes not yet in a cached roster.
        """
        await self._ensure_loaded()
        hero = self._by_id.get(hero_id)
        if hero:
            return hero.name
        log.warning("unknown hero_id %d — using placeholder", hero_id)
        return f"Hero#{hero_id}"

    async def all_heroes(self) -> list[HeroRef]:
        """Return all heroes sorted by ID."""
        await self._ensure_loaded()
        return sorted(self._by_id.values(), key=lambda h: h.id)

    async def total(self) -> int:
        await self._ensure_loaded()
        return len(self._by_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self._load()

    async def _load(self) -> None:
        log.debug("loading hero roster from API")
        # size=200 fetches all heroes in one request.
        # The roster has 132 heroes as of writing; bump size if it grows.
        data, _ = await self._client.fetch(
            "api/heroes",
            params={"size": 200, "order": "asc"},
            fresh_ttl=FRESH_TTL_LONG,
        )
        records = data["data"]["records"]
        for record in records:
            d = record["data"]
            hero = HeroRef(
                id=d["hero_id"],
                name=d["hero"]["data"]["name"],
            )
            self._by_id[hero.id] = hero
            self._by_name_lower[hero.name.lower()] = hero

        self._loaded = True
        log.debug("roster loaded: %d heroes", len(self._by_id))

    def _lookup_by_id(self, hero_id: int) -> HeroRef:
        if hero_id in self._by_id:
            return self._by_id[hero_id]
        raise HeroNotFoundError(hero_id)
