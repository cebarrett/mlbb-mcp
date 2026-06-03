"""
mlbb — library for querying Mobile Legends: Bang Bang hero data.

Typical usage
-------------
    from mlbb import MLBBClient, HeroRoster

    async with MLBBClient() as client:
        roster = HeroRoster(client)
        hero = await roster.resolve("Lancelot")
"""

from mlbb.cache import FRESH_TTL_LONG, FRESH_TTL_SHORT, Cache
from mlbb.client import BadRequestError, MLBBClient, MLBBError, UpstreamError
from mlbb.heroes import AmbiguousHeroError, HeroNotFoundError, HeroRoster
from mlbb.models import Citation, HeroRef, RankTier, TimeWindow, ToolError

__all__ = [
    # client
    "MLBBClient",
    "MLBBError",
    "UpstreamError",
    "BadRequestError",
    # heroes
    "HeroRoster",
    "HeroNotFoundError",
    "AmbiguousHeroError",
    # models
    "Citation",
    "HeroRef",
    "RankTier",
    "TimeWindow",
    "ToolError",
    # cache
    "Cache",
    "FRESH_TTL_SHORT",
    "FRESH_TTL_LONG",
]
