"""
Core data models for the mlbb package.

Three groups:
  - Enums: typed constants for API parameters (RankTier, TimeWindow)
  - Citation: provenance metadata attached to every stats-returning tool response
  - ApiResponse: typed envelope for upstream JSON
  - ToolError: structured error that tools return instead of raising exceptions
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums — typed API parameters
# ---------------------------------------------------------------------------


class RankTier(str, Enum):
    """Rank tiers accepted by the upstream API's `rank` query param."""
    ALL = "all"
    EPIC = "epic"
    LEGEND = "legend"
    MYTHIC = "mythic"
    HONOR = "honor"
    GLORY = "glory"


# Valid time windows as integers; converted to strings when building requests.
# Using Literal rather than an Enum so tool schemas expose plain integers.
TimeWindow = Literal[1, 3, 7, 15, 30]


# ---------------------------------------------------------------------------
# Citation — attached to every tool response that returns stats
# ---------------------------------------------------------------------------

DataFreshness = Literal["fresh", "stale"]


class Citation(BaseModel):
    """
    Provenance metadata for every stat a tool returns.

    The LLM is expected to surface this in its answer so users know where
    numbers came from, how current they are, and under what filter conditions
    (rank tier, time window) they were collected.
    """

    source: str = "mlbb.rone.dev"
    attribution: str = (
        "Data via ridwaanhall/api-mobilelegends; game data © Moonton"
    )
    retrieved_at: datetime.datetime = Field(
        description="UTC timestamp when this data was fetched or last confirmed fresh."
    )
    data_freshness: DataFreshness = Field(
        default="fresh",
        description=(
            "'fresh' = live from upstream; "
            "'stale' = served from cache because upstream was unreachable."
        ),
    )
    time_window_days: int | None = Field(
        default=None,
        description="Number of days the stats window covers, if applicable.",
    )
    rank_tier: str | None = Field(
        default=None,
        description="Rank tier filter applied to the query, if applicable.",
    )


# ---------------------------------------------------------------------------
# API envelope — typed wrapper for the upstream response structure
# ---------------------------------------------------------------------------

DataT = TypeVar("DataT")


class ApiResponse(BaseModel, Generic[DataT]):
    """
    Top-level envelope for all mlbb.rone.dev responses.

    Success: code=0, data is populated.
    Failure: code != 0 (or HTTP error status), data may be None.

    Note: upstream error responses use a different shape entirely
    (string `code`, extra `status` field), so we handle those at the
    HTTP layer before attempting to parse this model.
    """

    code: int
    message: str
    data: DataT | None = None


# ---------------------------------------------------------------------------
# Shared primitive
# ---------------------------------------------------------------------------


class HeroRef(BaseModel):
    """Minimal hero identity — used as a building block in other models."""

    id: int
    name: str


# ---------------------------------------------------------------------------
# ToolError — returned by tools instead of raising exceptions
# ---------------------------------------------------------------------------


class ToolError(BaseModel):
    """
    Structured error that tools return when they can't fulfill a request.

    Design note: tools return this as their result rather than raising, so
    the LLM can read the error and respond helpfully ("I couldn't find that
    hero") instead of receiving an opaque tool failure.

    The `error` field is a machine-readable code the LLM can pattern-match
    on; `message` is human-readable text it can quote directly.
    """

    error: str = Field(
        description=(
            "Machine-readable error code. "
            "Examples: 'hero_not_found', 'upstream_unavailable', 'invalid_parameter'."
        )
    )
    message: str = Field(description="Human-readable explanation of what went wrong.")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional extra context (e.g. which hero name failed to resolve).",
    )
