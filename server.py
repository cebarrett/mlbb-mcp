"""
MLBB MCP server.

Exposes tools for querying Mobile Legends: Bang Bang hero data.
Runs over stdio transport — Claude Desktop spawns this as a subprocess.

To test manually with the MCP inspector:
    .venv/bin/mcp dev server.py

To install in Claude Desktop, add to ~/Library/Application Support/Claude/claude_desktop_config.json:
    {
      "mcpServers": {
        "mlbb": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["/absolute/path/to/server.py"]
        }
      }
    }

Why FastMCP vs the low-level Server class
------------------------------------------
FastMCP is the high-level SDK API. The @app.tool() decorator:
  - Reads the function's type annotations to build the JSON schema the LLM sees
  - Reads the docstring to build the tool description
  - Handles serialization of return values
  - Registers the tool with the MCP protocol

The low-level Server class requires implementing list_tools / call_tool handlers
manually. FastMCP is the right choice for new servers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mlbb import HeroRoster, MLBBClient, RankTier, ToolError
from mlbb.endpoints.hero import (
    fetch_hero_counters,
    fetch_hero_synergies,
    fetch_hero_trends,
)
from mlbb.endpoints.rank import fetch_hero_rank_stats, fetch_top_heroes

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------

app = FastMCP(
    name="mlbb-mcp",
    instructions=(
        "This server provides Mobile Legends: Bang Bang hero stats backed by "
        "live ranked game data. Every response includes a citation block "
        "(source, retrieved_at, time_window, rank_tier). "
        "Always include citation details in your answers. "
        "Never fabricate stats — if a tool returns an error, say so."
    ),
)

# Client and roster are initialized in lifespan and reused across all tool
# calls. Stored at module level so tool functions can reach them.
_client: MLBBClient | None = None
_roster: HeroRoster | None = None


def _deps() -> tuple[MLBBClient, HeroRoster]:
    assert _client is not None and _roster is not None, "Server not initialized"
    return _client, _roster


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

_VALID_DAYS = (1, 3, 7, 15, 30)
_VALID_RANKS = ("all", "epic", "legend", "mythic", "honor", "glory")


def _validate_days_rank(days: int, rank: str) -> str | None:
    """Return a JSON ToolError string if params are invalid, else None."""
    if days not in _VALID_DAYS:
        return ToolError(
            error="invalid_parameter",
            message=f"days must be one of {list(_VALID_DAYS)} — got {days}",
        ).model_dump_json(indent=2)
    if rank not in _VALID_RANKS:
        return ToolError(
            error="invalid_parameter",
            message=f"rank must be one of: {', '.join(_VALID_RANKS)} — got {rank!r}",
        ).model_dump_json(indent=2)
    return None


# ---------------------------------------------------------------------------
# Tool: get_hero_winrate
# ---------------------------------------------------------------------------

@app.tool()
async def get_hero_winrate(
    hero: str,
    days: int = 7,
    rank: str = "mythic",
) -> str:
    """
    Get the win rate, pick rate, and ban rate for a hero at a given rank tier.

    Every response includes a citation block with the data source, retrieval
    time, time window, and rank tier so claims can be verified. If the hero
    cannot be found or upstream data is unavailable, a structured error is
    returned instead of fabricated stats.

    Parameters
    ----------
    hero:
        Hero name (full or partial, case-insensitive) or numeric ID.
        Examples: "Lancelot", "lance", "47"
    days:
        Time window in days. Must be one of: 1, 3, 7, 15, 30.
        Defaults to 7 (last week).
    rank:
        Rank tier to filter by. One of: all, epic, legend, mythic, honor, glory.
        Defaults to "mythic" — the tier with the most competitive play.
    """
    err = _validate_days_rank(days, rank)
    if err:
        return err
    client, roster = _deps()
    result = await fetch_hero_rank_stats(
        client, roster,
        hero=hero,
        days=days,       # type: ignore[arg-type]
        rank=RankTier(rank),
    )
    return result.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Tool: get_top_heroes
# ---------------------------------------------------------------------------

@app.tool()
async def get_top_heroes(
    rank: str = "mythic",
    days: int = 7,
    sort_by: str = "win_rate",
    limit: int = 10,
) -> str:
    """
    Get the top heroes ranked by win rate, pick rate, or ban rate.

    Useful for questions like "who's strong right now?", "what's the current
    meta?", or "which heroes are being banned the most?". Returns a ranked
    list with citation.

    Parameters
    ----------
    rank:
        Rank tier. One of: all, epic, legend, mythic, honor, glory.
    days:
        Time window in days. One of: 1, 3, 7, 15, 30.
    sort_by:
        Stat to rank by. One of: win_rate, pick_rate, ban_rate.
    limit:
        Number of heroes to return. Between 1 and 30. Defaults to 10.
    """
    err = _validate_days_rank(days, rank)
    if err:
        return err
    if sort_by not in ("win_rate", "pick_rate", "ban_rate"):
        return ToolError(
            error="invalid_parameter",
            message=f"sort_by must be one of: win_rate, pick_rate, ban_rate — got {sort_by!r}",
        ).model_dump_json(indent=2)

    client, _ = _deps()
    result = await fetch_top_heroes(
        client,
        days=days,           # type: ignore[arg-type]
        rank=RankTier(rank),
        sort_by=sort_by,
        limit=limit,
    )
    return result.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Tool: get_hero_counters
# ---------------------------------------------------------------------------

@app.tool()
async def get_hero_counters(
    hero: str,
    days: int = 7,
    rank: str = "mythic",
) -> str:
    """
    Get the heroes that counter a given hero — i.e. reduce its win rate most
    when on the enemy team.

    Returns up to 5 heroes with their win rate delta against the queried hero
    and their own overall win rate. Useful for draft decisions.

    Parameters
    ----------
    hero:
        Hero name (full or partial) or numeric ID.
    days:
        Time window in days. One of: 1, 3, 7, 15, 30.
    rank:
        Rank tier. One of: all, epic, legend, mythic, honor, glory.
    """
    err = _validate_days_rank(days, rank)
    if err:
        return err
    client, roster = _deps()
    result = await fetch_hero_counters(
        client, roster, hero=hero, days=days, rank=RankTier(rank),  # type: ignore[arg-type]
    )
    return result.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Tool: get_hero_synergies
# ---------------------------------------------------------------------------

@app.tool()
async def get_hero_synergies(
    hero: str,
    days: int = 7,
    rank: str = "mythic",
) -> str:
    """
    Get the heroes that synergize best with a given hero — i.e. boost its win
    rate most when on the same team.

    Returns up to 5 heroes with their win rate delta alongside the queried
    hero. Useful for "who should I pair with X?" questions.

    Parameters
    ----------
    hero:
        Hero name (full or partial) or numeric ID.
    days:
        Time window in days. One of: 1, 3, 7, 15, 30.
    rank:
        Rank tier. One of: all, epic, legend, mythic, honor, glory.
    """
    err = _validate_days_rank(days, rank)
    if err:
        return err
    client, roster = _deps()
    result = await fetch_hero_synergies(
        client, roster, hero=hero, days=days, rank=RankTier(rank),  # type: ignore[arg-type]
    )
    return result.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Tool: get_hero_trends
# ---------------------------------------------------------------------------

@app.tool()
async def get_hero_trends(
    hero: str,
    days: int = 7,
    rank: str = "mythic",
) -> str:
    """
    Get the day-by-day win rate, pick rate, and ban rate trend for a hero.

    Useful for "is this hero trending up or down?" or "has Fanny's win rate
    changed this week?" questions. Returns one entry per day, newest first.

    Parameters
    ----------
    hero:
        Hero name (full or partial) or numeric ID.
    days:
        Time window in days. One of: 1, 3, 7, 15, 30.
    rank:
        Rank tier. One of: all, epic, legend, mythic, honor, glory.
    """
    err = _validate_days_rank(days, rank)
    if err:
        return err
    client, roster = _deps()
    result = await fetch_hero_trends(
        client, roster, hero=hero, days=days, rank=RankTier(rank),  # type: ignore[arg-type]
    )
    return result.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _init() -> None:
    """Initialize shared client and roster before serving requests."""
    global _client, _roster
    # Anchor the cache directory to this file's location, not the working
    # directory. Claude Desktop spawns the server from an unrelated cwd
    # (often / or the home dir), so a relative ".cache" path won't resolve.
    cache_dir = str(Path(__file__).parent / ".cache")
    _client = MLBBClient(cache_dir=cache_dir)
    _roster = HeroRoster(_client)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_init())
    app.run(transport="stdio")
