# mlbb-mcp

An MCP server for Mobile Legends: Bang Bang hero data, backed by [ridwaanhall/api-mobilelegends](https://github.com/ridwaanhall/api-mobilelegends). Built as a learning project covering MCP server design, LLM tool design, grounded generation with citations, and resilient API caching.

100% written by Claude; 100% reviewed by me.

---

## Tools

| Tool | What it answers |
|---|---|
| `list_heroes` | Browse/search the hero roster; resolve ambiguous names |
| `get_hero_winrate` | Win/pick/ban rate for a hero at a rank tier and time window |
| `get_top_heroes` | Top N heroes by win rate, pick rate, or ban rate |
| `get_hero_counters` | Heroes that reduce a hero's win rate (counters) |
| `get_hero_synergies` | Heroes that increase a hero's win rate (teammates) |
| `get_hero_trends` | Day-by-day win/pick/ban rate over N days |
| `get_hero_build` | Recommended items, spell, and emblem by lane |
| `get_hero_profile` | Role, lane, specialties, difficulty, skills, lore |

Every stats-returning tool includes a `citation` block: `source`, `retrieved_at`, `data_freshness`, `time_window_days`, `rank_tier`. `data_freshness` is `"fresh"` or `"stale"` — the server serves cached data when upstream is unavailable rather than failing.

---

## Setup

```bash
git clone <repo>
cd mlbb-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your key (only needed for evals):

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY=...
```

---

## Run

**CLI** (no MCP server needed, useful for testing):

```bash
python cli.py heroes                  # list all heroes
python cli.py resolve lancelot        # resolve a name or ID
python cli.py resolve 47              # by numeric ID
```

**MCP server** (for Claude Desktop):

```bash
python server.py
```

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mlbb": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

Then restart Claude Desktop.

---

## Tests

Unit and integration tests (no network, no API key):

```bash
pytest tests/ -v
```

The stale-cache tests cover every failure mode — network error, timeout, HTTP 5xx, non-JSON 200 (CDN error page), error envelopes, 4xx behaviour, cache poison guard, and recovery.

---

## Evals

LLM behaviour evals (requires `ANTHROPIC_API_KEY` in `.env`, makes real API calls):

```bash
python evals/run_evals.py              # both evals
python evals/run_evals.py citation     # citation correctness only
python evals/run_evals.py fabrication  # fabrication refusal only
```

Two evals:
- **citation_correctness** — does Claude include rank tier, time window, percentage, and source when answering a stats question?
- **fabrication_refusal** — does Claude refuse to invent stats when the tool returns an error?

---

## Project layout

```
mlbb/
  client.py        # async httpx client with cache + stale-serve
  cache.py         # disk TTL cache (diskcache), freshness tracking
  heroes.py        # hero roster, name/ID resolution
  models.py        # Citation, ToolError, RankTier, HeroRef
  endpoints/
    rank.py        # /api/heroes/rank → get_hero_winrate, get_top_heroes
    hero.py        # /api/heroes/{id}/* → counters, synergies, trends, profile
    academy.py     # /api/academy/heroes/{id}/builds → get_hero_build
server.py          # FastMCP server, tool definitions
cli.py             # dev CLI
tests/             # pytest stale-cache integration tests
evals/             # LLM behaviour evals (needs Anthropic API key)
```

---

## Attribution

Game data © Moonton. API by [ridwaanhall](https://github.com/ridwaanhall/api-mobilelegends), hosted at [mlbb.rone.dev](https://mlbb.rone.dev).
