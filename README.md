# mlbb-mcp

An MCP server for Mobile Legends: Bang Bang hero data, backed by [ridwaanhall/api-mobilelegends](https://github.com/ridwaanhall/api-mobilelegends). Built as a learning project for personal use, covering MCP server design, LLM tool design, grounded generation with citations, resilient API caching, and LLM-as-judge evals.

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

Copy `.env.example` to `.env` and fill in your keys (only needed for evals, not for the MCP server or CLI):

```bash
cp .env.example .env
# ANTHROPIC_API_KEY — required for both eval scripts
# OPENAI_API_KEY    — required for evals/comprehensive_evals.py (GPT judge) only
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

LLM behavior evals — separate from unit tests. They make real API calls and cost money, so run them intentionally rather than in CI. Both require `ANTHROPIC_API_KEY` (and `OPENAI_API_KEY` for the comprehensive suite) in `.env`.

### `evals/run_evals.py` — focused behavioural checks

Two targeted evals that pre-inject tool results and grade the final response:

```bash
python evals/run_evals.py              # both
python evals/run_evals.py citation     # citation correctness only
python evals/run_evals.py fabrication  # fabrication refusal only
```

- **citation_correctness** — injects a real live tool result; checks Claude's response includes rank tier, time window, percentage, and source.
- **fabrication_refusal** — injects a `ToolError`; checks Claude refuses to invent stats rather than making something up.

### `evals/comprehensive_evals.py` — full tool-selection + quality suite

33 questions across all 8 tools. Claude actually calls tools with real parameters; GPT grades each trace on tool selection, citation quality, and fabrication.

```bash
python evals/comprehensive_evals.py              # all 33, Haiku answerer (~$0.10, ~2 min)
python evals/comprehensive_evals.py --sonnet     # all 33, Sonnet answerer (~$1–2, ~5 min)
python evals/comprehensive_evals.py 5            # first 5 only (smoke test)
python evals/comprehensive_evals.py 5 build      # first 5, filter by category
```

Use Haiku by default during development; switch to `--sonnet` when you want to measure Sonnet's behaviour specifically. The judge is always `gpt-4o-mini` — the grading task is mechanical enough that a frontier judge model isn't needed.

---

## Attribution

Game data © Moonton. API by [ridwaanhall](https://github.com/ridwaanhall/api-mobilelegends), hosted at [mlbb.rone.dev](https://mlbb.rone.dev).
