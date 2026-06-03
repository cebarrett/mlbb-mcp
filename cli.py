"""
Quick dev/test CLI for the mlbb library.

Usage
-----
    python cli.py heroes              # list all heroes (id, name)
    python cli.py resolve Lancelot    # resolve a name or ID to a HeroRef
    python cli.py resolve lance       # substring match
    python cli.py resolve 47          # by numeric ID

Run with --help for a usage summary.
"""

import argparse
import asyncio
import sys

from mlbb import AmbiguousHeroError, HeroNotFoundError, HeroRoster, MLBBClient


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_heroes(args: argparse.Namespace) -> None:
    """List all heroes sorted by ID."""
    async with MLBBClient() as client:
        roster = HeroRoster(client)
        heroes = await roster.all_heroes()

    print(f"{'ID':>4}  {'Name'}")
    print("-" * 30)
    for h in heroes:
        print(f"{h.id:>4}  {h.name}")
    print(f"\n{len(heroes)} heroes total.")


async def cmd_resolve(args: argparse.Namespace) -> None:
    """Resolve a hero name or ID and print the result."""
    identifier: str = args.identifier
    async with MLBBClient() as client:
        roster = HeroRoster(client)
        try:
            hero = await roster.resolve(identifier)
            print(f"Resolved: id={hero.id}  name={hero.name!r}")
        except HeroNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except AmbiguousHeroError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Dev CLI for the mlbb MCP library.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    sub.add_parser("heroes", help="List all heroes.")

    p_resolve = sub.add_parser("resolve", help="Resolve a hero name or ID.")
    p_resolve.add_argument(
        "identifier",
        help="Hero name (full or partial) or numeric ID.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "heroes":
            asyncio.run(cmd_heroes(args))
        case "resolve":
            asyncio.run(cmd_resolve(args))
        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
