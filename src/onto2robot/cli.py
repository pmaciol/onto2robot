"""Command line interface for onto2robot."""

from __future__ import annotations

import argparse

from .core import add


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onto2robot", description="Ontology to robot utilities")
    sub = parser.add_subparsers(dest="command")

    sum_p = sub.add_parser("sum", help="Add two integers")
    sum_p.add_argument("a", type=int)
    sum_p.add_argument("b", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sum":
        result = add(args.a, args.b)
        print(result)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
