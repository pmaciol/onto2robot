"""Command line interface for onto2robot."""

import argparse
from pathlib import Path

from core import MobileOntologyMeta, load_ontology


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onto2robot", description="Ontology to robot utilities")
    parser.add_argument("-input", type=str, help="Ontology to parse", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f"Selected ontology: {args.input}")
    if Path(args.input).is_file():
        ontology = load_ontology(args.input)
        ont = MobileOntologyMeta(ontology)
        rules = ont.get_rules()
        for rule in rules:
            print(
                f"Rule {rule.label[0]}\n \
                hasPremise: \
                    {
                    [
                        [value.name for value in prop[rule]]
                        for prop in rule.get_properties()
                        if prop.name == 'hasPremise'
                    ]
                } hasConclusion: \
                    {
                    [
                        [value.name for value in prop[rule]]
                        for prop in rule.get_properties()
                        if prop.name == 'hasConclusion'
                    ]
                }"
            )
        return 0
    print(f"Failed to process with ontology from path {args.input}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
