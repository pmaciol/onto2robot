"""Command line interface for onto2robot."""

import argparse
import json
from pathlib import Path

from onto2robot.core import MobileOntologyMeta
from onto2robot.fs_wrapper import SimpfulFuzzyWrapper
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper

UNIVERSE_MIN = 0.0
UNIVERSE_MAX = 40.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onto2robot", description="Ontology to robot utilities")
    parser.add_argument("--input", type=str, help="Ontology to parse", required=True)
    parser.add_argument("--ontologies_path", type=str, help="Path to ontologies directory", required=True)
    parser.add_argument("--goal", type=str, help="Goal individual name", required=True)
    parser.add_argument(
        "--fuzzy_model", type=str, choices=["scikit-fuzzy", "simpful"], help="Fuzzy logic library to use", required=True
    )
    parser.add_argument("--input_values", type=str, help="Input values as JSON string", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f"Selected ontology: {args.input}")
    ont = MobileOntologyMeta(args.input, Path(args.ontologies_path))
    rules = ont.get_rules()
    goal = args.goal
    linguistic_variables_spaces = ont.get_linguistic_variable_domains()
    reasoning_order, _ = ont.get_possible_chains([ont.get_individual_by_name(goal)])
    match args.fuzzy_model:
        case "scikit-fuzzy":
            print("Using Scikit-Fuzzy as the fuzzy logic library.")
            fs = ScikitFuzzyWrapper(
                linguistic_variables_spaces,
                rules=rules,
                goal_name=args.goal,
            )
        case "simpful":
            print("Using Simpful as the fuzzy logic library.")
            fs = SimpfulFuzzyWrapper(
                linguistic_variables_spaces,
                rules=rules,
            )
        case _:
            print(f"Unsupported fuzzy model: {args.fuzzy_model}")
            return 1

    input_values = json.loads(args.input_values)

    fs.set_start_values(input_values)
    print(reasoning_order)
    for layer in reversed(reasoning_order):
        fs.compute(layer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
