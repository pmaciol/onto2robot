"""Command line interface for onto2robot."""

import argparse
import json
from pathlib import Path

import numpy as np

from onto2robot.core import MobileOntologyMeta
from onto2robot.fs_wrapper import SimpfulFuzzyWrapper
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper

UNIVERSE_MIN = 0.0
UNIVERSE_MAX = 40.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onto2robot", description="Ontology to robot utilities")
    parser.add_argument("--input", type=str, help="Ontology to parse", required=True)
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
    if Path(args.input).is_file():
        ont = MobileOntologyMeta("mobile_robot_ontology")
        rules = ont.get_rules()
        goal = args.goal
        # TODO: replace with proper extraction from ontology
        linguistic_spaces = [
            ["low", "middle", "high"],
            ["left", "forward", "right"],
        ]
        linguistic_variables_spaces = ont.linguistic_value_spaces(linguistic_spaces)
        reasoning_order, source_variables = ont.get_possible_chains([ont.get_individual_by_name(goal)])

        if args.fuzzy_model == "scikit-fuzzy":
            universe = np.arange(UNIVERSE_MIN, UNIVERSE_MAX, 1)
            fs = ScikitFuzzyWrapper(
                linguistic_variables_spaces,
                args.goal,
                universe=universe,
                rules=rules,
            )
        elif args.fuzzy_model == "simpful":
            universe = (UNIVERSE_MIN, UNIVERSE_MAX)
            fs = SimpfulFuzzyWrapper(
                linguistic_variables_spaces,
                universe=universe,
                rules=rules,
            )
        input_values = json.loads(args.input_values)

        fs.set_start_values(input_values)
        print(reasoning_order)
        for layer in reversed(reasoning_order):
            fs.compute(layer)
        return 0
    print(f"Failed to process with ontology from path {args.input}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
