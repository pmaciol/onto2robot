"""Command line interface for onto2robot."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from onto2robot.core import MobileOntologyMeta
from onto2robot.fs_wrapper import SimpfulFuzzyWrapper
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onto2robot", description="Ontology to robot utilities")
    parser.add_argument("--input", type=str, help="Ontology to parse", required=True)
    parser.add_argument("--ontologies_path", type=str, help="Path to ontologies directory", required=True)
    parser.add_argument("--goal", type=str, help="Goal individual name", required=True)
    parser.add_argument(
        "--fuzzy_model", type=str, choices=["scikit-fuzzy", "simpful"], help="Fuzzy logic library to use", required=True
    )
    parser.add_argument(
        "--save_rules_fig", type=bool, help="Whether to save the fired rules as figures", required=False, default=False
    )
    parser.add_argument(
        "--save_rules_fig_path", type=Path, help="Path to save the fired rules as figures", required=False, default="."
    )
    parser.add_argument(
        "--save_consequents_fig",
        type=bool,
        help="Whether to save the consequents as figures",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--save_consequents_fig_path",
        type=Path,
        help="Path to save the consequents as figures",
        required=False,
        default=".",
    )
    parser.add_argument(
        "--input_values",
        type=str,
        help="Input values as JSON string, if not provided the predefined tests will be used",
        required=False,
        default="{}",
    )

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

    args_input_values = json.loads(args.input_values)
    if args_input_values:
        input_values = [dict[str, float](args_input_values)]
    else:
        input_values: list[dict[str, float]] = [
            {
                "R01sLF": 1.0,
                "R01sLS": 20.0,
                "R01sFL": 1.0,
                "R01sFR": 1.0,
                "R01sRF": 1.0,
                "R01sRS": 20.0,
                "R01sBL": 20.0,
                "R01sBR": 20.0,
            },
            {
                "R01sLF": 1.0,
                "R01sLS": 39.0,
                "R01sFL": 20.0,
                "R01sFR": 20.0,
                "R01sRF": 39.0,
                "R01sRS": 1.0,
                "R01sBL": 20.0,
                "R01sBR": 20.0,
            },
            {
                "R01sLF": 39.0,
                "R01sLS": 1.0,
                "R01sFL": 20.0,
                "R01sFR": 20.0,
                "R01sRF": 1.0,
                "R01sRS": 39.0,
                "R01sBL": 20.0,
                "R01sBR": 20.0,
            },
            {
                "R01sLF": 39.0,
                "R01sLS": 1.0,
                "R01sFL": 1.0,
                "R01sFR": 39.0,
                "R01sRF": 1.0,
                "R01sRS": 39.0,
                "R01sBL": 20.0,
                "R01sBR": 20.0,
            },
            {
                "R01sLF": 20.0,
                "R01sLS": 20.0,
                "R01sFL": 1.0,
                "R01sFR": 1.0,
                "R01sRF": 1.0,
                "R01sRS": 1.0,
                "R01sBL": 20.0,
                "R01sBR": 20.0,
            },
            {
                "R01sLF": 20.0,
                "R01sLS": 1.0,
                "R01sFL": 1.0,
                "R01sFR": 1.0,
                "R01sRF": 20.0,
                "R01sRS": 20.0,
                "R01sBL": 10.0,
                "R01sBR": 30.0,
            },
            {
                "R01sLF": 20.0,
                "R01sLS": 1.0,
                "R01sFL": 1.0,
                "R01sFR": 1.0,
                "R01sRF": 20.0,
                "R01sRS": 20.0,
                "R01sBL": 30.0,
                "R01sBR": 10.0,
            },
            {
                "R01sLF": 40.0,
                "R01sLS": 20.0,
                "R01sFL": 20.0,
                "R01sFR": 20.0,
                "R01sRF": 20.0,
                "R01sRS": 20.0,
                "R01sBL": 20.0,
                "R01sBR": 20.0,
            },
            {
                "R01sLF": 10.0,
                "R01sLS": 15.0,
                "R01sFL": 5.0,
                "R01sFR": 4.0,
                "R01sRF": 10.0,
                "R01sRS": 25.0,
                "R01sBL": 25.0,
                "R01sBR": 18.0,
            },
            {
                "R01sLF": 15.0,
                "R01sLS": 5.0,
                "R01sFL": 10.0,
                "R01sFR": 5.0,
                "R01sRF": 25.0,
                "R01sRS": 15.0,
                "R01sBL": 30.0,
                "R01sBR": 10.0,
            },
            {
                "R01sLF": 22.0,
                "R01sLS": 10.0,
                "R01sFL": 25.0,
                "R01sFR": 18.0,
                "R01sRF": 7.0,
                "R01sRS": 35.0,
                "R01sBL": 8.0,
                "R01sBR": 35.0,
            },
            {
                "R01sLF": 19.0,
                "R01sLS": 21.0,
                "R01sFL": 3.0,
                "R01sFR": 2.0,
                "R01sRF": 2.0,
                "R01sRS": 22.0,
                "R01sBL": 22.0,
                "R01sBR": 19.0,
            },
            {
                "R01sLF": 2.0,
                "R01sLS": 15.0,
                "R01sFL": 2.0,
                "R01sFR": 3.0,
                "R01sRF": 2.0,
                "R01sRS": 15.0,
                "R01sBL": 38.0,
                "R01sBR": 3.0,
            },
        ]
    summary_headers = None
    summary_results = []
    for iv in input_values:
        fs.set_start_values(iv)
        print(reasoning_order)
        outputs = deepcopy(iv)
        for layer in reversed(reasoning_order):
            result = fs.compute(layer)
            print(result)
            if result:
                outputs.update(result)

        if args.save_rules_fig:
            if isinstance(fs, ScikitFuzzyWrapper):
                fs.save_rules_as_figures(args.save_rules_fig_path)
            else:
                print("Saving rules as figures is only supported for ScikitFuzzyWrapper.")

        if args.save_consequents_fig:
            if isinstance(fs, ScikitFuzzyWrapper):
                fs.save_consequents_as_figures(args.save_consequents_fig_path)
            else:
                print("Saving consequents as figures is only supported for ScikitFuzzyWrapper.")

        if summary_headers is None:
            summary_headers = list(outputs.keys())

        summary_results.append(outputs)

    print(";".join(summary_headers or []))
    for result in summary_results:
        print(";".join([str(float(result.get(key, 0))) for key in summary_headers or []]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
