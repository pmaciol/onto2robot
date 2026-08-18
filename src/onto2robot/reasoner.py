from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from onto2robot.core import MobileOntologyMeta
from onto2robot.fs_wrapper import SimpfulFuzzyWrapper
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper

type FuzzyWrapperType = type[SimpfulFuzzyWrapper] | type[ScikitFuzzyWrapper]

SIDE_SENSOR_KEYS = ("R01sLS", "R01sLF", "R01sFL", "R01sFR", "R01sRF", "R01sRS")
BOTTOM_SENSOR_KEYS = ("R01sBL", "R01sBR")


def sensors_to_input_values(
    side_sensors_left_to_right: Sequence[float],
    bottom_sensors_left_to_right: Sequence[float],
) -> dict[str, float]:
    if len(side_sensors_left_to_right) != len(SIDE_SENSOR_KEYS):
        raise ValueError(f"Expected {len(SIDE_SENSOR_KEYS)} side sensor values, got {len(side_sensors_left_to_right)}.")
    if len(bottom_sensors_left_to_right) != len(BOTTOM_SENSOR_KEYS):
        raise ValueError(f"Expected {len(BOTTOM_SENSOR_KEYS)} bottom sensor values, got {len(bottom_sensors_left_to_right)}.")

    input_values = {
        sensor_name: sensor_value for sensor_name, sensor_value in zip(SIDE_SENSOR_KEYS, side_sensors_left_to_right, strict=True)
    }
    input_values.update(
        {
            sensor_name: sensor_value
            for sensor_name, sensor_value in zip(BOTTOM_SENSOR_KEYS, bottom_sensors_left_to_right, strict=True)
        }
    )
    return input_values


class Reasoner:
    def __init__(
        self,
        ontology_name: str,
        ontologies_path: Path | None = None,
        goal_name: str = "finalMove",
        FuzzyModel: FuzzyWrapperType = SimpfulFuzzyWrapper,
    ):
        ont = MobileOntologyMeta(ontology_name, ontologies_path)
        rules = ont.get_rules()
        goal = goal_name
        linguistic_variables_spaces = ont.get_linguistic_variable_domains()
        self.reasoning_order, _ = ont.get_possible_chains([ont.get_individual_by_name(goal)])
        self.fs = FuzzyModel(
            linguistic_variables_spaces,
            rules=rules,
            goal_name=goal,
        )

    def reason(self, input_values: dict[str, float]):
        self.fs.set_start_values(input_values)
        outputs = deepcopy(input_values)
        summary_headers = None
        summary_results = []
        for layer in reversed(self.reasoning_order):
            result = self.fs.compute(layer)
            if result:
                outputs.update(result)

            if summary_headers is None:
                summary_headers = list(outputs.keys())

            summary_results.append(outputs)

        return self.fs.get_goal_value()
