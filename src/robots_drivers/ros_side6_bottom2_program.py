from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

from onto2robot.core import MobileOntologyMeta
from onto2robot.fs_wrapper import SimpfulFuzzyWrapper
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper

SIDE_SENSOR_KEYS = ("R01sLS", "R01sLF", "R01sFL", "R01sFR", "R01sRF", "R01sRS")
BOTTOM_SENSOR_KEYS = ("R01sBL", "R01sBR")


class FuzzyModel(StrEnum):
    SCIKIT_FUZZY = "scikit-fuzzy"
    SIMPFUL = "simpful"


class RosSideBottom2Program:
    """ROS-compatible controller producing Twist from side/bottom distance sensors."""

    def __init__(
        self,
        ontology_name: str,
        ontologies_path: Path | None = None,
        goal_name: str = "finalMove",
        fuzzy_model: FuzzyModel = FuzzyModel.SCIKIT_FUZZY,
    ):
        self.goal_name = goal_name
        self.ontology = MobileOntologyMeta(ontology_name, ontologies_path)
        rules = self.ontology.get_rules()
        linguistic_variables_domains = self.ontology.get_linguistic_variable_domains()
        # reasoning_order, _ = self.ontology.get_possible_chains([self.ontology.get_individual_by_name(goal_name)])

        match fuzzy_model:
            case FuzzyModel.SCIKIT_FUZZY:
                self.reasoner = ScikitFuzzyWrapper(
                    linguistic_variables_domains,
                    rules=rules,
                    goal_name=goal_name,
                )
            case FuzzyModel.SIMPFUL:
                self.reasoner = SimpfulFuzzyWrapper(
                    linguistic_variables_domains,
                    rules=rules,
                    goal_name=goal_name,
                )
            case _:
                raise ValueError(f"fuzzy_model must be one of: {FuzzyModel.SCIKIT_FUZZY.value!r}, {FuzzyModel.SIMPFUL.value!r}.")

    def _make_input_values(
        self, side_sensors_left_to_right: list[float], bottom_sensors_left_to_right: list[float]
    ) -> dict[str, float]:
        return {
            **{name: value for name, value in zip(SIDE_SENSOR_KEYS, side_sensors_left_to_right, strict=True)},
            **{name: value for name, value in zip(BOTTOM_SENSOR_KEYS, bottom_sensors_left_to_right, strict=True)},
        }

    def _reason_direction(self, side_sensors_left_to_right: list[float], bottom_sensors_left_to_right: list[float]) -> float:
        input_values = self._make_input_values(side_sensors_left_to_right, bottom_sensors_left_to_right)
        self.reasoner.set_start_values(input_values)

        # TODO: encapsulate layers!

        outputs: dict[str, float] = dict(input_values)
        for layer in reversed(self.reasoning_order):
            result = self.reasoner.compute(layer)
            if result:
                outputs.update(result)

        return float(self.reasoner.get_goal_value(self.goal_name) or 0.0)

    def evaluate(self, side_sensors_left_to_right: list[float], bottom_sensors_left_to_right: list[float]) -> float:
        return self._reason_direction(side_sensors_left_to_right, bottom_sensors_left_to_right)

    def evaluate_named(self, sensors: Mapping[str, float]) -> float:
        side_sensors = [float(sensors[name]) for name in SIDE_SENSOR_KEYS]
        bottom_sensors = [float(sensors[name]) for name in BOTTOM_SENSOR_KEYS]
        return self.evaluate(side_sensors_left_to_right=side_sensors, bottom_sensors_left_to_right=bottom_sensors)

    def close(self) -> None:
        self.ontology.destroy()

    def __enter__(self) -> "RosSideBottom2Program":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
