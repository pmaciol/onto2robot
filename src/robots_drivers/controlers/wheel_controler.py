from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import mean
from typing import Protocol

from common.utils import clamp
from onto2robot.reasoner import Reasoner, sensors_to_input_values
from onto2robot.scikit_fuzz_wrapper import ScikitFuzzyWrapper


class ReasonerTypes(StrEnum):
    MOCK = "mock"
    AMRO = "amro"


@dataclass(frozen=True, slots=True)
class DifferentialWheelCommand:
    left: float
    right: float


class WheelController(Protocol):
    def __call__(
        self, side_sensors_left_to_right: Sequence[float], bottom_sensors_left_to_right: Sequence[float]
    ) -> DifferentialWheelCommand: ...


class DirectionToWheelCommand:
    def __init__(self, full_left: float, full_right: float, max_wheel: float):
        self.full_left = full_left
        self.full_right = full_right
        self.max_wheel = max_wheel
        self.forward = (full_left + full_right) / 2.0
        self.norm = 1.0 / (self.full_right - self.full_left)

    def __call__(self, direction: float) -> DifferentialWheelCommand:
        clamped_direction = clamp(direction, self.full_left, self.full_right)

        if clamped_direction <= self.forward:
            left_ratio = (clamped_direction - self.full_left) / (self.forward - self.full_left)
            left_wheel = (-1.0 + 2.0 * left_ratio) * self.max_wheel
            right_wheel = self.max_wheel
        else:
            right_ratio = (clamped_direction - self.forward) / (self.full_right - self.forward)
            left_wheel = self.max_wheel
            right_wheel = (1.0 - 2.0 * right_ratio) * self.max_wheel

        print(f"!!!!!!!!!!!!!!!! left {left_wheel}, right {right_wheel}")
        return DifferentialWheelCommand(left=left_wheel, right=right_wheel)


class Controller:
    def __init__(self, wheel_controller: DirectionToWheelCommand, reasoner: Reasoner, enable_logging: bool = True):
        self.wheel_controller = wheel_controller
        self.reasoner = reasoner
        self.enable_logging = enable_logging

    def __call__(
        self, side_sensors_left_to_right: Sequence[float], bottom_sensors_left_to_right: Sequence[float]
    ) -> DifferentialWheelCommand:
        input_values = sensors_to_input_values(side_sensors_left_to_right, bottom_sensors_left_to_right)
        if self.enable_logging:
            print("Controller input_values=%s", input_values)
        goal = self.reasoner.reason(input_values)
        if self.enable_logging:
            print("Controller goal=%s", goal)
        if goal is None:
            raise ValueError("Reasoner returned None for the goal value.")
        return self.wheel_controller(goal)


def controller_factory(resoner_type: ReasonerTypes) -> WheelController:
    if resoner_type == ReasonerTypes.MOCK:
        return mock_decide_wheel_control
    elif resoner_type == ReasonerTypes.AMRO:
        reasoner = Reasoner(
            # TODO all this stuff configurable from command line args
            ontology_name="amro_uc01_v01",
            ontologies_path=Path("ontologies/"),
            goal_name="finalMove",
            FuzzyModel=ScikitFuzzyWrapper,
        )
        ctrl = Controller(
            wheel_controller=DirectionToWheelCommand(full_left=0.0, full_right=40.0, max_wheel=5.0),
            reasoner=reasoner,
        )
        return ctrl
    else:
        raise ValueError(f"Unknown reasoner type: {resoner_type}")


def mock_decide_wheel_control(
    side_sensors_left_to_right: Sequence[float], bottom_sensors_left_to_right: Sequence[float]
) -> DifferentialWheelCommand:
    print("MOOOOOOOOOOOOOOOOOCK")
    """Return a deterministic mock wheel command from the side and line sensors.

    The heuristic assumes that larger readings mean more free space / stronger line signal.
    It biases the robot away from the more constrained side and nudges it toward the stronger bottom sensor.
    """

    # if len(side_sensors_left_to_right) != len(SIDE_SENSOR_KEYS):
    #     raise ValueError(f"Expected {len(SIDE_SENSOR_KEYS)} side sensor values.")
    # if len(bottom_sensors_left_to_right) != len(BOTTOM_SENSOR_KEYS):
    #     raise ValueError(f"Expected {len(BOTTOM_SENSOR_KEYS)} bottom sensor values.")

    left_side = mean(side_sensors_left_to_right[:3])
    right_side = mean(side_sensors_left_to_right[3:])
    bottom_left, bottom_right = bottom_sensors_left_to_right

    side_balance = (right_side - left_side) / 40.0
    line_balance = (bottom_right - bottom_left) / 20.0
    steering = clamp(0.65 * line_balance + 0.35 * side_balance, -1.0, 1.0)

    open_space = clamp(mean((*side_sensors_left_to_right, *bottom_sensors_left_to_right)) / 40.0, 0.0, 1.0)
    forward_speed = clamp(0.2 + 0.8 * open_space, 0.2, 1.0)
    turn_delta = steering * 0.5

    return DifferentialWheelCommand(
        left=clamp(forward_speed - turn_delta, 0.0, 1.0),
        right=clamp(forward_speed + turn_delta, 0.0, 1.0),
    )
