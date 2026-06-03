from dataclasses import dataclass, field
from typing import Protocol


class ReasonerProtocol(Protocol):
    def __call__(self, side_sensors_left_to_right: list[float], bottom_sensors_left_to_right: list[float]) -> float: ...


@dataclass
class LinearVelocity:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class AngularVelocity:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Twist:
    linear: LinearVelocity = field(default_factory=LinearVelocity)
    angular: AngularVelocity = field(default_factory=AngularVelocity)


class SideBottom2Twist:
    def __init__(
        self,
        reasoner: ReasonerProtocol,
        side_sensors_count: int = 6,
        bottom_sensors_count: int = 2,
        direction_scale: tuple[float, float] = (0.0, 1.0),
        max_forward_speed: float = 1.0,
        min_forward_speed: float = 0.1,
        max_angular_speed: float = 1.0,
    ):
        self.min_forward_speed = min_forward_speed
        self.max_forward_speed = max_forward_speed
        self.max_angular_speed = max_angular_speed
        self.reasoner = reasoner
        self.side_sensors_count = side_sensors_count
        self.bottom_sensors_count = bottom_sensors_count
        self.direction_scale = direction_scale
        self.front_direction = direction_scale[0] + (direction_scale[1] - direction_scale[0]) / 2
        self.left_direction = direction_scale[0]
        self.right_direction = direction_scale[1]

    def _get_speed(self, direction: float) -> float:
        direction_deviation = 1 - abs(direction - self.front_direction) / (
            (self.right_direction - self.left_direction) / 2
        )
        return max(direction_deviation * self.max_forward_speed, self.min_forward_speed)

    def _get_angular_velocity(self, direction: float) -> float:
        direction_deviation = (direction - self.front_direction) / ((self.right_direction - self.left_direction) / 2)
        return direction_deviation * self.max_angular_speed

    def update_and_evaluate(
        self, side_sensors_left_to_right: list[float], bottom_sensors_left_to_right: list[float]
    ) -> Twist:
        if (
            len(side_sensors_left_to_right) != self.side_sensors_count
            or len(bottom_sensors_left_to_right) != self.bottom_sensors_count
        ):
            print(
                f"Invalid sensor data length. Expected {self.side_sensors_count} "
                f"side sensors and {self.bottom_sensors_count} bottom sensors."
            )
            return Twist()
        direction = self.reasoner(side_sensors_left_to_right, bottom_sensors_left_to_right)
        speed = self._get_speed(direction)
        angular_velocity = self._get_angular_velocity(direction)
        return Twist(linear=LinearVelocity(x=speed), angular=AngularVelocity(z=angular_velocity))
