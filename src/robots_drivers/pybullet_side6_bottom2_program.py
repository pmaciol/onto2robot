from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, radians, sin
from statistics import mean
from typing import Protocol

import pybullet

from common.utils import clamp

SIDE_SENSOR_KEYS = ("R01sLS", "R01sLF", "R01sFL", "R01sFR", "R01sRF", "R01sRS")
BOTTOM_SENSOR_KEYS = ("R01sBL", "R01sBR")
SENSOR_VALUE_MAX = 40.0

SIDE_SENSOR_ANGLES_DEG = (90.0, 54.0, 18.0, -18.0, -54.0, -90.0)
BOTTOM_SENSOR_ANGLES_DEG = (20.0, -20.0)


@dataclass(frozen=True, slots=True)
class MissionStage:
    time_seconds: float
    direction: str
    speed: float


@dataclass(frozen=True, slots=True)
class RobotDimensions:
    body_radius: float = 0.05
    bottom_sensor_radius: float = 0.04
    body_height: float = 0.03
    wheel_radius: float = 0.02
    wheel_width: float = 0.01
    sensor_range: float = 0.4
    sensor_clearance: float = 0.001


@dataclass(frozen=True, slots=True)
class DifferentialWheelCommand:
    left: float
    right: float


class WheelController(Protocol):
    def __call__(
        self, side_sensors_left_to_right: Sequence[float], bottom_sensors_left_to_right: Sequence[float]
    ) -> DifferentialWheelCommand: ...


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float
    y: float
    z: float

    def __init__(self, bases: Sequence[float]) -> None:
        object.__setattr__(self, "x", bases[0])
        object.__setattr__(self, "y", bases[1])
        object.__setattr__(self, "z", bases[2])

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


@dataclass(frozen=True, slots=True)
class SensorRay:
    sensor_name: str
    start: Vec3
    end: Vec3


def _local_xy(angle_degrees: float, radius: float) -> tuple[float, float]:
    angle_radians = radians(angle_degrees)
    return radius * cos(angle_radians), radius * sin(angle_radians)


def _make_side_sensor_ray(
    sensor_name: str,
    angle_degrees: float,
    radial_distance: float,
    body_half_height: float,
    ray_length: float,
) -> SensorRay:
    x_pos, y_pos = _local_xy(angle_degrees, radial_distance)
    direction_x, direction_y = _local_xy(angle_degrees, 1.0)
    start = Vec3([x_pos, y_pos, body_half_height])
    end = Vec3(
        [
            x_pos + direction_x * ray_length,
            y_pos + direction_y * ray_length,
            body_half_height,
        ]
    )
    return SensorRay(sensor_name, start, end)


def _make_bottom_sensor_ray(
    sensor_name: str,
    angle_degrees: float,
    radial_distance: float,
    body_half_height: float,
    sensor_clearance: float,
    ray_length: float,
) -> SensorRay:
    x_pos, y_pos = _local_xy(angle_degrees, radial_distance)
    start = Vec3([x_pos, y_pos, -body_half_height + sensor_clearance])
    end = Vec3([x_pos, y_pos, -body_half_height - ray_length])
    return SensorRay(sensor_name, start, end)


def mock_decide_wheel_control(
    side_sensors_left_to_right: Sequence[float], bottom_sensors_left_to_right: Sequence[float]
) -> DifferentialWheelCommand:
    """Return a deterministic mock wheel command from the side and line sensors.

    The heuristic assumes that larger readings mean more free space / stronger line signal.
    It biases the robot away from the more constrained side and nudges it toward the stronger bottom sensor.
    """

    if len(side_sensors_left_to_right) != len(SIDE_SENSOR_KEYS):
        raise ValueError(f"Expected {len(SIDE_SENSOR_KEYS)} side sensor values.")
    if len(bottom_sensors_left_to_right) != len(BOTTOM_SENSOR_KEYS):
        raise ValueError(f"Expected {len(BOTTOM_SENSOR_KEYS)} bottom sensor values.")

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


class PyBulletSide6Bottom2Program:
    """PyBullet model of a cylindrical robot with six side sensors and two bottom sensors."""

    def run_control_loop(self, steps: int) -> list[tuple[dict[str, float], DifferentialWheelCommand]]:
        history: list[tuple[dict[str, float], DifferentialWheelCommand]] = []
        for _ in range(steps):
            history.append(self._control_step())
        return history

    def reset(self) -> None:
        body_half_height = self.dimensions.body_height / 2
        base_position = (0.0, 0.0, self.dimensions.wheel_radius + body_half_height)
        base_orientation = pybullet.getQuaternionFromEuler((0.0, 0.0, 0.0))
        pybullet.resetBasePositionAndOrientation(
            self.body_id,
            base_position,
            base_orientation,
            physicsClientId=self._physics_client_id,
        )
        pybullet.resetJointState(self.body_id, 0, 0.0, physicsClientId=self._physics_client_id)
        pybullet.resetJointState(self.body_id, 1, 0.0, physicsClientId=self._physics_client_id)

    def get_position(self) -> Vec3:
        """Return the robot's current position in the world."""
        base_position, _ = pybullet.getBasePositionAndOrientation(self.body_id, self._physics_client_id)
        return Vec3(base_position)

    def get_rotation(self) -> tuple[float, float, float, float]:
        """Return the robot's current rotation as a quaternion (x, y, z, w)."""
        _, base_orientation = pybullet.getBasePositionAndOrientation(self.body_id, self._physics_client_id)
        return base_orientation

    def get_initial_position(self) -> Vec3 | None:
        """Return the robot's initial position set in the constructor."""
        return self._initial_base_position

    def get_initial_rotation_degrees(self) -> float:
        """Return the robot's initial yaw rotation in degrees set in the constructor."""
        return self._initial_base_yaw_degrees

    def __init__(
        self,
        dimensions: RobotDimensions | None = None,
        sensor_max_distance: float | None = None,
        wheel_controller: WheelController = mock_decide_wheel_control,
        max_wheel_velocity: float = 8.0,
        max_motor_force: float = 2.5,
        client_id: int | None = None,
        base_position: Vec3 | None = None,
        base_yaw_degrees: float = 0.0,
        base_mass: float = 1.0,
        wheel_mass: float = 0.05,
    ):
        self.dimensions = dimensions or RobotDimensions()
        self.sensor_max_distance = sensor_max_distance or self.dimensions.sensor_range
        self.wheel_controller = wheel_controller
        self.max_wheel_velocity = max_wheel_velocity
        self.max_motor_force = max_motor_force
        self._physics_client_id = -1 if client_id is None else client_id
        self._initial_base_position = base_position
        self._initial_base_yaw_degrees = base_yaw_degrees
        self._initial_base_mass = base_mass
        self._initial_wheel_mass = wheel_mass
        try:
            self._build_robot(
                base_position=base_position,
                base_yaw_degrees=base_yaw_degrees,
                base_mass=base_mass,
                wheel_mass=wheel_mass,
            )
        except pybullet.error as error:
            if "Not connected to physics server" not in str(error):
                raise

    @property
    def client_id(self) -> int:
        return self._physics_client_id

    @property
    def side_sensor_keys(self) -> tuple[str, ...]:
        return SIDE_SENSOR_KEYS

    @property
    def bottom_sensor_keys(self) -> tuple[str, ...]:
        return BOTTOM_SENSOR_KEYS

    @property
    def side_sensor_angles_deg(self) -> tuple[float, ...]:
        return SIDE_SENSOR_ANGLES_DEG

    @property
    def bottom_sensor_angles_deg(self) -> tuple[float, ...]:
        return BOTTOM_SENSOR_ANGLES_DEG

    def _iter_sensor_positions(self, sensor_keys: Sequence[str], sensor_angles_deg: Sequence[float]) -> list[tuple[str, float]]:
        return [(sensor_name, angle_degrees) for sensor_name, angle_degrees in zip(sensor_keys, sensor_angles_deg, strict=True)]

    def _wheel_command_from_sensors(
        self, side_sensors_left_to_right: Sequence[float], bottom_sensors_left_to_right: Sequence[float]
    ) -> DifferentialWheelCommand:
        if len(side_sensors_left_to_right) != len(self.side_sensor_keys):
            raise ValueError(f"Expected {len(self.side_sensor_keys)} side sensor values, got {len(side_sensors_left_to_right)}.")
        if len(bottom_sensors_left_to_right) != len(self.bottom_sensor_keys):
            raise ValueError(
                f"Expected {len(self.bottom_sensor_keys)} bottom sensor values, got {len(bottom_sensors_left_to_right)}."
            )
        return self.wheel_controller(side_sensors_left_to_right, bottom_sensors_left_to_right)

    def _iter_sensor_rays(self) -> list[SensorRay]:
        body_half_height = self.dimensions.body_height / 2
        ray_length = self.sensor_max_distance + self.dimensions.sensor_clearance

        return [
            _make_side_sensor_ray(
                sensor_name,
                angle_degrees,
                self.dimensions.body_radius,
                body_half_height,
                ray_length,
            )
            for sensor_name, angle_degrees in self._iter_sensor_positions(self.side_sensor_keys, self.side_sensor_angles_deg)
        ] + [
            _make_bottom_sensor_ray(
                sensor_name,
                angle_degrees,
                self.dimensions.bottom_sensor_radius,
                body_half_height,
                self.dimensions.sensor_clearance,
                ray_length,
            )
            for sensor_name, angle_degrees in self._iter_sensor_positions(self.bottom_sensor_keys, self.bottom_sensor_angles_deg)
        ]

    def _sensor_value_from_hit_fraction(self, hit_fraction: float) -> float:
        return clamp(hit_fraction, 0.0, 1.0) * SENSOR_VALUE_MAX

    def _read_sensors_in_world(self) -> dict[str, float]:
        base_position, base_orientation = pybullet.getBasePositionAndOrientation(self.body_id, self._physics_client_id)

        def to_world(local: Sequence[float]) -> Vec3:
            return Vec3(pybullet.multiplyTransforms(base_position, base_orientation, local, (0.0, 0.0, 0.0, 1.0))[0])

        rays_list = list(self._iter_sensor_rays())
        sensor_names: list[str] = [ray.sensor_name for ray in rays_list]

        results = pybullet.rayTestBatch(
            [tuple(v) for v in [to_world(tuple(ray.start)) for ray in rays_list]],
            [tuple(v) for v in [to_world(tuple(ray.end)) for ray in rays_list]],
            self._physics_client_id,
        )
        return {
            sensor_name: self._sensor_value_from_hit_fraction(float(result[2]))
            for sensor_name, result in zip(sensor_names, results, strict=True)
        }

    def _apply_wheel_command(self, command: DifferentialWheelCommand) -> None:
        wheel_targets = (command.left * self.max_wheel_velocity, command.right * self.max_wheel_velocity)
        for joint_index, target_velocity in enumerate(wheel_targets):
            pybullet.setJointMotorControl2(
                self.body_id,
                jointIndex=joint_index,
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=self.max_motor_force,
                physicsClientId=self._physics_client_id,
            )

    def set_simulation_step_state(self, stage: MissionStage) -> None:
        normalized_direction = stage.direction.strip().lower()
        normalized_speed = clamp(stage.speed, 0.0, 1.0)

        if normalized_direction == "forward":
            self._apply_wheel_command(DifferentialWheelCommand(left=normalized_speed, right=normalized_speed))
            return
        if normalized_direction == "backward":
            self._apply_wheel_command(DifferentialWheelCommand(left=-normalized_speed, right=-normalized_speed))
            return
        if normalized_direction == "left":
            self._apply_wheel_command(DifferentialWheelCommand(left=-normalized_speed, right=normalized_speed))
            return
        if normalized_direction == "right":
            self._apply_wheel_command(DifferentialWheelCommand(left=normalized_speed, right=-normalized_speed))
            return
        raise ValueError(f"Unsupported direction '{stage.direction}'. Use forward, backward, left, or right.")

    def _control_step(
        self,
    ) -> tuple[dict[str, float], DifferentialWheelCommand]:
        readings = self._read_sensors_in_world()
        side_sensors = [readings[name] for name in self.side_sensor_keys]
        bottom_sensors = [readings[name] for name in self.bottom_sensor_keys]
        command = self._wheel_command_from_sensors(side_sensors, bottom_sensors)
        self._apply_wheel_command(command)
        pybullet.stepSimulation(physicsClientId=self._physics_client_id)
        return readings, command

    def _build_robot(
        self,
        base_position: Vec3 | None = None,
        base_yaw_degrees: float = 0.0,
        base_mass: float = 1.0,
        wheel_mass: float = 0.05,
    ) -> int:
        physics_client_id = self._physics_client_id
        body_half_height = self.dimensions.body_height / 2
        wheel_center_height = self.dimensions.wheel_radius
        base_height = wheel_center_height + body_half_height
        base_position = base_position or Vec3([0.0, 0.0, base_height])
        base_position_xyz = tuple(base_position)
        wheel_offset_y = self.dimensions.body_radius + self.dimensions.wheel_width / 2 + self.dimensions.sensor_clearance

        base_collision = pybullet.createCollisionShape(
            pybullet.GEOM_CYLINDER,
            radius=self.dimensions.body_radius,
            height=self.dimensions.body_height,
            physicsClientId=physics_client_id,
        )
        base_visual = pybullet.createVisualShape(
            pybullet.GEOM_CYLINDER,
            radius=self.dimensions.body_radius,
            length=self.dimensions.body_height,
            rgbaColor=(0.18, 0.46, 0.82, 1.0),
            physicsClientId=physics_client_id,
        )

        wheel_collision = pybullet.createCollisionShape(
            pybullet.GEOM_CYLINDER,
            radius=self.dimensions.wheel_radius,
            height=self.dimensions.wheel_width,
            physicsClientId=physics_client_id,
        )
        wheel_visual = pybullet.createVisualShape(
            pybullet.GEOM_CYLINDER,
            radius=self.dimensions.wheel_radius,
            length=self.dimensions.wheel_width,
            rgbaColor=(0.08, 0.08, 0.08, 1.0),
            physicsClientId=physics_client_id,
        )

        body_orientation = pybullet.getQuaternionFromEuler((0.0, 0.0, radians(base_yaw_degrees)))
        wheel_orientation = pybullet.getQuaternionFromEuler((1.5707963267948966, 0.0, 0.0))

        created_body_id = pybullet.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=base_position_xyz,
            baseOrientation=body_orientation,
            linkMasses=[wheel_mass, wheel_mass],
            linkCollisionShapeIndices=[wheel_collision, wheel_collision],
            linkVisualShapeIndices=[wheel_visual, wheel_visual],
            linkPositions=[
                (0.0, wheel_offset_y, -body_half_height),
                (0.0, -wheel_offset_y, -body_half_height),
            ],
            linkOrientations=[wheel_orientation, wheel_orientation],
            linkInertialFramePositions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            linkInertialFrameOrientations=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)],
            linkParentIndices=[0, 0],
            linkJointTypes=[pybullet.JOINT_REVOLUTE, pybullet.JOINT_REVOLUTE],
            # The wheel links are rotated by +90 degrees around X. In that joint frame,
            # local +Z maps to the physical wheel axle direction in the parent frame.
            linkJointAxis=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
            physicsClientId=physics_client_id,
        )
        self.body_id = int(created_body_id)
        return self.body_id
