from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, radians, sin, sqrt

import pybullet

from common.utils import clamp
from onto2robot.reasoner import BOTTOM_SENSOR_KEYS, SIDE_SENSOR_KEYS
from robots_drivers.controlers.wheel_controler import DifferentialWheelCommand, WheelController, mock_decide_wheel_control

SENSOR_VALUE_MAX = 40.0
SENSOR_MARKER_RADIUS = 0.008
SENSOR_BEAM_RADIUS = 0.004
SIDE_SENSOR_MARKER_COLOR = (0.2, 0.85, 0.35, 1.0)
BOTTOM_SENSOR_MARKER_COLOR = (0.95, 0.55, 0.15, 1.0)

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
    body_height: float = 0.05
    wheel_radius: float = 0.02
    wheel_width: float = 0.01
    sensor_range: float = 0.4
    sensor_clearance: float = 0.001


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


def _sensor_marker_color(sensor_name: str) -> tuple[float, float, float, float]:
    if sensor_name in BOTTOM_SENSOR_KEYS:
        return BOTTOM_SENSOR_MARKER_COLOR
    return SIDE_SENSOR_MARKER_COLOR


def _ray_length(ray: SensorRay) -> float:
    return sqrt((ray.end.x - ray.start.x) ** 2 + (ray.end.y - ray.start.y) ** 2 + (ray.end.z - ray.start.z) ** 2)


def _point_on_ray(ray: SensorRay, distance: float) -> tuple[float, float, float]:
    ray_length = _ray_length(ray)
    if ray_length == 0.0:
        return (ray.start.x, ray.start.y, ray.start.z)

    scale = distance / ray_length
    return (
        ray.start.x + (ray.end.x - ray.start.x) * scale,
        ray.start.y + (ray.end.y - ray.start.y) * scale,
        ray.start.z + (ray.end.z - ray.start.z) * scale,
    )


def _ray_midpoint(ray: SensorRay, length: float | None = None) -> tuple[float, float, float]:
    if length is None:
        length = _ray_length(ray)
    return _point_on_ray(ray, length / 2.0)


def _visual_beam_midpoint(ray: SensorRay, beam_length: float) -> tuple[float, float, float]:
    return _ray_midpoint(ray, beam_length)


def _ray_orientation(ray: SensorRay) -> tuple[float, float, float, float]:
    ray_length = _ray_length(ray)
    if ray_length == 0.0:
        return (0.0, 0.0, 0.0, 1.0)

    direction_x = (ray.end.x - ray.start.x) / ray_length
    direction_y = (ray.end.y - ray.start.y) / ray_length
    direction_z = (ray.end.z - ray.start.z) / ray_length
    dot_product = direction_z

    if dot_product > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot_product < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)

    cross_x = -direction_y
    cross_y = direction_x
    scale = sqrt((1.0 + dot_product) * 2.0)
    inverse_scale = 1.0 / scale
    return (
        cross_x * inverse_scale,
        cross_y * inverse_scale,
        0.0,
        scale / 2.0,
    )


class PyBulletSide6Bottom2Program:
    """PyBullet model of a cylindrical robot with six side sensors and two bottom sensors."""

    def _target_bottom_gap(self) -> float:
        return self.dimensions.wheel_radius / 10.0

    def _default_base_height(self) -> float:
        body_half_height = self.dimensions.body_height / 2.0
        return body_half_height + self._target_bottom_gap()

    def _wheel_axis_height(self) -> float:
        body_top_height = self._default_base_height() + self.dimensions.body_height / 2.0
        max_axis_for_covered_wheel_top = body_top_height - self.dimensions.wheel_radius - self.dimensions.sensor_clearance
        return min(self.dimensions.wheel_radius, max_axis_for_covered_wheel_top)

    def run_control_loop(self, steps: int) -> list[tuple[dict[str, float], DifferentialWheelCommand]]:
        history: list[tuple[dict[str, float], DifferentialWheelCommand]] = []
        for _ in range(steps):
            history.append(self._control_step())
        return history

    def apply_wheel_controller_step(self) -> DifferentialWheelCommand:
        """Advance one simulation step using the configured wheel controller."""
        _, command = self._control_step()
        return command

    def reset(self) -> None:
        base_position = (0.0, 0.0, self._default_base_height())
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

    def get_sensor_readings(self) -> dict[str, float]:
        """Return the current side and bottom sensor readings in world space."""
        return self._read_sensors_in_world()

    def get_bottom_sensor_readings(self) -> dict[str, float]:
        """Return the current bottom sensor readings in sensor-key order."""
        readings = self._read_sensors_in_world()
        return {sensor_name: readings[sensor_name] for sensor_name in self.bottom_sensor_keys}

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
        line_track_body_ids: Sequence[int] | None = None,
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
        self._line_track_body_ids = None if line_track_body_ids is None else frozenset(line_track_body_ids)
        self._bottom_sensor_key_set = set(BOTTOM_SENSOR_KEYS)
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
        side_sensor_height = body_half_height - self.dimensions.sensor_clearance
        ray_length = self.sensor_max_distance + self.dimensions.sensor_clearance

        return [
            _make_side_sensor_ray(
                sensor_name,
                angle_degrees,
                self.dimensions.body_radius,
                side_sensor_height,
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

    def _sensor_value_from_hit_fraction(self, hit_object_id: int, hit_fraction: float, ray_length: float) -> float:
        if hit_object_id < 0:
            return 0.0

        hit_distance = clamp(hit_fraction, 0.0, 1.0) * ray_length
        if hit_distance >= 0.2:
            return 0.0
        if hit_distance <= 0.01:
            return SENSOR_VALUE_MAX

        scaled = (0.2 - hit_distance) / 0.19
        return clamp(scaled, 0.0, 1.0) * SENSOR_VALUE_MAX

    def _read_sensors_in_world(self) -> dict[str, float]:
        base_position, base_orientation = pybullet.getBasePositionAndOrientation(self.body_id, self._physics_client_id)

        def to_world(local: Sequence[float]) -> Vec3:
            return Vec3(pybullet.multiplyTransforms(base_position, base_orientation, local, (0.0, 0.0, 0.0, 1.0))[0])

        rays_list = list(self._iter_sensor_rays())
        sensor_names: list[str] = [ray.sensor_name for ray in rays_list]
        ray_lengths: list[float] = [
            ((ray.end.x - ray.start.x) ** 2 + (ray.end.y - ray.start.y) ** 2 + (ray.end.z - ray.start.z) ** 2) ** 0.5
            for ray in rays_list
        ]

        results = pybullet.rayTestBatch(
            [tuple(v) for v in [to_world(tuple(ray.start)) for ray in rays_list]],
            [tuple(v) for v in [to_world(tuple(ray.end)) for ray in rays_list]],
            physicsClientId=self._physics_client_id,
        )

        def sensor_value(sensor_name: str, result: tuple[object, object, object, object, object], ray_length: float) -> float:
            hit_object_id = int(result[0])
            if (
                sensor_name in self._bottom_sensor_key_set
                and self._line_track_body_ids is not None
                and hit_object_id not in self._line_track_body_ids
            ):
                return 0.0
            return self._sensor_value_from_hit_fraction(hit_object_id, float(result[2]), ray_length)

        return {
            sensor_name: sensor_value(sensor_name, result, ray_length)
            for sensor_name, result, ray_length in zip(sensor_names, results, ray_lengths, strict=True)
        }

    def _apply_wheel_command(self, command: DifferentialWheelCommand) -> None:
        # PyBullet wheel joint orientation yields opposite linear motion for positive angular velocity.
        # Invert signs so "forward" commands move toward the front sensor arc.
        wheel_targets = (-command.left * self.max_wheel_velocity, -command.right * self.max_wheel_velocity)
        for joint_index, target_velocity in enumerate(wheel_targets):
            pybullet.setJointMotorControl2(
                self.body_id,
                jointIndex=joint_index,
                controlMode=pybullet.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=self.max_motor_force,
                physicsClientId=self._physics_client_id,
            )

    def set_simulation_stage(self, stage: MissionStage) -> None:
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
        print("Sensor readings: %s", readings)
        side_sensors = [readings[name] for name in self.side_sensor_keys]
        bottom_sensors = [readings[name] for name in self.bottom_sensor_keys]
        command = self._wheel_command_from_sensors(side_sensors, bottom_sensors)
        self._apply_wheel_command(command)
        pybullet.stepSimulation(physicsClientId=self._physics_client_id)
        return readings, command

    def _append_visual_sensor_links(
        self,
        sensor_rays: Sequence[SensorRay],
        beam_length: float,
        link_masses: list[float],
        link_collision_shape_indices: list[int],
        link_visual_shape_indices: list[int],
        link_positions: list[tuple[float, float, float]],
        link_orientations: list[tuple[float, float, float, float]],
        link_inertial_frame_positions: list[tuple[float, float, float]],
        link_inertial_frame_orientations: list[tuple[float, float, float, float]],
        link_parent_indices: list[int],
        link_joint_types: list[int],
        link_joint_axis: list[tuple[float, float, float]],
        marker_visual_shape_indices: dict[str, int],
        beam_visual_shape_indices: dict[str, int],
    ) -> None:
        fixed_orientation = (0.0, 0.0, 0.0, 1.0)
        for ray in sensor_rays:
            link_masses.append(0.0)
            link_collision_shape_indices.append(-1)
            link_visual_shape_indices.append(marker_visual_shape_indices[ray.sensor_name])
            link_positions.append((ray.start.x, ray.start.y, ray.start.z))
            link_orientations.append(fixed_orientation)
            link_inertial_frame_positions.append((0.0, 0.0, 0.0))
            link_inertial_frame_orientations.append(fixed_orientation)
            link_parent_indices.append(0)
            link_joint_types.append(pybullet.JOINT_FIXED)
            link_joint_axis.append((0.0, 0.0, 0.0))

            link_masses.append(0.0)
            link_collision_shape_indices.append(-1)
            link_visual_shape_indices.append(beam_visual_shape_indices[ray.sensor_name])
            link_positions.append(_visual_beam_midpoint(ray, beam_length))
            link_orientations.append(_ray_orientation(ray))
            link_inertial_frame_positions.append((0.0, 0.0, 0.0))
            link_inertial_frame_orientations.append(fixed_orientation)
            link_parent_indices.append(0)
            link_joint_types.append(pybullet.JOINT_FIXED)
            link_joint_axis.append((0.0, 0.0, 0.0))

    def _build_robot(
        self,
        base_position: Vec3 | None = None,
        base_yaw_degrees: float = 0.0,
        base_mass: float = 1.0,
        wheel_mass: float = 0.05,
    ) -> int:
        physics_client_id = self._physics_client_id
        base_height = self._default_base_height()
        wheel_center_height = self._wheel_axis_height()
        wheel_link_z = wheel_center_height - base_height
        base_position = base_position or Vec3([0.0, 0.0, base_height])
        base_position_xyz = tuple(base_position)
        wheel_offset_y = self.dimensions.body_radius + self.dimensions.wheel_width / 2 + self.dimensions.sensor_clearance
        sensor_rays = self._iter_sensor_rays()

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
        marker_visual_shape_indices = {
            ray.sensor_name: pybullet.createVisualShape(
                pybullet.GEOM_SPHERE,
                radius=SENSOR_MARKER_RADIUS,
                rgbaColor=_sensor_marker_color(ray.sensor_name),
                physicsClientId=physics_client_id,
            )
            for ray in sensor_rays
        }
        beam_visual_shape_indices = {
            ray.sensor_name: pybullet.createVisualShape(
                pybullet.GEOM_CYLINDER,
                radius=SENSOR_BEAM_RADIUS,
                length=self.sensor_max_distance,
                rgbaColor=_sensor_marker_color(ray.sensor_name),
                physicsClientId=physics_client_id,
            )
            for ray in sensor_rays
        }

        body_orientation = pybullet.getQuaternionFromEuler((0.0, 0.0, radians(base_yaw_degrees)))
        wheel_orientation = pybullet.getQuaternionFromEuler((1.5707963267948966, 0.0, 0.0))

        fixed_orientation = (0.0, 0.0, 0.0, 1.0)
        link_masses = [wheel_mass, wheel_mass]
        link_collision_shape_indices = [wheel_collision, wheel_collision]
        link_visual_shape_indices = [wheel_visual, wheel_visual]
        link_positions = [
            (0.0, wheel_offset_y, wheel_link_z),
            (0.0, -wheel_offset_y, wheel_link_z),
        ]
        link_orientations = [wheel_orientation, wheel_orientation]
        link_inertial_frame_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        link_inertial_frame_orientations = [fixed_orientation, fixed_orientation]
        link_parent_indices = [0, 0]
        link_joint_types = [pybullet.JOINT_REVOLUTE, pybullet.JOINT_REVOLUTE]
        link_joint_axis = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]

        self._append_visual_sensor_links(
            sensor_rays,
            self.sensor_max_distance,
            link_masses,
            link_collision_shape_indices,
            link_visual_shape_indices,
            link_positions,
            link_orientations,
            link_inertial_frame_positions,
            link_inertial_frame_orientations,
            link_parent_indices,
            link_joint_types,
            link_joint_axis,
            marker_visual_shape_indices,
            beam_visual_shape_indices,
        )

        created_body_id = pybullet.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=base_position_xyz,
            baseOrientation=body_orientation,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shape_indices,
            linkVisualShapeIndices=link_visual_shape_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frame_positions,
            linkInertialFrameOrientations=link_inertial_frame_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            # The wheel links are rotated by +90 degrees around X. In that joint frame,
            # local +Z maps to the physical wheel axle direction in the parent frame.
            linkJointAxis=link_joint_axis,
            physicsClientId=physics_client_id,
        )
        self.body_id = int(created_body_id)

        if self._line_track_body_ids:
            joint_count = pybullet.getNumJoints(self.body_id, physicsClientId=physics_client_id)
            for line_track_body_id in self._line_track_body_ids:
                for robot_link_index in range(-1, joint_count):
                    pybullet.setCollisionFilterPair(
                        int(line_track_body_id),
                        self.body_id,
                        -1,
                        robot_link_index,
                        0,
                        physicsClientId=physics_client_id,
                    )

        return self.body_id
