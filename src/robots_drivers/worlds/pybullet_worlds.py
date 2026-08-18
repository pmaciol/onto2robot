from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2, cos, pi, radians, sin, sqrt
from typing import ClassVar, Protocol, cast

import pybullet

from robots_drivers.controlers.wheel_controler import WheelController
from robots_drivers.pybullet_side6_bottom2_program import PyBulletSide6Bottom2Program, RobotDimensions, Vec3


class WorldPyBulletProtocol(Protocol):
    GEOM_CYLINDER: ClassVar[int]
    GEOM_BOX: ClassVar[int]

    def resetSimulation(self, physicsClientId: int | None = None) -> None: ...

    def setGravity(self, x: float, y: float, z: float, physicsClientId: int | None = None) -> None: ...

    def setTimeStep(self, timeStep: float, physicsClientId: int | None = None) -> None: ...

    def createCollisionShape(self, shapeType: int, **kwargs: object) -> int: ...

    def createVisualShape(self, shapeType: int, **kwargs: object) -> int: ...

    def createMultiBody(self, **kwargs: object) -> int: ...

    def getQuaternionFromEuler(self, euler: Sequence[float]) -> Sequence[float]: ...

    def setCollisionFilterGroupMask(
        self,
        bodyUniqueId: int,
        linkIndex: int,
        collisionFilterGroup: int,
        collisionFilterMask: int,
    ) -> None: ...


DEFAULT_WORLD_PYBULLET_MODULE: WorldPyBulletProtocol = cast(WorldPyBulletProtocol, pybullet)


@dataclass(frozen=True, slots=True)
class CylindricalObstacle:
    radius: float
    height: float
    center_x: float
    center_y: float
    rgba_color: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 1.0)


@dataclass(frozen=True, slots=True)
class WorldConfigCircular:
    floor_radius: float
    floor_height: float = 0.01
    wall_height: float = 0.10
    wall_thickness: float = 0.03
    wall_segments: int = 48
    floor_color: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)
    wall_color: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 1.0)


@dataclass(frozen=True, slots=True)
class PaintedPathConfig:
    radius_x: float
    radius_y: float
    width: float
    segment_count: int = 96
    height: float = 0.001
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class CircularWorld:
    floor_id: int
    wall_ids: tuple[int, ...]
    obstacle_ids: tuple[int, ...]
    floor_top_z: float
    config: WorldConfigCircular
    physics_client_id: int
    path_ids: tuple[int, ...] = ()


def north_polar_to_xy(radius: float, bearing_degrees_from_north: float) -> tuple[float, float]:
    """Convert a north-based polar coordinate to world XY.

    0 degrees points north (+Y), 90 degrees points east (+X).
    """

    angle_radians = radians(bearing_degrees_from_north)
    x_position = radius * sin(angle_radians)
    y_position = radius * cos(angle_radians)
    return x_position, y_position


def _ellipse_point(radius_x: float, radius_y: float, angle: float) -> tuple[float, float]:
    return radius_x * cos(angle), radius_y * sin(angle)


def _ellipse_tangent_yaw(radius_x: float, radius_y: float, angle: float) -> float:
    tangent_x = -radius_x * sin(angle)
    tangent_y = radius_y * cos(angle)
    return atan2(tangent_y, tangent_x) - (pi / 2.0)


def _point_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


def _create_painted_path(
    physics_client_id: int,
    floor_top_z: float,
    config: PaintedPathConfig,
    pybullet_module: WorldPyBulletProtocol,
) -> tuple[int, ...]:
    if config.radius_x <= 0.0:
        raise ValueError("radius_x must be greater than 0.")
    if config.radius_y <= 0.0:
        raise ValueError("radius_y must be greater than 0.")
    if config.width <= 0.0:
        raise ValueError("width must be greater than 0.")
    if config.segment_count < 12:
        raise ValueError("segment_count must be at least 12.")
    if config.height <= 0.0:
        raise ValueError("height must be greater than 0.")

    half_height = config.height / 2.0
    path_ids: list[int] = []
    angles = [2.0 * pi * index / config.segment_count for index in range(config.segment_count)]
    points = [_ellipse_point(config.radius_x, config.radius_y, angle) for angle in angles]

    for segment_index, angle in enumerate(angles):
        point_x, point_y = points[segment_index]
        previous_point = points[segment_index - 1]
        next_point = points[(segment_index + 1) % config.segment_count]
        segment_length = max(_point_distance(previous_point, next_point) / 2.0, config.width)

        collision_shape = pybullet_module.createCollisionShape(
            pybullet_module.GEOM_BOX,
            halfExtents=(config.width / 2.0, segment_length / 2.0, half_height),
            physicsClientId=physics_client_id,
        )
        visual_shape = pybullet_module.createVisualShape(
            pybullet_module.GEOM_BOX,
            halfExtents=(config.width / 2.0, segment_length / 2.0, half_height),
            rgbaColor=config.color,
            physicsClientId=physics_client_id,
        )
        segment_orientation = pybullet_module.getQuaternionFromEuler(
            (0.0, 0.0, _ellipse_tangent_yaw(config.radius_x, config.radius_y, angle))
        )
        path_id = int(
            pybullet_module.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=(point_x, point_y, floor_top_z + half_height),
                baseOrientation=segment_orientation,
                physicsClientId=physics_client_id,
            )
        )
        path_ids.append(path_id)

    return tuple(path_ids)


def create_circular_world(
    physics_client_id: int,
    config: WorldConfigCircular,
    obstacles: Sequence[CylindricalObstacle],
    *,
    painted_path: PaintedPathConfig | None = None,
    gravity_z: float = -9.81,
    time_step: float = 1.0 / 60.0,
    pybullet_module: WorldPyBulletProtocol = DEFAULT_WORLD_PYBULLET_MODULE,
) -> CircularWorld:
    if config.floor_radius <= 0.0:
        raise ValueError("floor_radius must be greater than 0.")
    if config.floor_height <= 0.0:
        raise ValueError("floor_height must be greater than 0.")
    if config.wall_height <= 0.0:
        raise ValueError("wall_height must be greater than 0.")
    if config.wall_thickness <= 0.0:
        raise ValueError("wall_thickness must be greater than 0.")
    if config.wall_segments < 8:
        raise ValueError("wall_segments must be at least 8.")
    if time_step <= 0.0:
        raise ValueError("time_step must be greater than 0.")

    pybullet_module.resetSimulation(physicsClientId=physics_client_id)
    pybullet_module.setGravity(0.0, 0.0, gravity_z, physicsClientId=physics_client_id)
    pybullet_module.setTimeStep(time_step, physicsClientId=physics_client_id)

    floor_collision = pybullet_module.createCollisionShape(
        pybullet_module.GEOM_CYLINDER,
        radius=config.floor_radius,
        height=config.floor_height,
        physicsClientId=physics_client_id,
    )
    floor_visual = pybullet_module.createVisualShape(
        pybullet_module.GEOM_CYLINDER,
        radius=config.floor_radius,
        length=config.floor_height,
        rgbaColor=config.floor_color,
        physicsClientId=physics_client_id,
    )
    floor_id = int(
        pybullet_module.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_collision,
            baseVisualShapeIndex=floor_visual,
            basePosition=(0.0, 0.0, -config.floor_height / 2.0),
            physicsClientId=physics_client_id,
        )
    )

    wall_ids: list[int] = []
    wall_center_radius = config.floor_radius + config.wall_thickness / 2.0
    wall_segment_chord_length = 2.0 * wall_center_radius * sin(pi / config.wall_segments)
    # Add one wall-thickness of overlap to avoid gaps between neighboring segments.
    wall_segment_length = max(wall_segment_chord_length + config.wall_thickness, config.wall_thickness)
    wall_collision = pybullet_module.createCollisionShape(
        pybullet_module.GEOM_BOX,
        halfExtents=(config.wall_thickness / 2.0, wall_segment_length / 2.0, config.wall_height / 2.0),
        physicsClientId=physics_client_id,
    )
    wall_visual = pybullet_module.createVisualShape(
        pybullet_module.GEOM_BOX,
        halfExtents=(config.wall_thickness / 2.0, wall_segment_length / 2.0, config.wall_height / 2.0),
        rgbaColor=config.wall_color,
        physicsClientId=physics_client_id,
    )

    for segment_index in range(config.wall_segments):
        angle = 2.0 * pi * segment_index / config.wall_segments
        x_position = wall_center_radius * cos(angle)
        y_position = wall_center_radius * sin(angle)
        # Keep the long axis tangential and the thin axis radial.
        segment_yaw = angle
        segment_orientation = pybullet_module.getQuaternionFromEuler((0.0, 0.0, segment_yaw))
        wall_id = int(
            pybullet_module.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=(x_position, y_position, config.wall_height / 2.0),
                baseOrientation=segment_orientation,
                physicsClientId=physics_client_id,
            )
        )
        wall_ids.append(wall_id)

    obstacle_ids: list[int] = []
    for obstacle in obstacles:
        if obstacle.radius <= 0.0:
            raise ValueError("Obstacle radius must be greater than 0.")
        if obstacle.height <= 0.0:
            raise ValueError("Obstacle height must be greater than 0.")
        center_distance = (obstacle.center_x**2 + obstacle.center_y**2) ** 0.5
        if center_distance + obstacle.radius > config.floor_radius:
            raise ValueError("Obstacle lies outside the circular floor.")

        obstacle_collision = pybullet_module.createCollisionShape(
            pybullet_module.GEOM_CYLINDER,
            radius=obstacle.radius,
            height=obstacle.height,
            physicsClientId=physics_client_id,
        )
        obstacle_visual = pybullet_module.createVisualShape(
            pybullet_module.GEOM_CYLINDER,
            radius=obstacle.radius,
            length=obstacle.height,
            rgbaColor=obstacle.rgba_color,
            physicsClientId=physics_client_id,
        )
        obstacle_id = int(
            pybullet_module.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=obstacle_collision,
                baseVisualShapeIndex=obstacle_visual,
                basePosition=(obstacle.center_x, obstacle.center_y, obstacle.height / 2.0),
                physicsClientId=physics_client_id,
            )
        )
        obstacle_ids.append(obstacle_id)

    path_ids: tuple[int, ...] = ()
    if painted_path is not None:
        path_ids = _create_painted_path(
            physics_client_id=physics_client_id,
            floor_top_z=0.0,
            config=painted_path,
            pybullet_module=pybullet_module,
        )

    return CircularWorld(
        floor_id=floor_id,
        wall_ids=tuple(wall_ids),
        obstacle_ids=tuple(obstacle_ids),
        floor_top_z=0.0,
        config=config,
        physics_client_id=physics_client_id,
        path_ids=path_ids,
    )


def add_robot_from_polar_north(
    world: CircularWorld,
    *,
    reasoner: WheelController,
    start_radius: float,
    start_bearing_degrees_from_north: float,
    start_yaw_degrees_from_north: float = 0.0,
    dimensions: RobotDimensions | None = None,
    max_wheel_velocity: float = 1.0,
    max_motor_force: float = 2.5,
) -> PyBulletSide6Bottom2Program:
    if start_radius < 0.0:
        raise ValueError("start_radius must be non-negative.")

    robot_dimensions = dimensions or RobotDimensions()
    robot_clearance_radius = start_radius + robot_dimensions.body_radius
    if robot_clearance_radius > world.config.floor_radius:
        raise ValueError("Requested robot start position is outside floor radius.")

    x_position, y_position = north_polar_to_xy(start_radius, start_bearing_degrees_from_north)
    base_height = world.floor_top_z + robot_dimensions.body_height / 2.0 + robot_dimensions.wheel_radius / 10.0

    # PyBullet yaw uses +X as 0 degrees; north-based heading uses +Y as 0 degrees.
    pybullet_yaw_degrees = 90.0 - start_yaw_degrees_from_north

    return PyBulletSide6Bottom2Program(
        dimensions=robot_dimensions,
        client_id=world.physics_client_id,
        base_position=Vec3([x_position, y_position, base_height]),
        base_yaw_degrees=pybullet_yaw_degrees,
        max_wheel_velocity=max_wheel_velocity,
        max_motor_force=max_motor_force,
        wheel_controller=reasoner,
        line_track_body_ids=world.path_ids,
    )
