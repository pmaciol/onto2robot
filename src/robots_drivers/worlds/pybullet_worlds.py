from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi, radians, sin
from typing import ClassVar, Protocol, cast

import pybullet

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


DEFAULT_WORLD_PYBULLET_MODULE: WorldPyBulletProtocol = cast(WorldPyBulletProtocol, pybullet)


@dataclass(frozen=True, slots=True)
class CylindricalObstacle:
    radius: float
    height: float
    center_x: float
    center_y: float
    rgba_color: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 1.0)


@dataclass(frozen=True, slots=True)
class WorldConfig:
    floor_radius: float
    floor_height: float = 0.01
    wall_height: float = 0.10
    wall_thickness: float = 0.03
    wall_segments: int = 48
    floor_color: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)
    wall_color: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 1.0)


@dataclass(frozen=True, slots=True)
class CircularWorld:
    floor_id: int
    wall_ids: tuple[int, ...]
    obstacle_ids: tuple[int, ...]
    floor_top_z: float
    config: WorldConfig


def north_polar_to_xy(radius: float, bearing_degrees_from_north: float) -> tuple[float, float]:
    """Convert a north-based polar coordinate to world XY.

    0 degrees points north (+Y), 90 degrees points east (+X).
    """

    angle_radians = radians(bearing_degrees_from_north)
    x_position = radius * sin(angle_radians)
    y_position = radius * cos(angle_radians)
    return x_position, y_position


# import pybullet as p
# import math

# def create_wall_collision(radius, height, thickness=0.1, segments=64, base_position=(0, 0, 0)):
#     """
#     Create a continuous circular wall in PyBullet.

#     Args:
#         radius (float): Radius of the wall (distance from center to wall center).
#         height (float): Height of the wall.
#         thickness (float): Thickness of the wall.
#         segments (int): Number of segments (higher = smoother circle).
#         base_position (tuple): Center position of the wall (x, y, z).

#     Returns:
#         list: IDs of created wall segments.
#     """

#     wall_ids = []
#     angle_step = 2 * math.pi / segments

#     for i in range(segments):
#         angle = i * angle_step

#         # Position of each segment
#         x = base_position[0] + radius * math.cos(angle)
#         y = base_position[1] + radius * math.sin(angle)
#         z = base_position[2] + height / 2.0

#         # Orientation: rotate each segment tangentially
#         yaw = angle + math.pi / 2.0
#         orientation = p.getQuaternionFromEuler([0, 0, yaw])

#         # Segment length so they connect tightly
#         segment_length = 2 * radius * math.sin(math.pi / segments)

#         collision_shape = p.createCollisionShape(
#             shapeType=p.GEOM_BOX,
#             halfExtents=[segment_length / 2.0, thickness / 2.0, height / 2.0]
#         )

#         visual_shape = p.createVisualShape(
#             shapeType=p.GEOM_BOX,
#             halfExtents=[segment_length / 2.0, thickness / 2.0, height / 2.0],
#             rgbaColor=[0.7, 0.1, 0.1, 1.0]
#         )

#         wall_id = p.createMultiBody(
#             baseMass=0,  # static object
#             baseCollisionShapeIndex=collision_shape,
#             baseVisualShapeIndex=visual_shape,
#             basePosition=[x, y, z],
#             baseOrientation=orientation
#         )

#         wall_ids.append(wall_id)

#     return wall_ids


def create_circular_world(
    physics_client_id: int,
    config: WorldConfig,
    obstacles: Sequence[CylindricalObstacle],
    *,
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

    return CircularWorld(
        floor_id=floor_id,
        wall_ids=tuple(wall_ids),
        obstacle_ids=tuple(obstacle_ids),
        floor_top_z=0.0,
        config=config,
    )


def add_robot_from_polar_north(
    world: CircularWorld,
    physics_client_id: int,
    *,
    start_radius: float,
    start_bearing_degrees_from_north: float,
    start_yaw_degrees_from_north: float = 0.0,
    dimensions: RobotDimensions | None = None,
    max_wheel_velocity: float = 6.0,
    max_motor_force: float = 2.5,
) -> PyBulletSide6Bottom2Program:
    if start_radius < 0.0:
        raise ValueError("start_radius must be non-negative.")

    robot_dimensions = dimensions or RobotDimensions()
    robot_clearance_radius = start_radius + robot_dimensions.body_radius
    if robot_clearance_radius > world.config.floor_radius:
        raise ValueError("Requested robot start position is outside floor radius.")

    x_position, y_position = north_polar_to_xy(start_radius, start_bearing_degrees_from_north)
    base_height = world.floor_top_z + robot_dimensions.wheel_radius + robot_dimensions.body_height / 2.0

    # PyBullet yaw uses +X as 0 degrees; north-based heading uses +Y as 0 degrees.
    pybullet_yaw_degrees = 90.0 - start_yaw_degrees_from_north

    return PyBulletSide6Bottom2Program(
        dimensions=robot_dimensions,
        client_id=physics_client_id,
        base_position=Vec3([x_position, y_position, base_height]),
        base_yaw_degrees=pybullet_yaw_degrees,
        max_wheel_velocity=max_wheel_velocity,
        max_motor_force=max_motor_force,
    )
