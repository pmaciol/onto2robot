from math import isclose, pi, sin
from typing import ClassVar

import pytest

import robots_drivers.worlds.pybullet_worlds as worlds_module
from robots_drivers.pybullet_side6_bottom2_program import RobotDimensions, Vec3


class FakePyBullet:
    GEOM_CYLINDER: ClassVar[int] = 1
    GEOM_BOX: ClassVar[int] = 2

    def __init__(self):
        self.reset_calls = []
        self.gravity_calls = []
        self.time_step_calls = []
        self.multi_body_calls = []
        self.collision_shape_calls = []
        self.visual_shape_calls = []

    def resetSimulation(self, physicsClientId=None):
        self.reset_calls.append(physicsClientId)

    def setGravity(self, x, y, z, physicsClientId=None):
        self.gravity_calls.append((x, y, z, physicsClientId))

    def setTimeStep(self, timeStep, physicsClientId=None):
        self.time_step_calls.append((timeStep, physicsClientId))

    def createCollisionShape(self, *args, **kwargs):
        self.collision_shape_calls.append((args, kwargs))
        return len(self.collision_shape_calls)

    def createVisualShape(self, *args, **kwargs):
        self.visual_shape_calls.append((args, kwargs))
        return len(self.visual_shape_calls)

    def createMultiBody(self, *args, **kwargs):
        self.multi_body_calls.append((args, kwargs))
        return 100 + len(self.multi_body_calls)

    def getQuaternionFromEuler(self, euler):
        return tuple(euler)


def test_north_polar_conversion_uses_y_axis_for_zero_degrees():
    x_position, y_position = worlds_module.north_polar_to_xy(1.0, 0.0)

    assert isclose(x_position, 0.0, abs_tol=1e-9)
    assert isclose(y_position, 1.0, abs_tol=1e-9)

    x_position, y_position = worlds_module.north_polar_to_xy(1.0, 90.0)
    assert isclose(x_position, 1.0, abs_tol=1e-9)
    assert isclose(y_position, 0.0, abs_tol=1e-9)


def test_create_circular_world_builds_floor_wall_and_obstacles():
    fake_pybullet = FakePyBullet()
    config = worlds_module.WorldConfig(floor_radius=1.0, wall_segments=12)
    obstacles = [
        worlds_module.CylindricalObstacle(radius=0.1, height=0.2, center_x=0.3, center_y=0.2),
        worlds_module.CylindricalObstacle(radius=0.05, height=0.15, center_x=-0.2, center_y=0.1),
    ]

    world = worlds_module.create_circular_world(
        physics_client_id=7,
        config=config,
        obstacles=obstacles,
        time_step=0.01,
        pybullet_module=fake_pybullet,
    )

    assert world.floor_id > 0
    assert len(world.wall_ids) == 12
    assert len(world.obstacle_ids) == 2
    assert world.floor_top_z == 0.0
    assert fake_pybullet.reset_calls == [7]
    assert fake_pybullet.gravity_calls == [(0.0, 0.0, -9.81, 7)]
    assert fake_pybullet.time_step_calls == [(0.01, 7)]


def test_create_circular_world_uses_continuous_wall_segment_length():
    fake_pybullet = FakePyBullet()
    config = worlds_module.WorldConfig(floor_radius=1.2, wall_thickness=0.05, wall_segments=24)

    worlds_module.create_circular_world(
        physics_client_id=5,
        config=config,
        obstacles=[],
        pybullet_module=fake_pybullet,
    )

    wall_center_radius = config.floor_radius + config.wall_thickness / 2.0
    expected_chord = 2.0 * wall_center_radius * sin(pi / config.wall_segments)
    expected_min_length = expected_chord + config.wall_thickness

    wall_collision_calls = [
        kwargs for args, kwargs in fake_pybullet.collision_shape_calls if args and args[0] == fake_pybullet.GEOM_BOX
    ]
    assert wall_collision_calls
    half_extents = wall_collision_calls[0]["halfExtents"]
    actual_segment_length = half_extents[1] * 2.0
    assert actual_segment_length >= expected_min_length


def test_create_circular_world_orients_wall_segments_tangentially():
    fake_pybullet = FakePyBullet()
    config = worlds_module.WorldConfig(floor_radius=1.2, wall_thickness=0.05, wall_segments=24)

    worlds_module.create_circular_world(
        physics_client_id=5,
        config=config,
        obstacles=[],
        pybullet_module=fake_pybullet,
    )

    first_wall_call = fake_pybullet.multi_body_calls[1]
    kwargs = first_wall_call[1]

    assert kwargs["basePosition"] == (config.floor_radius + config.wall_thickness / 2.0, 0.0, config.wall_height / 2.0)
    assert kwargs["baseOrientation"] == (0.0, 0.0, 0.0)


def test_create_circular_world_rejects_outside_obstacle():
    fake_pybullet = FakePyBullet()
    config = worlds_module.WorldConfig(floor_radius=0.5)
    obstacles = [
        worlds_module.CylindricalObstacle(radius=0.2, height=0.2, center_x=0.4, center_y=0.0),
    ]

    with pytest.raises(ValueError, match="outside"):
        worlds_module.create_circular_world(
            physics_client_id=7,
            config=config,
            obstacles=obstacles,
            pybullet_module=fake_pybullet,
        )


def test_add_robot_from_polar_north_places_robot_on_floor(monkeypatch):
    captured_arguments = {}

    class FakeRobot:
        def __init__(self, **kwargs):
            captured_arguments.update(kwargs)

    monkeypatch.setattr(worlds_module, "PyBulletSide6Bottom2Program", FakeRobot)

    world = worlds_module.CircularWorld(
        floor_id=1,
        wall_ids=(),
        obstacle_ids=(),
        floor_top_z=0.0,
        config=worlds_module.WorldConfig(floor_radius=1.0),
    )
    dimensions = RobotDimensions()

    worlds_module.add_robot_from_polar_north(
        world=world,
        physics_client_id=9,
        start_radius=0.5,
        start_bearing_degrees_from_north=0.0,
        start_yaw_degrees_from_north=0.0,
        dimensions=dimensions,
        max_wheel_velocity=7.0,
        max_motor_force=3.0,
    )

    base_position = captured_arguments["base_position"]
    assert isinstance(base_position, Vec3)
    assert isclose(base_position.x, 0.0, abs_tol=1e-9)
    assert isclose(base_position.y, 0.5, abs_tol=1e-9)

    expected_z = dimensions.wheel_radius + dimensions.body_height / 2.0
    assert isclose(base_position.z, expected_z, abs_tol=1e-9)
    assert isclose(captured_arguments["base_yaw_degrees"], 90.0, abs_tol=1e-9)
    assert captured_arguments["client_id"] == 9
    assert captured_arguments["max_wheel_velocity"] == 7.0
    assert captured_arguments["max_motor_force"] == 3.0


def test_add_robot_from_polar_north_rejects_outside_start_radius():
    world = worlds_module.CircularWorld(
        floor_id=1,
        wall_ids=(),
        obstacle_ids=(),
        floor_top_z=0.0,
        config=worlds_module.WorldConfig(floor_radius=0.1),
    )

    with pytest.raises(ValueError, match="outside"):
        worlds_module.add_robot_from_polar_north(
            world=world,
            physics_client_id=4,
            start_radius=0.1,
            start_bearing_degrees_from_north=45.0,
        )
