from math import isclose
from pathlib import Path

import pybullet as real_pybullet
import pytest

import robots_drivers.controlers.wheel_controler as wheel_module
import robots_drivers.pybullet_mission_runner as mission_module
import robots_drivers.pybullet_side6_bottom2_program as driver_module
from robots_drivers.worlds.pybullet_worlds import PaintedPathConfig, add_robot_from_polar_north, create_circular_world


class FakePyBullet:
    GEOM_CYLINDER = 1
    GEOM_SPHERE = 2
    JOINT_REVOLUTE = 2
    JOINT_FIXED = 4
    VELOCITY_CONTROL = 3

    def __init__(self):
        self.motor_calls = []
        self.ray_batches = []
        self.step_calls = 0
        self.reset_base_calls = []
        self.reset_joint_calls = []
        self.create_multibody_calls = []
        self.visual_shape_calls = []

    def createCollisionShape(self, *args, **kwargs):
        return 11

    def createVisualShape(self, *args, **kwargs):
        self.visual_shape_calls.append((args, kwargs))
        return 12

    def getQuaternionFromEuler(self, euler):
        return tuple(euler)

    def createMultiBody(self, *args, **kwargs):
        self.create_multibody_calls.append((args, kwargs))
        return 99

    def getBasePositionAndOrientation(self, body_id, physicsClientId=None):
        return (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)

    def multiplyTransforms(self, base_position, base_orientation, local_position, local_orientation):
        world_position = tuple(base + local for base, local in zip(base_position, local_position, strict=True))
        return world_position, local_orientation

    def rayTestBatch(self, starts, ends, physicsClientId=None):
        self.ray_batches.append((starts, ends, physicsClientId))
        return [(0, 0, 0.25, None, None) for _ in starts]

    def setJointMotorControl2(self, bodyUniqueId, jointIndex, controlMode, targetVelocity, force, physicsClientId=None):
        self.motor_calls.append(
            {
                "bodyUniqueId": bodyUniqueId,
                "jointIndex": jointIndex,
                "controlMode": controlMode,
                "targetVelocity": targetVelocity,
                "force": force,
                "physicsClientId": physicsClientId,
            }
        )

    def stepSimulation(self, physicsClientId=None):
        self.step_calls += 1

    def resetBasePositionAndOrientation(self, bodyUniqueId, posObj, ornObj, physicsClientId=None):
        self.reset_base_calls.append((bodyUniqueId, posObj, ornObj, physicsClientId))

    def resetJointState(self, bodyUniqueId, jointIndex, targetValue, physicsClientId=None):
        self.reset_joint_calls.append((bodyUniqueId, jointIndex, targetValue, physicsClientId))


BOTTOM_SENSOR_KEYS = driver_module.BOTTOM_SENSOR_KEYS
SIDE_SENSOR_KEYS = driver_module.SIDE_SENSOR_KEYS
SIDE_SENSOR_ANGLES_DEG = driver_module.SIDE_SENSOR_ANGLES_DEG
BOTTOM_SENSOR_ANGLES_DEG = driver_module.BOTTOM_SENSOR_ANGLES_DEG

PyBulletSide6Bottom2Program = driver_module.PyBulletSide6Bottom2Program
DifferentialWheelCommand = driver_module.DifferentialWheelCommand
mock_decide_wheel_control = driver_module.mock_decide_wheel_control


def test_sensor_layout_matches_requested_geometry():
    robot = PyBulletSide6Bottom2Program()

    assert robot.side_sensor_keys == SIDE_SENSOR_KEYS
    assert robot.bottom_sensor_keys == BOTTOM_SENSOR_KEYS
    assert robot.side_sensor_angles_deg == SIDE_SENSOR_ANGLES_DEG
    assert robot.bottom_sensor_angles_deg == BOTTOM_SENSOR_ANGLES_DEG

    rays = robot._iter_sensor_rays()  # noqa: SLF001
    assert len(rays) == 8


def test_run_control_loop_returns_requested_number_of_steps(monkeypatch):
    robot = PyBulletSide6Bottom2Program()
    calls: list[int] = []

    def fake_control_step():
        calls.append(1)
        return ({SIDE_SENSOR_KEYS[0]: 10.0, BOTTOM_SENSOR_KEYS[0]: 5.0}, DifferentialWheelCommand(left=0.4, right=0.6))

    monkeypatch.setattr(robot, "_control_step", fake_control_step)  # noqa: SLF001

    history = robot.run_control_loop(steps=3)

    assert len(history) == 3
    assert len(calls) == 3
    for readings, command in history:
        assert readings[SIDE_SENSOR_KEYS[0]] == 10.0
        assert command == DifferentialWheelCommand(left=0.4, right=0.6)


def test_run_control_loop_with_zero_steps_returns_empty_history():
    robot = PyBulletSide6Bottom2Program()

    assert robot.run_control_loop(steps=0) == []


def test_getters_expose_current_and_initial_pose(monkeypatch):
    fake_pybullet = FakePyBullet()
    monkeypatch.setattr(driver_module, "pybullet", fake_pybullet)

    initial_position = driver_module.Vec3([0.25, -0.15, 0.08])
    initial_yaw = 35.0
    robot = PyBulletSide6Bottom2Program(base_position=initial_position, base_yaw_degrees=initial_yaw)

    position = robot.get_position()
    rotation = robot.get_rotation()

    assert position == driver_module.Vec3([1.0, 2.0, 3.0])
    assert rotation == (0.0, 0.0, 0.0, 1.0)
    assert robot.get_initial_position() == initial_position
    assert robot.get_initial_rotation_degrees() == initial_yaw


def test_build_robot_adds_sensor_origin_markers_and_direction_lines(monkeypatch):
    fake_pybullet = FakePyBullet()
    monkeypatch.setattr(driver_module, "pybullet", fake_pybullet)

    robot = PyBulletSide6Bottom2Program()
    rays = robot._iter_sensor_rays()  # noqa: SLF001
    create_multibody_kwargs = fake_pybullet.create_multibody_calls[0][1]

    assert len(create_multibody_kwargs["linkMasses"]) == 18
    assert create_multibody_kwargs["linkJointTypes"][:2] == [fake_pybullet.JOINT_REVOLUTE, fake_pybullet.JOINT_REVOLUTE]
    assert create_multibody_kwargs["linkJointTypes"][2:] == [fake_pybullet.JOINT_FIXED] * (len(rays) * 2)
    assert create_multibody_kwargs["linkCollisionShapeIndices"][2:] == [-1] * (len(rays) * 2)
    assert create_multibody_kwargs["linkPositions"][2::2] == [tuple(ray.start) for ray in rays]
    assert create_multibody_kwargs["linkPositions"][3::2] == [
        driver_module._visual_beam_midpoint(ray, robot.sensor_max_distance) for ray in rays
    ]
    assert create_multibody_kwargs["linkOrientations"][3::2] == [driver_module._ray_orientation(ray) for ray in rays]

    assert len(fake_pybullet.visual_shape_calls) == 18
    marker_shape_calls = fake_pybullet.visual_shape_calls[2:10]
    beam_shape_calls = fake_pybullet.visual_shape_calls[10:]

    for (_, kwargs), ray in zip(marker_shape_calls, rays, strict=True):
        assert kwargs["radius"] == driver_module.SENSOR_MARKER_RADIUS
        assert kwargs["rgbaColor"] == driver_module._sensor_marker_color(ray.sensor_name)

    for (_, kwargs), ray in zip(beam_shape_calls, rays, strict=True):
        assert kwargs["radius"] == driver_module.SENSOR_BEAM_RADIUS
        assert kwargs["length"] == pytest.approx(robot.sensor_max_distance)
        assert kwargs["rgbaColor"] == driver_module._sensor_marker_color(ray.sensor_name)


def test_wheel_command_from_sensors_raises_for_wrong_side_sensor_count():
    robot = PyBulletSide6Bottom2Program()
    with pytest.raises(ValueError, match="side sensor"):
        robot._wheel_command_from_sensors([1.0, 2.0], [10.0, 10.0])  # noqa: SLF001


def test_wheel_command_from_sensors_raises_for_wrong_bottom_sensor_count():
    robot = PyBulletSide6Bottom2Program()
    with pytest.raises(ValueError, match="bottom sensor"):
        robot._wheel_command_from_sensors([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0])  # noqa: SLF001


def test_control_step_reads_rays_and_drives_wheels(monkeypatch):
    fake_pybullet = FakePyBullet()
    monkeypatch.setattr(driver_module, "pybullet", fake_pybullet)

    robot = PyBulletSide6Bottom2Program()
    readings, command = robot._control_step()  # noqa: SLF001

    assert len(readings) == 8
    assert isclose(readings[SIDE_SENSOR_KEYS[0]], 21.0)
    assert isclose(command.left, command.right)
    assert fake_pybullet.step_calls == 1
    assert len(fake_pybullet.ray_batches) == 1
    assert len(fake_pybullet.motor_calls) == 2
    assert fake_pybullet.motor_calls[0]["jointIndex"] == 0
    assert fake_pybullet.motor_calls[1]["jointIndex"] == 1


def test_reset_zeros_base_and_joints(monkeypatch):
    fake_pybullet = FakePyBullet()
    monkeypatch.setattr(driver_module, "pybullet", fake_pybullet)

    robot = PyBulletSide6Bottom2Program()
    robot.reset()

    assert fake_pybullet.reset_base_calls
    assert len(fake_pybullet.reset_joint_calls) == 2


def test_custom_base_position_height_is_applied_on_spawn():
    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        real_pybullet.resetSimulation(physicsClientId=client_id)
        real_pybullet.setGravity(0.0, 0.0, 0.0, physicsClientId=client_id)

        requested_height = 0.25
        robot = PyBulletSide6Bottom2Program(
            client_id=client_id,
            base_position=driver_module.Vec3([0.0, 0.0, requested_height]),
        )
        position, _ = real_pybullet.getBasePositionAndOrientation(robot.body_id, physicsClientId=client_id)

        assert isclose(position[2], requested_height, abs_tol=1e-6)
    finally:
        real_pybullet.disconnect(client_id)


def test_controller_factory_creates_controller_and_converts_sensor_inputs(monkeypatch):
    class FakeReasoner:
        last_instance = None

        def __init__(self, *args, **kwargs):
            self.last_input_values = None
            FakeReasoner.last_instance = self

        def reason(self, input_values):
            self.last_input_values = input_values
            return 20.0

    monkeypatch.setattr(wheel_module, "Reasoner", FakeReasoner)

    controller = wheel_module.controller_factory(wheel_module.ReasonerTypes.AMRO)

    assert isinstance(controller, wheel_module.Controller)

    side_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    bottom_values = [7.0, 8.0]
    command = controller(side_values, bottom_values)

    assert command.left == 5.0
    assert command.right == 5.0
    assert FakeReasoner.last_instance is not None
    assert FakeReasoner.last_instance.last_input_values == {
        sensor_name: sensor_value
        for sensor_name, sensor_value in zip(
            (*SIDE_SENSOR_KEYS, *BOTTOM_SENSOR_KEYS), (*side_values, *bottom_values), strict=True
        )
    }


def test_pybullet_program_accepts_factory_controller_for_basic_wheel_command(monkeypatch):
    class FakeReasoner:
        def __init__(self, *args, **kwargs):
            pass

        def reason(self, input_values):
            return 20.0

    fake_pybullet = FakePyBullet()
    monkeypatch.setattr(wheel_module, "Reasoner", FakeReasoner)
    monkeypatch.setattr(driver_module, "pybullet", fake_pybullet)

    controller = wheel_module.controller_factory(wheel_module.ReasonerTypes.AMRO)
    robot = PyBulletSide6Bottom2Program(wheel_controller=controller)

    command = robot._wheel_command_from_sensors(  # noqa: SLF001
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0],
    )

    assert isinstance(robot.wheel_controller, wheel_module.Controller)
    assert command == DifferentialWheelCommand(left=5.0, right=5.0)


def test_pybullet_program_with_real_reasoner_controller_factory():
    # Use a real ontology-backed reasoner (no monkeypatch) to validate the integration path.
    controller = wheel_module.controller_factory(wheel_module.ReasonerTypes.AMRO)
    robot = PyBulletSide6Bottom2Program(wheel_controller=controller)

    side_values = [20.0, 1.0, 1.0, 1.0, 1.0, 20.0]
    bottom_values = [20.0, 20.0]

    first_command = robot._wheel_command_from_sensors(side_values, bottom_values)  # noqa: SLF001
    second_command = robot._wheel_command_from_sensors(side_values, bottom_values)  # noqa: SLF001

    assert isinstance(robot.wheel_controller, wheel_module.Controller)
    assert -5.0 <= first_command.left <= 5.0
    assert -5.0 <= first_command.right <= 5.0
    assert isclose(first_command.left, second_command.left, abs_tol=1e-6)
    assert isclose(first_command.right, second_command.right, abs_tol=1e-6)


def test_sensor_values_thresholds_with_mock_world(monkeypatch):
    class ThresholdRayPyBullet(FakePyBullet):
        def rayTestBatch(self, starts, ends, physicsClientId=None):
            self.ray_batches.append((starts, ends, physicsClientId))
            ray_length = 0.401
            far_fraction = 0.21 / ray_length
            near_fraction = 0.005 / ray_length

            readings = [(-1, -1, far_fraction, None, None) for _ in starts]
            readings[0] = (0, 0, near_fraction, None, None)
            return readings

    fake_pybullet = ThresholdRayPyBullet()
    monkeypatch.setattr(driver_module, "pybullet", fake_pybullet)

    robot = PyBulletSide6Bottom2Program()
    readings = robot._read_sensors_in_world()  # noqa: SLF001

    # Proper threshold behavior: >0.2m -> 0, <0.01m -> 40
    assert readings[SIDE_SENSOR_KEYS[1]] == 0.0
    assert readings[SIDE_SENSOR_KEYS[0]] == 40.0


def test_sensor_values_thresholds_with_real_world_central_obstacle():
    obstacles_path = Path(__file__).parent / "data" / "sample_world_central_obstacle.json"

    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        world = create_circular_world(
            client_id,
            mission_module.WorldConfigCircular(floor_radius=1.5),
            mission_module.load_obstacles(obstacles_path),
            time_step=1.0 / 60.0,
        )

        far_robot = add_robot_from_polar_north(
            world,
            start_radius=0.50,
            start_bearing_degrees_from_north=0.0,
            start_yaw_degrees_from_north=180.0,
            reasoner=mock_decide_wheel_control,
        )
        far_readings = far_robot._read_sensors_in_world()  # noqa: SLF001

        near_robot = add_robot_from_polar_north(
            world,
            start_radius=0.172,
            start_bearing_degrees_from_north=0.0,
            start_yaw_degrees_from_north=180.0,
            reasoner=mock_decide_wheel_control,
        )
        near_readings = near_robot._read_sensors_in_world()  # noqa: SLF001

        far_front = max(far_readings["R01sFL"], far_readings["R01sFR"])
        near_front = max(near_readings["R01sFL"], near_readings["R01sFR"])

        assert far_front == 0.0
        assert near_front == 40.0
    finally:
        real_pybullet.disconnect(client_id)


def _add_beam_obstacle(client_id: int, center_x: float, center_y: float, center_z: float) -> int:
    collision = real_pybullet.createCollisionShape(
        real_pybullet.GEOM_CYLINDER,
        radius=0.005,
        height=0.20,
        physicsClientId=client_id,
    )
    visual = real_pybullet.createVisualShape(
        real_pybullet.GEOM_CYLINDER,
        radius=0.005,
        length=0.20,
        rgbaColor=(0.85, 0.2, 0.2, 1.0),
        physicsClientId=client_id,
    )
    return int(
        real_pybullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=(center_x, center_y, center_z),
            physicsClientId=client_id,
        )
    )


def _point_on_sensor_ray(robot: PyBulletSide6Bottom2Program, sensor_name: str, fraction: float) -> tuple[float, float, float]:
    base_position, base_orientation = real_pybullet.getBasePositionAndOrientation(robot.body_id, robot.client_id)
    ray = next(ray for ray in robot._iter_sensor_rays() if ray.sensor_name == sensor_name)  # noqa: SLF001
    start = real_pybullet.multiplyTransforms(base_position, base_orientation, tuple(ray.start), (0.0, 0.0, 0.0, 1.0))[0]
    end = real_pybullet.multiplyTransforms(base_position, base_orientation, tuple(ray.end), (0.0, 0.0, 0.0, 1.0))[0]
    return tuple(start[i] + (end[i] - start[i]) * fraction for i in range(3))


def test_sensor_values_show_right_side_obstacle_pattern_with_real_world():
    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        world = create_circular_world(
            client_id,
            mission_module.WorldConfigCircular(floor_radius=1.5),
            [],
            time_step=1.0 / 60.0,
        )
        robot = add_robot_from_polar_north(
            world,
            start_radius=0.20,
            start_bearing_degrees_from_north=0.0,
            start_yaw_degrees_from_north=90.0,
            reasoner=mock_decide_wheel_control,
        )

        rf_x, rf_y, rf_z = _point_on_sensor_ray(robot, "R01sRF", 0.009 / 0.401)
        rs_x, rs_y, rs_z = _point_on_sensor_ray(robot, "R01sRS", 0.009 / 0.401)
        _add_beam_obstacle(client_id, rf_x, rf_y, rf_z)
        _add_beam_obstacle(client_id, rs_x, rs_y, rs_z)

        readings = robot._read_sensors_in_world()  # noqa: SLF001

        assert readings["R01sLS"] == 0.0
        assert readings["R01sLF"] == 0.0
        assert readings["R01sRF"] == 40.0
        assert readings["R01sRS"] == 40.0
    finally:
        real_pybullet.disconnect(client_id)


def test_sensor_values_show_left_side_obstacle_pattern_with_real_world():
    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        world = create_circular_world(
            client_id,
            mission_module.WorldConfigCircular(floor_radius=1.5),
            [],
            time_step=1.0 / 60.0,
        )
        robot = add_robot_from_polar_north(
            world,
            start_radius=0.20,
            start_bearing_degrees_from_north=0.0,
            start_yaw_degrees_from_north=90.0,
            reasoner=mock_decide_wheel_control,
        )

        ls_x, ls_y, ls_z = _point_on_sensor_ray(robot, "R01sLS", 0.009 / 0.401)
        lf_x, lf_y, lf_z = _point_on_sensor_ray(robot, "R01sLF", 0.009 / 0.401)
        _add_beam_obstacle(client_id, ls_x, ls_y, ls_z)
        _add_beam_obstacle(client_id, lf_x, lf_y, lf_z)

        readings = robot._read_sensors_in_world()  # noqa: SLF001

        assert readings["R01sLS"] == 40.0
        assert readings["R01sLF"] == 40.0
        assert readings["R01sRF"] == 0.0
        assert readings["R01sRS"] == 0.0
    finally:
        real_pybullet.disconnect(client_id)


def test_bottom_sensors_signal_a_painted_path_during_forward_motion():
    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        world = create_circular_world(
            client_id,
            mission_module.WorldConfigCircular(floor_radius=1.2),
            [],
            painted_path=PaintedPathConfig(radius_x=0.18, radius_y=0.18, width=0.12, segment_count=24),
            time_step=1.0 / 60.0,
        )
        robot = add_robot_from_polar_north(
            world,
            start_radius=0.18,
            start_bearing_degrees_from_north=0.0,
            start_yaw_degrees_from_north=0.0,
            reasoner=mock_decide_wheel_control,
            max_wheel_velocity=8.0,
            max_motor_force=5.0,
        )

        detected_step = None
        for step_index in range(180):
            robot.set_simulation_stage(driver_module.MissionStage(time_seconds=1.0, direction="forward", speed=1.0))
            real_pybullet.stepSimulation(physicsClientId=robot.client_id)

            bottom_readings = robot.get_bottom_sensor_readings()
            if any(value > 0.0 for value in bottom_readings.values()):
                detected_step = step_index
                break

        assert detected_step is not None
        assert min(robot.get_bottom_sensor_readings().values()) >= 0.0
    finally:
        real_pybullet.disconnect(client_id)


def test_bottom_sensors_ignore_floor_when_path_is_far_from_robot():
    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        world = create_circular_world(
            client_id,
            mission_module.WorldConfigCircular(floor_radius=1.5),
            [],
            painted_path=PaintedPathConfig(radius_x=1.0, radius_y=1.0, width=0.05, segment_count=72),
            time_step=1.0 / 60.0,
        )
        robot = add_robot_from_polar_north(
            world,
            start_radius=0.55,
            start_bearing_degrees_from_north=15.0,
            start_yaw_degrees_from_north=0.0,
            reasoner=mock_decide_wheel_control,
            max_wheel_velocity=8.0,
            max_motor_force=5.0,
        )

        bottom_readings = robot.get_bottom_sensor_readings()

        assert bottom_readings == {"R01sBL": 0.0, "R01sBR": 0.0}
    finally:
        real_pybullet.disconnect(client_id)
