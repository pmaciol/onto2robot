from math import isclose

import pybullet as real_pybullet
import pytest

import robots_drivers.pybullet_side6_bottom2_program as driver_module


class FakePyBullet:
    GEOM_CYLINDER = 1
    JOINT_REVOLUTE = 2
    VELOCITY_CONTROL = 3

    def __init__(self):
        self.motor_calls = []
        self.ray_batches = []
        self.step_calls = 0
        self.reset_base_calls = []
        self.reset_joint_calls = []
        self.create_multibody_calls = []

    def createCollisionShape(self, *args, **kwargs):
        return 11

    def createVisualShape(self, *args, **kwargs):
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
    assert readings[SIDE_SENSOR_KEYS[0]] == 10.0
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
