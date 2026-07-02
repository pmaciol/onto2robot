from math import isclose
from pathlib import Path

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
PyBulletSide6Bottom2Program = driver_module.PyBulletSide6Bottom2Program
DifferentialWheelCommand = driver_module.DifferentialWheelCommand
mock_decide_wheel_control = driver_module.mock_decide_wheel_control


def _run_steps_with_optional_video(
    robot,
    pybullet_module,
    steps: int,
    obstacle_id: int,
    client_id: int,
    output_path: Path | None = None,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> bool:
    """Run control steps, optionally rendering and encoding frames to a video file."""

    writer = None
    imageio = None
    np = None
    try:
        if output_path is not None:
            imageio = pytest.importorskip("imageio.v2")
            np = pytest.importorskip("numpy")
            view_matrix = pybullet_module.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=(0.0, 0.0, 0.03),
                distance=1.45,
                yaw=35.0,
                pitch=-35.0,
                roll=0.0,
                upAxisIndex=2,
            )
            projection_matrix = pybullet_module.computeProjectionMatrixFOV(
                fov=60.0,
                aspect=width / height,
                nearVal=0.01,
                farVal=3.0,
            )
            writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")
        else:
            view_matrix = None
            projection_matrix = None

        for _ in range(steps):
            robot._control_step()  # noqa: SLF001
            if writer is not None and view_matrix is not None and projection_matrix is not None:
                _, _, rgba, _, _ = pybullet_module.getCameraImage(
                    width,
                    height,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    renderer=pybullet_module.ER_TINY_RENDERER,
                    physicsClientId=client_id,
                )
                frame = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
                writer.append_data(frame)

            if pybullet_module.getContactPoints(robot.body_id, obstacle_id, physicsClientId=client_id):
                return True
    finally:
        if writer is not None:
            writer.close()

    return False


def test_sensor_layout_matches_requested_geometry():
    robot = PyBulletSide6Bottom2Program()

    assert robot.side_sensor_keys == SIDE_SENSOR_KEYS
    assert robot.bottom_sensor_keys == BOTTOM_SENSOR_KEYS
    assert robot.side_sensor_angles_deg == (90.0, 54.0, 18.0, -18.0, -54.0, -90.0)
    assert robot.bottom_sensor_angles_deg == (20.0, -20.0)

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


def test_build_robot_uses_wheel_axle_joint_axis_in_rotated_link_frame(monkeypatch):
    fake_pybullet = FakePyBullet()
    monkeypatch.setattr(driver_module, "pybullet", fake_pybullet)

    PyBulletSide6Bottom2Program()

    _, kwargs = fake_pybullet.create_multibody_calls[-1]
    assert kwargs["linkJointAxis"] == [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]


def test_mock_decision_is_balanced_for_symmetric_inputs():
    command = mock_decide_wheel_control([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], [10.0, 10.0])

    assert isclose(command.left, command.right)
    assert 0.0 <= command.left <= 1.0
    assert 0.0 <= command.right <= 1.0


def test_mock_decision_turns_toward_stronger_bottom_signal():
    command = mock_decide_wheel_control([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], [5.0, 25.0])

    assert command.right > command.left


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


def test_robot_starts_centered_with_wheels_touching_floor_and_moves_forward_10cm():
    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        real_pybullet.resetSimulation(physicsClientId=client_id)
        real_pybullet.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
        time_step = 1.0 / 120.0
        real_pybullet.setTimeStep(time_step, physicsClientId=client_id)

        floor_radius = 1.0
        floor_height = 0.01
        floor_collision = real_pybullet.createCollisionShape(
            real_pybullet.GEOM_CYLINDER,
            radius=floor_radius,
            height=floor_height,
            physicsClientId=client_id,
        )
        floor_visual = real_pybullet.createVisualShape(
            real_pybullet.GEOM_CYLINDER,
            radius=floor_radius,
            length=floor_height,
            rgbaColor=(0.75, 0.75, 0.75, 1.0),
            physicsClientId=client_id,
        )
        floor_id = real_pybullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_collision,
            baseVisualShapeIndex=floor_visual,
            basePosition=(0.0, 0.0, -floor_height / 2.0),
            physicsClientId=client_id,
        )

        dimensions = driver_module.RobotDimensions()
        expected_base_height = dimensions.wheel_radius + dimensions.body_height / 2.0
        robot = PyBulletSide6Bottom2Program(
            client_id=client_id,
            base_position=driver_module.Vec3([0.0, 0.0, expected_base_height]),
        )

        for _ in range(20):
            real_pybullet.stepSimulation(physicsClientId=client_id)

        initial_position, _ = real_pybullet.getBasePositionAndOrientation(robot.body_id, physicsClientId=client_id)
        assert isclose(initial_position[0], 0.0, abs_tol=1e-3)
        assert isclose(initial_position[1], 0.0, abs_tol=1e-3)
        assert expected_base_height - 0.015 <= initial_position[2] <= expected_base_height + 0.002

        contacts = real_pybullet.getContactPoints(robot.body_id, floor_id, physicsClientId=client_id)
        assert contacts
        wheel_links_in_contact = {contact[3] for contact in contacts}
        assert 0 in wheel_links_in_contact
        assert 1 in wheel_links_in_contact

        imageio = pytest.importorskip("imageio.v2")
        np = pytest.importorskip("numpy")
        view_matrix = real_pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.0, 0.0, 0.03),
            distance=1.45,
            yaw=35.0,
            pitch=-35.0,
            roll=0.0,
            upAxisIndex=2,
        )
        projection_matrix = real_pybullet.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=640 / 480,
            nearVal=0.01,
            farVal=3.0,
        )
        writer = imageio.get_writer("test_forward_10cm_video.mp4", fps=30, codec="libx264")

        moved_position = initial_position
        target_distance = 0.10
        try:
            for _ in range(2000):
                real_pybullet.resetBaseVelocity(
                    robot.body_id,
                    linearVelocity=(0.2, 0.0, 0.0),
                    angularVelocity=(0.0, 0.0, 0.0),
                    physicsClientId=client_id,
                )
                real_pybullet.stepSimulation(physicsClientId=client_id)
                moved_position, _ = real_pybullet.getBasePositionAndOrientation(robot.body_id, physicsClientId=client_id)

                _, _, rgba, _, _ = real_pybullet.getCameraImage(
                    640,
                    480,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    renderer=real_pybullet.ER_TINY_RENDERER,
                    physicsClientId=client_id,
                )
                frame = np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
                writer.append_data(frame)

                if moved_position[0] - initial_position[0] >= target_distance:
                    break
        finally:
            writer.close()

        traveled_x = moved_position[0] - initial_position[0]
        assert traveled_x >= target_distance
        assert traveled_x <= target_distance + 0.01

        assert abs(moved_position[1] - initial_position[1]) < 0.02
        assert isclose(moved_position[2], expected_base_height, abs_tol=0.01)
    finally:
        real_pybullet.disconnect(client_id)


def test_robot_avoids_obstacle_in_real_pybullet_world():
    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        real_pybullet.resetSimulation(physicsClientId=client_id)
        real_pybullet.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
        time_step = 1.0 / 120.0
        real_pybullet.setTimeStep(time_step, physicsClientId=client_id)

        floor_collision = real_pybullet.createCollisionShape(
            real_pybullet.GEOM_CYLINDER,
            radius=1.5,
            height=0.01,
            physicsClientId=client_id,
        )
        floor_visual = real_pybullet.createVisualShape(
            real_pybullet.GEOM_CYLINDER,
            radius=1.5,
            length=0.01,
            rgbaColor=(0.75, 0.75, 0.75, 1.0),
            physicsClientId=client_id,
        )
        real_pybullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_collision,
            baseVisualShapeIndex=floor_visual,
            basePosition=(0.0, 0.0, -0.005),
            physicsClientId=client_id,
        )

        obstacle_collision = real_pybullet.createCollisionShape(
            real_pybullet.GEOM_CYLINDER,
            radius=0.10,
            height=0.10,
            physicsClientId=client_id,
        )
        obstacle_visual = real_pybullet.createVisualShape(
            real_pybullet.GEOM_CYLINDER,
            radius=0.10,
            length=0.10,
            rgbaColor=(0.85, 0.2, 0.2, 1.0),
            physicsClientId=client_id,
        )
        obstacle_id = real_pybullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=obstacle_collision,
            baseVisualShapeIndex=obstacle_visual,
            basePosition=(0.20, 0.0, 0.05),
            physicsClientId=client_id,
        )

        def simple_avoidance_controller(side_sensors_left_to_right, bottom_sensors_left_to_right):
            del bottom_sensors_left_to_right
            front_left = side_sensors_left_to_right[2]
            front_right = side_sensors_left_to_right[3]
            if min(front_left, front_right) < 0.1:
                if front_left <= front_right:
                    print(f"Turning right: front_left={front_left:.1f} front_right={front_right:.1f}")
                    return DifferentialWheelCommand(left=1.0, right=0.2)
                print(f"Turning left: front_left={front_left:.1f} front_right={front_right:.1f}")
                return DifferentialWheelCommand(left=0.2, right=1.0)
            print(f"Moving forward: front_left={front_left:.1f} front_right={front_right:.1f}")
            return DifferentialWheelCommand(left=1.0, right=1.0)

        # 1 cm/s linear speed at wheel radius 2 cm -> angular speed 0.5 rad/s.
        max_wheel_velocity = 0.5
        max_motor_force = 3.0
        robot = PyBulletSide6Bottom2Program(
            wheel_controller=simple_avoidance_controller,
            max_wheel_velocity=max_wheel_velocity,
            max_motor_force=max_motor_force,
            client_id=client_id,
            base_position=driver_module.Vec3([-0.1, -0.3, 0.04]),
        )

        collided = _run_steps_with_optional_video(
            robot=robot,
            pybullet_module=real_pybullet,
            steps=200,
            obstacle_id=obstacle_id,
            client_id=client_id,
            output_path="test_avoidance_video.mp4",
        )

        robot_position, _ = real_pybullet.getBasePositionAndOrientation(robot.body_id, physicsClientId=client_id)
        assert not collided
        assert abs(robot_position[1]) > 0.01
    finally:
        real_pybullet.disconnect(client_id)
