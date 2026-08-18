from pathlib import Path

import pybullet as real_pybullet
import pytest

import robots_drivers.pybullet_mission_runner as mission_module
from robots_drivers.controlers.wheel_controler import mock_decide_wheel_control
from robots_drivers.pybullet_side6_bottom2_program import Vec3
from robots_drivers.worlds.pybullet_worlds import CircularWorld, add_robot_from_polar_north, create_circular_world


class FakePyBullet:
    VELOCITY_CONTROL = 3

    def __init__(self, on_step):
        self.on_step = on_step
        self.motor_calls = []
        self.step_calls = 0

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
        self.on_step()


class FakeRobot:
    def __init__(self):
        self.body_id = 42
        self.max_wheel_velocity = 2.0
        self.max_motor_force = 1.5
        self._step = 0
        self.motor_calls = []

    @property
    def client_id(self):
        return 17

    def _apply_wheel_command(self, command):
        self.motor_calls.append(
            {
                "left": command.left * self.max_wheel_velocity,
                "right": command.right * self.max_wheel_velocity,
            }
        )

    def advance(self):
        self._step += 1

    def set_simulation_stage(self, stage):
        direction = stage.direction.strip().lower()
        if direction == "forward":
            left = stage.speed
            right = stage.speed
        elif direction == "backward":
            left = -stage.speed
            right = -stage.speed
        elif direction == "left":
            left = -stage.speed
            right = stage.speed
        elif direction == "right":
            left = stage.speed
            right = -stage.speed
        else:
            raise ValueError(f"Unsupported direction '{stage.direction}'.")

        self.motor_calls.append(
            {
                "left": left * self.max_wheel_velocity,
                "right": right * self.max_wheel_velocity,
            }
        )

    def get_position(self):
        return Vec3([float(self._step), 0.0, 0.04])

    def get_rotation(self):
        # Yaw=0 in PyBullet means +X; this should map to 90 deg from world north.
        return (0.0, 0.0, 0.0, 1.0)


def test_two_stage_mission_file_is_loaded_and_executed():
    mission_path = Path(__file__).parent / "data" / "two_stage_mission.json"
    stages = mission_module.load_mission_stages(mission_path)

    assert len(stages) == 2
    assert stages[0].direction == "forward"
    assert stages[0].time_seconds == 20.0
    assert stages[1].direction == "backward"
    assert stages[1].speed == 0.25

    robot = FakeRobot()
    fake_pybullet = FakePyBullet(on_step=robot.advance)
    logged_positions = []

    executed_steps = mission_module.run_mission(
        robot=robot,  # pyright: ignore[reportArgumentType]; FakeRobot keeps this test focused on step counting and logging contract
        stages=stages,
        step_time=0.5,
        position_logger=lambda elapsed, position, heading: logged_positions.append((elapsed, position, heading)),
        pybullet_module=fake_pybullet,  # pyright: ignore[reportArgumentType]; fake_pybullet replace a real pybullet module for better control of the test
    )

    assert executed_steps == 90
    assert fake_pybullet.step_calls == 90
    assert logged_positions[0][2] == pytest.approx(90.0)


def test_load_obstacles_parses_json_file(tmp_path):
    obstacles_file = tmp_path / "obstacles.json"
    obstacles_file.write_text('[{"radius": 0.1, "height": 0.2, "center": [0.3, -0.4]}]', encoding="utf-8")

    obstacles = mission_module.load_obstacles(obstacles_file)

    assert len(obstacles) == 1
    assert obstacles[0].radius == 0.1
    assert obstacles[0].height == 0.2
    assert obstacles[0].center_x == 0.3
    assert obstacles[0].center_y == -0.4


def test_list_touching_world_obstacles_includes_loaded_obstacles_and_boundary():
    class FakeContactPyBullet:
        def getContactPoints(self, body_a, body_b, physicsClientId=None):
            del body_a, physicsClientId
            touching_body_ids = {301, 402}
            return [object()] if body_b in touching_body_ids else []

    robot = FakeRobot()
    world = CircularWorld(
        floor_id=99,
        wall_ids=(401, 402, 403),
        obstacle_ids=(301, 302),
        floor_top_z=0.0,
        config=mission_module.WorldConfigCircular(floor_radius=1.5),
        physics_client_id=robot.client_id,
    )

    touching = mission_module.list_touching_world_obstacles(robot, world, pybullet_module=FakeContactPyBullet())  # pyright: ignore[reportArgumentType]; Fake used to eliminate real pybullet dependency

    assert touching == ["obstacle[1]#301", "boundary[2]#402"]


def test_forward_mission_contact_with_central_obstacle_arises_during_simulation():
    mission_path = Path(__file__).parent / "data" / "test_forward.json"
    obstacles_path = Path(__file__).parent / "data" / "sample_world_central_obstacle.json"

    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        step_time = 1.0 / 60.0
        world = create_circular_world(
            client_id,
            mission_module.WorldConfigCircular(floor_radius=1.5),
            mission_module.load_obstacles(obstacles_path),
            time_step=step_time,
        )
        robot = add_robot_from_polar_north(
            world,
            start_radius=0.55,
            start_bearing_degrees_from_north=15.0,
            reasoner=mock_decide_wheel_control,
        )
        stages = mission_module.load_mission_stages(mission_path)

        assert mission_module.list_touching_world_obstacles(robot, world, pybullet_module=real_pybullet) == []

        first_touch_time = None
        first_touching = None
        elapsed_time = 0.0

        for stage in stages:
            stage_steps = max(1, mission_module.ceil(stage.time_seconds / step_time))
            for _ in range(stage_steps):
                robot.set_simulation_stage(stage)
                real_pybullet.stepSimulation(physicsClientId=robot.client_id)
                elapsed_time += step_time
                touching = mission_module.list_touching_world_obstacles(robot, world, pybullet_module=real_pybullet)
                if touching:
                    first_touch_time = elapsed_time
                    first_touching = touching
                    break
            if first_touch_time is not None:
                break

        assert first_touch_time is not None
        assert first_touch_time > 0.1
        assert first_touch_time < stages[0].time_seconds
        assert first_touching == [f"obstacle[1]#{world.obstacle_ids[0]}"]
    finally:
        real_pybullet.disconnect(client_id)


def test_parser_accepts_world_configuration_arguments():
    parser = mission_module.build_parser()

    args = parser.parse_args(
        [
            "--mission",
            "mission.json",
            "--floor-radius",
            "2.0",
            "--wall-height",
            "0.2",
            "--wall-thickness",
            "0.05",
            "--wall-segments",
            "64",
            "--robot-start-radius",
            "0.4",
            "--robot-start-bearing",
            "30",
            "--robot-start-yaw",
            "15",
            "--video",
            "mission.mp4",
        ]
    )

    assert args.floor_radius == 2.0
    assert args.wall_height == 0.2
    assert args.wall_thickness == 0.05
    assert args.wall_segments == 64
    assert args.robot_start_radius == 0.4
    assert args.robot_start_bearing == 30
    assert args.robot_start_yaw == 15
    assert args.video == Path("mission.mp4")


def test_parser_accepts_painted_path_arguments():
    parser = mission_module.build_parser()

    args = parser.parse_args(
        [
            "--mission",
            "mission.json",
            "--path-radius-x",
            "0.6",
            "--path-radius-y",
            "0.4",
            "--path-width",
            "0.05",
            "--path-segment-count",
            "48",
            "--path-height",
            "0.002",
        ]
    )

    assert args.path_radius_x == 0.6
    assert args.path_radius_y == 0.4
    assert args.path_width == 0.05
    assert args.path_segment_count == 48
    assert args.path_height == 0.002
