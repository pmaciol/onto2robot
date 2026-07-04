from pathlib import Path

import pybullet as real_pybullet

import robots_drivers.pybullet_mission_runner as mission_module
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
        self._client_id = 17
        self.max_wheel_velocity = 2.0
        self.max_motor_force = 1.5
        self._step = 0
        self.motor_calls = []

    @property
    def client_id(self):
        return self._client_id

    def _apply_wheel_command(self, command):
        self.motor_calls.append(
            {
                "left": command.left * self.max_wheel_velocity,
                "right": command.right * self.max_wheel_velocity,
            }
        )

    def advance(self):
        self._step += 1

    def set_simulation_step_state(self, stage):
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
        robot=robot,
        stages=stages,
        step_time=0.5,
        position_logger=lambda elapsed, position: logged_positions.append((elapsed, position)),
        pybullet_module=fake_pybullet,
    )

    assert executed_steps == 90
    assert fake_pybullet.step_calls == 90
    assert len(robot.motor_calls) == 90
    assert len(logged_positions) == 90

    assert logged_positions[0][0] == 0.5
    assert logged_positions[-1][0] == 45.0
    assert logged_positions[-1][1] == Vec3([90.0, 0.0, 0.04])

    assert robot.motor_calls[0]["left"] == 1.0
    assert robot.motor_calls[0]["right"] == 1.0
    assert robot.motor_calls[-1]["left"] == -0.5
    assert robot.motor_calls[-1]["right"] == -0.5


def test_load_obstacles_parses_json_file(tmp_path):
    obstacles_file = tmp_path / "obstacles.json"
    obstacles_file.write_text('[{"radius": 0.1, "height": 0.2, "center": [0.3, -0.4]}]', encoding="utf-8")

    obstacles = mission_module.load_obstacles(obstacles_file)  # noqa: SLF001

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
        config=mission_module.WorldConfig(floor_radius=1.5),
    )

    touching = mission_module.list_touching_world_obstacles(robot, world, pybullet_module=FakeContactPyBullet())

    assert touching == ["obstacle[1]#301", "boundary[2]#402"]


def test_forward_mission_contact_with_central_obstacle_arises_during_simulation():
    mission_path = Path(__file__).parent / "data" / "test_forward.json"
    obstacles_path = Path(__file__).parent / "data" / "sample_world_central_obstacle.json"

    client_id = real_pybullet.connect(real_pybullet.DIRECT)
    try:
        step_time = 1.0 / 60.0
        world = create_circular_world(
            client_id,
            mission_module.WorldConfig(floor_radius=1.5),
            mission_module.load_obstacles(obstacles_path),
            time_step=step_time,
        )
        robot = add_robot_from_polar_north(
            world,
            client_id,
            start_radius=0.25,
            start_bearing_degrees_from_north=15.0,
            start_yaw_degrees_from_north=0.0,
        )
        stages = mission_module.load_mission_stages(mission_path)

        assert mission_module.list_touching_world_obstacles(robot, world, pybullet_module=real_pybullet) == []

        first_touch_time = None
        first_touching = None
        elapsed_time = 0.0

        for stage in stages:
            stage_steps = max(1, mission_module.ceil(stage.time_seconds / step_time))
            for _ in range(stage_steps):
                robot.set_simulation_step_state(stage)
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


def test_main_wires_cli_world_config_to_world_and_robot(tmp_path, monkeypatch):
    mission_path = Path(__file__).parent / "data" / "two_stage_mission.json"
    obstacles_path = Path(__file__).parent / "data" / "sample_world_obstacles.json"
    output_path = tmp_path / "positions.csv"
    video_path = tmp_path / "mission.mp4"
    captured: dict[str, object] = {}

    class FakePyBulletModule:
        DIRECT = 0
        GUI = 1

        def connect(self, mode):
            captured["connect_mode"] = mode
            return 17

        def disconnect(self, client_id):
            captured["disconnect_client_id"] = client_id

    def fake_create_circular_world(physics_client_id, config, obstacles, time_step):
        captured["world_client_id"] = physics_client_id
        captured["world_config"] = config
        captured["world_obstacles"] = obstacles
        captured["world_time_step"] = time_step
        return "world"

    def fake_add_robot_from_polar_north(
        world,
        physics_client_id,
        start_radius,
        start_bearing_degrees_from_north,
        start_yaw_degrees_from_north,
        max_wheel_velocity,
        max_motor_force,
    ):
        captured["robot_world"] = world
        captured["robot_client_id"] = physics_client_id
        captured["robot_start_radius"] = start_radius
        captured["robot_start_bearing"] = start_bearing_degrees_from_north
        captured["robot_start_yaw"] = start_yaw_degrees_from_north
        captured["robot_max_wheel_velocity"] = max_wheel_velocity
        captured["robot_max_motor_force"] = max_motor_force
        return "robot"

    def fake_run_mission(robot, stages, step_time, position_logger, world=None, touch_logger=None):
        captured["run_robot"] = robot
        captured["run_stages_count"] = len(stages)
        captured["run_step_time"] = step_time
        captured["run_world"] = world
        captured["run_has_touch_logger"] = touch_logger is not None
        position_logger(0.125, Vec3([1.0, 2.0, 3.0]))
        return 1

    def fake_build_video_recorder(output_path_arg, physics_client_id, step_time):
        captured["video_path"] = output_path_arg
        captured["video_client_id"] = physics_client_id
        captured["video_step_time"] = step_time
        captured["video_frames"] = 0
        captured["video_closed"] = False

        def record_frame():
            captured["video_frames"] = int(captured["video_frames"]) + 1

        def close_recorder():
            captured["video_closed"] = True

        return record_frame, close_recorder

    monkeypatch.setattr(mission_module, "pybullet", FakePyBulletModule())
    monkeypatch.setattr(mission_module, "create_circular_world", fake_create_circular_world)
    monkeypatch.setattr(mission_module, "add_robot_from_polar_north", fake_add_robot_from_polar_north)
    monkeypatch.setattr(mission_module, "run_mission", fake_run_mission)
    monkeypatch.setattr(mission_module, "_build_video_recorder", fake_build_video_recorder)

    exit_code = mission_module.main(
        [
            "--mission",
            str(mission_path),
            "--world-obstacles",
            str(obstacles_path),
            "--output",
            str(output_path),
            "--time-step",
            "0.125",
            "--floor-radius",
            "2.5",
            "--floor-height",
            "0.02",
            "--wall-height",
            "0.3",
            "--wall-thickness",
            "0.07",
            "--wall-segments",
            "72",
            "--robot-start-radius",
            "0.6",
            "--robot-start-bearing",
            "35",
            "--robot-start-yaw",
            "-10",
            "--max-wheel-velocity",
            "7.5",
            "--max-motor-force",
            "3.3",
            "--video",
            str(video_path),
        ]
    )

    assert exit_code == 0
    assert captured["connect_mode"] == 0
    assert captured["disconnect_client_id"] == 17

    world_config = captured["world_config"]
    assert world_config.floor_radius == 2.5
    assert world_config.floor_height == 0.02
    assert world_config.wall_height == 0.3
    assert world_config.wall_thickness == 0.07
    assert world_config.wall_segments == 72
    assert captured["world_client_id"] == 17
    assert captured["world_time_step"] == 0.125

    world_obstacles = captured["world_obstacles"]
    assert len(world_obstacles) == 2
    assert world_obstacles[0].center_x == 0.35
    assert world_obstacles[1].center_y == -0.45

    assert captured["robot_world"] == "world"
    assert captured["robot_client_id"] == 17
    assert captured["robot_start_radius"] == 0.6
    assert captured["robot_start_bearing"] == 35.0
    assert captured["robot_start_yaw"] == -10.0
    assert captured["robot_max_wheel_velocity"] == 7.5
    assert captured["robot_max_motor_force"] == 3.3

    assert captured["run_robot"] == "robot"
    assert captured["run_stages_count"] == 2
    assert captured["run_step_time"] == 0.125
    assert captured["run_world"] == "world"
    assert captured["run_has_touch_logger"] is True

    assert captured["video_path"] == video_path
    assert captured["video_client_id"] == 17
    assert captured["video_step_time"] == 0.125
    assert captured["video_frames"] == 1
    assert captured["video_closed"] is True

    output_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert output_lines[0] == "time;x;y;z"
    assert output_lines[1] == "0.125;1.0000;2.0000;3.0000"
