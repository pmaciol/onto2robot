from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from math import ceil
from pathlib import Path
from typing import Protocol, TextIO

import pybullet

from robots_drivers.pybullet_side6_bottom2_program import MissionStage, Vec3
from robots_drivers.worlds.pybullet_worlds import (
    CircularWorld,
    CylindricalObstacle,
    WorldConfig,
    add_robot_from_polar_north,
    create_circular_world,
)


class MissionRobotProtocol(Protocol):
    @property
    def body_id(self) -> int: ...

    @property
    def client_id(self) -> int: ...

    def set_simulation_step_state(self, stage: MissionStage) -> None: ...

    def get_position(self) -> Vec3: ...


def list_touching_world_obstacles(
    robot: MissionRobotProtocol,
    world: CircularWorld,
    *,
    pybullet_module=pybullet,
) -> list[str]:
    touching_obstacles: list[str] = []

    for obstacle_index, obstacle_id in enumerate(world.obstacle_ids, start=1):
        contacts = pybullet_module.getContactPoints(robot.body_id, obstacle_id, physicsClientId=robot.client_id)
        if contacts:
            touching_obstacles.append(f"obstacle[{obstacle_index}]#{obstacle_id}")

    for wall_segment_index, wall_id in enumerate(world.wall_ids, start=1):
        contacts = pybullet_module.getContactPoints(robot.body_id, wall_id, physicsClientId=robot.client_id)
        if contacts:
            touching_obstacles.append(f"boundary[{wall_segment_index}]#{wall_id}")

    return touching_obstacles


def _parse_stage(raw_stage: dict, stage_index: int) -> MissionStage:
    if not isinstance(raw_stage, dict):
        raise ValueError(f"Stage {stage_index} must be a JSON object.")

    missing_keys = {"time", "direction", "speed"} - set(raw_stage)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Stage {stage_index} is missing keys: {missing}.")

    try:
        stage_time = float(raw_stage["time"])
        stage_direction = str(raw_stage["direction"])
        stage_speed = float(raw_stage["speed"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Stage {stage_index} has invalid field types.") from error

    if stage_time <= 0.0:
        raise ValueError(f"Stage {stage_index} time must be greater than 0.")
    if stage_speed < 0.0:
        raise ValueError(f"Stage {stage_index} speed must be non-negative.")

    return MissionStage(time_seconds=stage_time, direction=stage_direction, speed=stage_speed)


def load_mission_stages(path: Path) -> list[MissionStage]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_stages = raw.get("stages") if isinstance(raw, dict) else raw

    if not isinstance(raw_stages, list):
        raise ValueError("Mission file must be a list of stages or an object with a 'stages' list.")

    stages = [_parse_stage(stage, index) for index, stage in enumerate(raw_stages, start=1)]
    if not stages:
        raise ValueError("Mission file must include at least one stage.")
    return stages


def run_mission(
    robot: MissionRobotProtocol,
    stages: list[MissionStage],
    step_time: float,
    position_logger,
    world: CircularWorld | None = None,
    touch_logger: Callable[[float, list[str]], None] | None = None,
    pybullet_module=pybullet,
) -> int:
    if step_time <= 0.0:
        raise ValueError("step_time must be greater than 0.")

    elapsed_time = 0.0
    executed_steps = 0

    for stage in stages:
        stage_steps = max(1, ceil(stage.time_seconds / step_time))

        for _ in range(stage_steps):
            robot.set_simulation_step_state(stage)
            pybullet_module.stepSimulation(physicsClientId=robot.client_id)
            executed_steps += 1
            elapsed_time += step_time
            position_logger(elapsed_time, robot.get_position())
            if world is not None and touch_logger is not None:
                touching_obstacles = list_touching_world_obstacles(robot, world, pybullet_module=pybullet_module)
                if touching_obstacles:
                    touch_logger(elapsed_time, touching_obstacles)

    return executed_steps


def _format_position_line(elapsed_time: float, position: Vec3) -> str:
    return f"{elapsed_time:.3f};{position.x:.4f};{position.y:.4f};{position.z:.4f}"


def _build_position_logger(output_stream: TextIO | None):
    if output_stream is None:

        def log_to_stdout(elapsed_time: float, position: Vec3) -> None:
            print(_format_position_line(elapsed_time, position))

        return log_to_stdout

    output_stream.write("time;x;y;z\n")

    def log_to_stream(elapsed_time: float, position: Vec3) -> None:
        output_stream.write(_format_position_line(elapsed_time, position) + "\n")

    return log_to_stream


def _build_video_recorder(
    output_path: Path,
    physics_client_id: int,
    step_time: float,
    pybullet_module=pybullet,
) -> tuple[Callable[[], None], Callable[[], None]]:
    import imageio.v2 as imageio
    import numpy as np

    if step_time <= 0.0:
        raise ValueError("step_time must be greater than 0.")

    width = 640
    height = 480
    fps = max(1, round(1.0 / step_time))

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

    def record_frame() -> None:
        _, _, rgba, _, _ = pybullet_module.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=pybullet_module.ER_TINY_RENDERER,
            physicsClientId=physics_client_id,
        )
        frame = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        writer.append_data(frame)

    def close_recorder() -> None:
        writer.close()

    return record_frame, close_recorder


def load_obstacles(path: Path | None) -> list[CylindricalObstacle]:
    if path is None:
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("World obstacles file must contain a list.")

    obstacles: list[CylindricalObstacle] = []
    for obstacle_index, obstacle in enumerate(raw, start=1):
        if not isinstance(obstacle, dict):
            raise ValueError(f"Obstacle {obstacle_index} must be a JSON object.")

        missing_keys = {"radius", "height", "center"} - set(obstacle)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Obstacle {obstacle_index} is missing keys: {missing}.")

        center = obstacle["center"]
        if not isinstance(center, list | tuple) or len(center) != 2:
            raise ValueError(f"Obstacle {obstacle_index} center must be [x, y].")

        try:
            radius = float(obstacle["radius"])
            height = float(obstacle["height"])
            center_x = float(center[0])
            center_y = float(center[1])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Obstacle {obstacle_index} has invalid numeric values.") from error

        obstacles.append(
            CylindricalObstacle(
                radius=radius,
                height=height,
                center_x=center_x,
                center_y=center_y,
            )
        )

    return obstacles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pybullet-mission", description="Run staged robot mission in PyBullet")
    parser.add_argument("--mission", type=Path, required=True, help="Path to mission JSON file")
    parser.add_argument("--visualize", action="store_true", help="Run simulation with PyBullet GUI")
    parser.add_argument("--output", type=Path, help="Optional output file for position logs")
    parser.add_argument("--video", type=Path, help="Optional output movie file (for example mission.mp4)")
    parser.add_argument("--time-step", type=float, default=1.0 / 60.0, help="Simulation time step in seconds")
    parser.add_argument("--max-wheel-velocity", type=float, default=6.0, help="Maximum wheel angular velocity")
    parser.add_argument("--max-motor-force", type=float, default=2.5, help="Maximum motor force")
    parser.add_argument("--floor-radius", type=float, default=1.5, help="Circular floor radius")
    parser.add_argument("--floor-height", type=float, default=0.01, help="Circular floor height")
    parser.add_argument("--wall-height", type=float, default=0.10, help="Boundary wall height")
    parser.add_argument("--wall-thickness", type=float, default=0.03, help="Boundary wall thickness")
    parser.add_argument("--wall-segments", type=int, default=48, help="Boundary wall segment count")
    parser.add_argument("--world-obstacles", type=Path, help="Path to world obstacles JSON file")
    parser.add_argument("--robot-start-radius", type=float, default=0.0, help="Robot start radius from world center")
    parser.add_argument(
        "--robot-start-bearing",
        type=float,
        default=0.0,
        help="Robot start bearing in degrees from North (clockwise positive)",
    )
    parser.add_argument(
        "--robot-start-yaw",
        type=float,
        default=0.0,
        help="Robot heading in degrees from North (clockwise positive)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stages = load_mission_stages(args.mission)
    obstacles = load_obstacles(args.world_obstacles)
    connection_mode = pybullet.GUI if args.visualize else pybullet.DIRECT
    client_id = pybullet.connect(connection_mode)
    close_video_recorder: Callable[[], None] | None = None

    try:
        world_config = WorldConfig(
            floor_radius=args.floor_radius,
            floor_height=args.floor_height,
            wall_height=args.wall_height,
            wall_thickness=args.wall_thickness,
            wall_segments=args.wall_segments,
        )
        world = create_circular_world(
            physics_client_id=client_id,
            config=world_config,
            obstacles=obstacles,
            time_step=args.time_step,
        )
        robot = add_robot_from_polar_north(
            world=world,
            physics_client_id=client_id,
            start_radius=args.robot_start_radius,
            start_bearing_degrees_from_north=args.robot_start_bearing,
            start_yaw_degrees_from_north=args.robot_start_yaw,
            max_wheel_velocity=args.max_wheel_velocity,
            max_motor_force=args.max_motor_force,
        )

        video_recorder: Callable[[], None] | None = None
        if args.video:
            video_recorder, close_video_recorder = _build_video_recorder(args.video, client_id, args.time_step)

        def touch_logger(elapsed_time: float, touching_obstacles: list[str]) -> None:
            joined_obstacles = ", ".join(touching_obstacles)
            print(f"touch@{elapsed_time:.3f}: {joined_obstacles}", file=sys.stderr)

        if args.output:
            with args.output.open("w", encoding="utf-8") as output_file:
                base_logger = _build_position_logger(output_file)

                def logger(elapsed_time: float, position: Vec3) -> None:
                    base_logger(elapsed_time, position)
                    if video_recorder is not None:
                        video_recorder()

                run_mission(robot, stages, args.time_step, logger, world=world, touch_logger=touch_logger)
        else:
            print("time;x;y;z")
            base_logger = _build_position_logger(None)

            def logger(elapsed_time: float, position: Vec3) -> None:
                base_logger(elapsed_time, position)
                if video_recorder is not None:
                    video_recorder()

            run_mission(robot, stages, args.time_step, logger, world=world, touch_logger=touch_logger)

        return 0
    finally:
        if close_video_recorder is not None:
            close_video_recorder()
        pybullet.disconnect(client_id)


if __name__ == "__main__":
    raise SystemExit(main())
