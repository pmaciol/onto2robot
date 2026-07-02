# onto2robot

The system for generating resoning engines based on ontologies


### Prerequisites
- Python 3.12+
- uv installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Install Dependencies
```bash
uv sync --dev
```

### Run Tests
```bash
uv run pytest
```

### CLI Usage
After syncing, run the stub CLI:
```bash
uv run onto2robot --help
```

### PyBullet Mission Runner Example
Run the mission runner with the sample circular world obstacle configuration:
```bash
uv run python -m robots_drivers.pybullet_mission_runner \
	--mission ./tests/test_drivers/data/two_stage_mission.json \
	--world-obstacles ./tests/test_drivers/data/sample_world_obstacles.json \
	--floor-radius 1.5 \
	--robot-start-radius 0.25 \
	--robot-start-bearing 15 \
	--robot-start-yaw 0 \
	--visualize
```
