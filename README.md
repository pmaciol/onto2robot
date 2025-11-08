# onto2robot

Utilities for converting or integrating ontologies with robotic systems.

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and a `src` layout.

### Prerequisites
- Python 3.10+
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

## Project Structure
```
pyproject.toml
src/
  onto2robot/
    __init__.py
    core.py
    cli.py
tests/
  test_core.py
```

## Next Steps
- Add real ontology parsing logic in `core.py`.
- Expand CLI commands.
- Add type checking (e.g. mypy) and linting.
