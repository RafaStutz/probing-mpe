from __future__ import annotations

import tomllib
from enum import Enum
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
PYPROJECT_PATH: Path = PROJECT_ROOT / "pyproject.toml"
RUNTIME_DEPENDENCIES: set[str] = {"numpy", "PyYAML", "python-dotenv", "wandb"}
DEV_DEPENDENCIES: set[str] = {"pytest"}
DEPENDENCY_NAME_MAX_SPLIT = 1


class PyprojectKey(str, Enum):
    project = "project"
    dependencies = "dependencies"
    optional_dependencies = "optional-dependencies"
    dev = "dev"


def test_pyproject_declares_runtime_dependencies() -> None:
    metadata = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = metadata[PyprojectKey.project.value]
    declared_dependencies = {
        dependency.split(" ", maxsplit=DEPENDENCY_NAME_MAX_SPLIT)[0]
        for dependency in project[PyprojectKey.dependencies.value]
    }

    assert RUNTIME_DEPENDENCIES <= declared_dependencies


def test_pyproject_declares_dev_test_extra() -> None:
    metadata = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    optional_dependencies = metadata[PyprojectKey.project.value][
        PyprojectKey.optional_dependencies.value
    ]
    declared_dev_dependencies = {
        dependency.split(" ", maxsplit=DEPENDENCY_NAME_MAX_SPLIT)[0]
        for dependency in optional_dependencies[PyprojectKey.dev.value]
    }

    assert DEV_DEPENDENCIES <= declared_dev_dependencies
