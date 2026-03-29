from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
_ROBOTS_DIR = _ROOT / "robots"

AVAILABLE_ROBOTS = sorted(
    d.name for d in _ROBOTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
)


def _robot_dir(robot: str) -> _Path:
    path = _ROBOTS_DIR / robot
    if not path.is_dir():
        raise ValueError(
            f"Unknown robot {robot!r}. Available robots: {AVAILABLE_ROBOTS}"
        )
    return path


def get_scene_path(robot: str) -> _Path:
    """Return the path to the robot's scene.xml file."""
    path = _robot_dir(robot) / "scene.xml"
    if not path.exists():
        raise FileNotFoundError(f"scene.xml not found for robot {robot!r}")
    return path


def get_config_path(robot: str) -> _Path:
    """Return the path to the robot's config.yaml file."""
    path = _robot_dir(robot) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config.yaml not found for robot {robot!r}")
    return path


def get_mjcf_path(robot: str) -> _Path:
    """Return the path to the robot's MJCF model file (the .xml that is not scene.xml)."""
    robot_dir = _robot_dir(robot)
    mjcf_files = [
        f for f in robot_dir.glob("*.xml") if f.name != "scene.xml"
    ]
    if len(mjcf_files) == 0:
        raise FileNotFoundError(f"No MJCF file found for robot {robot!r}")
    if len(mjcf_files) > 1:
        raise RuntimeError(
            f"Multiple MJCF files found for robot {robot!r}: {[f.name for f in mjcf_files]}"
        )
    return mjcf_files[0]
