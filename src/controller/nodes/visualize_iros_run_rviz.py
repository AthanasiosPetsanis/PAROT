#!/usr/bin/env python3
"""ROS package entry point for testing/visualize_iros_run_rviz.py."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "testing" / "visualize_iros_run_rviz.py"
    if not script.is_file():
        raise FileNotFoundError(f"Could not find replay script: {script}")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
