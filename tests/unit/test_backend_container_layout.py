"""Regression test for the backend package layout used by its Docker image."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_backend_imports_from_container_layout(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    container_root = tmp_path / "container"
    srv = container_root / "srv"
    srv.mkdir(parents=True)

    shutil.copytree(repo_root / "backend" / "app", srv / "app")
    shutil.copytree(repo_root / "video_analysis", container_root / "video_analysis")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(srv)
    completed = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=srv,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
