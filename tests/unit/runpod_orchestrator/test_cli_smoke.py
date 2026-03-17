from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True)


def _paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    pod_spec = repo_root / "src" / "runpod_orchestrator" / "pod_spec.yaml"
    bootstrap = repo_root / "src" / "runpod_orchestrator" / "bootstrap.sh"
    return pod_spec, bootstrap


def test_cli_help() -> None:
    result = _run([sys.executable, "-m", "runpod_orchestrator.cli", "--help"])
    assert result.returncode == 0, result.stderr
    assert "RunPod utility CLI" in result.stdout


def test_cli_create_dry_run() -> None:
    pod_spec, bootstrap = _paths()
    result = _run(
        [
            sys.executable,
            "-m",
            "runpod_orchestrator.cli",
            "create",
            "--pod-spec",
            str(pod_spec),
            "--bootstrap-script",
            str(bootstrap),
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "==== DRY RUN ====" in result.stdout


def test_launcher_dry_run() -> None:
    pod_spec, bootstrap = _paths()
    result = _run(
        [
            sys.executable,
            "-m",
            "runpod_orchestrator.launcher",
            "--pod-spec",
            str(pod_spec),
            "--bootstrap-script",
            str(bootstrap),
            "--dry-run",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "==== DRY RUN ====" in result.stdout
