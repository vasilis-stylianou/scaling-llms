from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    """Run a single command for smoke checks and exit on failure.

    Executes the provided command list, printing output and aborting when
    the command returns a non-zero exit code.
    """

    print(f"[smoke] running: {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)


def main() -> None:
    """Run a small set of smoke checks exercising the package CLI.

    Runs CLI help and dry-run flows to verify basic package entrypoints work.
    """

    repo_root = Path(__file__).resolve().parents[2]
    pod_spec = repo_root / "src" / "runpod_orchestrator" / "pod_spec.yaml"
    bootstrap = repo_root / "src" / "runpod_orchestrator" / "bootstrap.sh"

    commands = [
        [sys.executable, "-m", "runpod_orchestrator.cli", "--help"],
        [sys.executable, "-m", "runpod_orchestrator.cli", "create", "--help"],
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
        ],
        [
            sys.executable,
            "-m",
            "runpod_orchestrator.launcher",
            "--pod-spec",
            str(pod_spec),
            "--bootstrap-script",
            str(bootstrap),
            "--dry-run",
        ],
    ]

    for command in commands:
        _run(command)

    print("[smoke] all runpod_orchestrator smoke checks passed")


if __name__ == "__main__":
    main()
