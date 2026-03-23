from __future__ import annotations

import argparse
from pathlib import Path

from runpod_orchestrator.config import OrchestratorConfig
from runpod_orchestrator.orchestrator import OrchestratorService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RunPod orchestrator from a YAML config.",
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to orchestrator YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config_path.expanduser().resolve()

    config = OrchestratorConfig.from_yaml(config_path)
    service = OrchestratorService()
    result = service.run(config)

    print(f"pod_id={result.pod.pod_id}")
    print(f"ssh={result.pod.ssh_command(config.pod_spec.identity_file)}")
    print(f"tmux_session={result.tmux_session_name}")
    print(f"log_path={result.log_path}")


if __name__ == "__main__":
    main()