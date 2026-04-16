from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

from runpod_orchestrator.exceptions import ConfigError
from runpod_orchestrator.specs import (
    PodSpec,
    RetryPolicy,
    WorkflowOptions,
)


COMMAND_LOGS_DIR_REMOTE = "/workspace/command_logs"
SCRIPT_YAML_REMOTE = "/workspace/runtime_configs/run_experiments.yaml"
EXPERIMENT_CONFIGS_PY_REMOTE = "/workspace/runtime_configs/experiment_configs.py"

REMOTE_REPO_DIR = "/workspace/repos/scaling-llms"
IDENTITY_FILE = "~/.ssh/runpod_key"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")


def _expand_env(value: Any) -> str:
    return os.path.expandvars(str(value))


def _expand_env_mapping(mapping: dict[str, Any] | None) -> dict[str, str]:
    if not mapping:
        return {}
    return {str(k): _expand_env(v) for k, v in mapping.items()}


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid '{key}' section")
    return value


@dataclass(frozen=True)
class PodOrchestratorConfig:
    pod_spec: PodSpec
    workflow: WorkflowOptions = field(default_factory=WorkflowOptions)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PodOrchestratorConfig:
        config_path = Path(path).expanduser().resolve()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        pod_data = _require_dict(data, "pod_spec")
        workflow_data = data.get("workflow", {}) or {}
        retry_data = workflow_data.get("retry_policy", {}) or {}

        try:
            pod_spec = PodSpec(
                name=str(pod_data["name"]),
                image_name=str(pod_data["image_name"]),
                gpu_type_id=str(pod_data["gpu_type_id"]) if pod_data.get("gpu_type_id") else None,
                cpu_type_id=str(pod_data["cpu_type_id"]) if pod_data.get("cpu_type_id") else None,
                gpu_count=int(pod_data.get("gpu_count", 1)),
                cloud_type=str(pod_data.get("cloud_type", "SECURE")),
                container_disk_in_gb=int(pod_data.get("container_disk_in_gb", 15)),
                volume_in_gb=int(pod_data.get("volume_in_gb", 20)),
                ports=str(pod_data.get("ports", "22/tcp")),
                env=_expand_env_mapping(pod_data.get("env")),
            )
        except (KeyError, ValueError) as exc:
            raise ConfigError(f"Invalid pod_spec: {exc}") from exc

        workflow = WorkflowOptions(
            timeout_s=int(workflow_data.get("timeout_s", 900)),
            poll_s=int(workflow_data.get("poll_s", 5)),
            terminate_on_failure=bool(workflow_data.get("terminate_on_failure", False)),
            retry_policy=RetryPolicy(
                max_attempts=int(retry_data.get("max_attempts", 5)),
                retry_base_s=float(retry_data.get("retry_base_s", 2.0)),
            ),
        )

        return cls(pod_spec=pod_spec, workflow=workflow)