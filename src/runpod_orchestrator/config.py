from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

from runpod_orchestrator.exceptions import ConfigError
from runpod_orchestrator.specs import (
    CommandSpec,
    PodSpec,
    ProvisioningSpec,
    RetryPolicy,
    WorkflowOptions,
)


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
    provisioning: ProvisioningSpec
    command_spec: CommandSpec
    identity_file: str
    runpod_api_key: str | None = None
    work_dir: str | None = None
    workflow: WorkflowOptions = field(default_factory=WorkflowOptions)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PodOrchestratorConfig:
        config_path = Path(path).expanduser().resolve()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        pod_data = _require_dict(data, "pod_spec")
        provisioning_data = _require_dict(data, "provisioning")
        command_data = _require_dict(data, "command_spec")
        workflow_data = data.get("workflow", {}) or {}
        retry_data = workflow_data.get("retry_policy", {}) or {}

        identity_file = str(pod_data.get("identity_file", "~/.ssh/runpod_key"))
        runpod_api_key = str(data.get("runpod_api_key")) if data.get("runpod_api_key") is not None else None

        try:
            pod_spec = PodSpec(
                name=str(pod_data["name"]),
                image_name=str(pod_data["image_name"]),
                gpu_type_id=str(pod_data["gpu_type_id"]) if "gpu_type_id" in pod_data else None,
                cpu_type_id=str(pod_data["cpu_type_id"]) if "cpu_type_id" in pod_data else None,
                cloud_type=str(pod_data.get("cloud_type", "SECURE")),
                container_disk_in_gb=int(pod_data.get("container_disk_in_gb", 15)),
                volume_in_gb=int(pod_data.get("volume_in_gb", 20)),
                ports=str(pod_data.get("ports", "22/tcp")),
                env=_expand_env_mapping(pod_data.get("env")),
            )
        except KeyError as exc:
            raise ConfigError(f"Missing pod_spec field: {exc.args[0]}") from exc

        try:
            provisioning = ProvisioningSpec(
                repo_dir=str(provisioning_data["repo_dir"]),
                repo_url=str(provisioning_data["repo_url"]),
                repo_branch=(
                    str(provisioning_data["repo_branch"])
                    if provisioning_data.get("repo_branch") is not None
                    else None
                ),
                rclone_config_local=(
                    str(provisioning_data["rclone_config_local"])
                    if provisioning_data.get("rclone_config_local") is not None
                    else None
                ),
                rclone_config_remote=str(
                    provisioning_data.get(
                        "rclone_config_remote",
                        "/root/.config/rclone/rclone.conf",
                    )
                ),
                create_jupyter_kernel=bool(
                    provisioning_data.get("create_jupyter_kernel", False)
                ),
                kernel_name=str(provisioning_data.get("kernel_name", "scaling-llms")),
                kernel_display_name=str(
                    provisioning_data.get("kernel_display_name", "Python (scaling-llms)")
                ),
                poetry_install_args=[
                    str(arg)
                    for arg in (provisioning_data.get("poetry_install_args", []) or [])
                ],
                env_file_local=(
                    str(provisioning_data["env_file_local"])
                    if provisioning_data.get("env_file_local") is not None
                    else None
                ),
                env_file_remote=(
                    str(provisioning_data["env_file_remote"])
                    if provisioning_data.get("env_file_remote") is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise ConfigError(f"Missing provisioning field: {exc.args[0]}") from exc

        try:
            command_spec = CommandSpec(
                command=str(command_data["command"]),
                work_dir=str(command_data.get("repo_dir", provisioning.repo_dir)),
                tmux_session_name=str(command_data.get("tmux_session_name", "job")),
                log_path=str(command_data.get("log_path", "/workspace/tmux_logs/job.log")),
            )
        except KeyError as exc:
            raise ConfigError(f"Missing command_spec field: {exc.args[0]}") from exc

        workflow = WorkflowOptions(
            reuse_if_exists=bool(workflow_data.get("reuse_if_exists", True)),
            timeout_s=int(workflow_data.get("timeout_s", 900)),
            poll_s=int(workflow_data.get("poll_s", 5)),
            terminate_after_launch=bool(
                workflow_data.get("terminate_after_launch", False)
            ),
            terminate_on_failure=bool(
                workflow_data.get("terminate_on_failure", False)
            ),
            retry_policy=RetryPolicy(
                max_attempts=int(retry_data.get("max_attempts", 5)),
                retry_base_s=float(retry_data.get("retry_base_s", 2.0)),
            ),
        )

        return cls(
            pod_spec=pod_spec,
            provisioning=provisioning,
            command_spec=command_spec,
            workflow=workflow,
            identity_file=identity_file,
            runpod_api_key=runpod_api_key,
            work_dir=data.get("work_dir"),  # TODO

        )
