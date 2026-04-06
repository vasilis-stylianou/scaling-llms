from __future__ import annotations

import re
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

_RUNTIME_CONFIG_REMOTE = "/workspace/runtime_configs/run_experiments.yaml"
_RCLONE_CONFIG_REMOTE = "/root/.config/rclone/rclone.conf"

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


def _slugify(text: str) -> str:
    """Convert an arbitrary string into a safe identifier (underscores, no spaces)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text).strip("_") or "job"


def _build_command(script: str, gpu_count: int, runtime_config_remote: str) -> str:
    """
    Build the poetry run command appropriate for the target GPU count.

    - gpu_count <= 1  → plain `poetry run python <script> <config>`
    - gpu_count >  1  → `poetry run python -m torch.distributed.run
                              --standalone --nnodes=1 --nproc_per_node=N
                              <script> <config> --backend nccl`
    """
    if gpu_count > 1:
        return (
            f"poetry run python -m torch.distributed.run "
            f"--standalone --nnodes=1 --nproc_per_node={gpu_count} "
            f"{script} {runtime_config_remote} --backend nccl"
        )
    return f"poetry run python {script} {runtime_config_remote}"


@dataclass(frozen=True)
class PodOrchestratorConfig:
    pod_spec: PodSpec
    provisioning: ProvisioningSpec
    command_spec: CommandSpec
    identity_file: str
    runpod_api_key: str | None = None
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

        # identity_file lives at the top level of the YAML, not inside pod_spec
        identity_file = str(data.get("identity_file", "~/.ssh/runpod_key"))
        runpod_api_key = (
            str(data["runpod_api_key"]) if data.get("runpod_api_key") is not None else None
        )

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

        try:
            repo_dir = str(provisioning_data["repo_dir"]).strip()

            if provisioning_data.get("env_file_local") is not None:
                env_file_local = str(provisioning_data["env_file_local"]).strip()
                if provisioning_data.get("env_file_remote") is None:
                    env_file_remote = f"{repo_dir}/.env"
            else:
                env_file_local = None
                env_file_remote = None

            provisioning = ProvisioningSpec(
                repo_dir=repo_dir,
                repo_url=str(provisioning_data["repo_url"]),
                repo_branch=(
                    str(provisioning_data["repo_branch"])
                    if provisioning_data.get("repo_branch") is not None
                    else "main"
                ),
                rclone_config_local=(
                    str(provisioning_data["rclone_config_local"])
                    if provisioning_data.get("rclone_config_local") is not None
                    else None
                ),
                rclone_config_remote=str(
                    provisioning_data.get(
                        "rclone_config_remote",
                        _RCLONE_CONFIG_REMOTE,
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
                env_file_local=env_file_local,
                env_file_remote=env_file_remote,
            )
        except KeyError as exc:
            raise ConfigError(f"Missing provisioning field: {exc.args[0]}") from exc

        try:
            # --- script / command ---
            script = str(command_data["script"])
            runtime_config_remote = str(
                command_data.get("runtime_config_remote", _RUNTIME_CONFIG_REMOTE)
            )
            command = _build_command(script, pod_spec.gpu_count, runtime_config_remote)

            # --- work_dir always comes from provisioning ---
            work_dir = provisioning.repo_dir

            # --- tmux session name: explicit override or derived from pod name ---
            raw_session = command_data.get("tmux_session_name")
            tmux_session_name = (
                str(raw_session) if raw_session else _slugify(pod_spec.name)
            )

            # --- log path: explicit override or derived from session name ---
            raw_log = command_data.get("log_path")
            log_path = (
                str(raw_log)
                if raw_log
                else f"/workspace/tmux_logs/{tmux_session_name}.log"
            )

            # --- upload_files: built from runtime_config_local if provided ---
            runtime_config_local = command_data.get("runtime_config_local")
            upload_files: tuple[tuple[str, str], ...]
            if runtime_config_local is not None:
                upload_files = ((str(runtime_config_local), runtime_config_remote),)
            else:
                upload_files = ()

            command_spec = CommandSpec(
                command=command,
                work_dir=work_dir,
                tmux_session_name=tmux_session_name,
                log_path=log_path,
                stop_pod_at_success=bool(command_data.get("stop_pod_at_success", False)),
                stop_pod_at_failure=bool(command_data.get("stop_pod_at_failure", False)),
                upload_files=upload_files,
            )
        except KeyError as exc:
            raise ConfigError(f"Missing command_spec field: {exc.args[0]}") from exc

        workflow = WorkflowOptions(
            reuse_if_exists=bool(workflow_data.get("reuse_if_exists", False)),
            timeout_s=int(workflow_data.get("timeout_s", 900)),
            poll_s=int(workflow_data.get("poll_s", 5)),
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
        )