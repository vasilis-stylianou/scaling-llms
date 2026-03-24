from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

from runpod_orch.exceptions import ConfigError
from runpod_orch.specs import (
    JobLauncherSpec,
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
    bootstrap_script: str
    provisioning: ProvisioningSpec
    job_launcher_spec: JobLauncherSpec
    work_dir: str | None = None
    workflow: WorkflowOptions = field(default_factory=WorkflowOptions)

    @property
    def bootstrap_script_path(self) -> Path:
        if self.work_dir:
            return Path(self.work_dir).expanduser().resolve() / Path(self.bootstrap_script)
        else:
            return Path(self.bootstrap_script).expanduser().resolve()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PodOrchestratorConfig:
        config_path = Path(path).expanduser().resolve()
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        pod_data = _require_dict(data, "pod_spec")
        provisioning_data = _require_dict(data, "provisioning")
        job_launcher_data = _require_dict(data, "job_launcher_spec")
        workflow_data = data.get("workflow", {}) or {}
        retry_data = workflow_data.get("retry_policy", {}) or {}

        pod_identity_file = str(pod_data.get("identity_file", "~/.ssh/runpod_key"))

        try:
            pod_spec = PodSpec(
                name=str(pod_data["name"]),
                image_name=str(pod_data["image_name"]),
                gpu_type_id=str(pod_data["gpu_type_id"]),
                cloud_type=str(pod_data.get("cloud_type", "SECURE")),
                container_disk_in_gb=int(pod_data.get("container_disk_in_gb", 20)),
                volume_in_gb=int(pod_data.get("volume_in_gb", 40)),
                ports=str(pod_data.get("ports", "22/tcp")),
                identity_file=pod_identity_file,
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
                identity_file=str(
                    provisioning_data.get("identity_file", pod_identity_file)
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
            )
        except KeyError as exc:
            raise ConfigError(f"Missing provisioning field: {exc.args[0]}") from exc

        try:
            job_launcher_spec = JobLauncherSpec(
                command=str(job_launcher_data["command"]),
                repo_dir=str(job_launcher_data.get("repo_dir", provisioning.repo_dir)),
                identity_file=str(
                    job_launcher_data.get("identity_file", pod_identity_file)
                ),
                tmux_session_name=str(job_launcher_data.get("tmux_session_name", "job")),
                log_path=str(job_launcher_data.get("log_path", "/workspace/runs/job.log")),
            )
        except KeyError as exc:
            raise ConfigError(f"Missing job_launcher_spec field: {exc.args[0]}") from exc

        bootstrap_script = data.get("bootstrap_script")
        if bootstrap_script is None:
            raise ConfigError("Missing 'bootstrap_script'")

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
            bootstrap_script=str(bootstrap_script),
            provisioning=provisioning,
            job_launcher_spec=job_launcher_spec,
            workflow=workflow,
            work_dir=data.get("work_dir"),
        )
