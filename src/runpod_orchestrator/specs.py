from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ------------------------------
# HELPER CLASSES
# ------------------------------
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    retry_base_s: float = 2.0


@dataclass(frozen=True)
class PodConnectionInfo:
    pod_id: str
    name: str
    desired_status: str | None
    runtime_status: str | None
    public_ip: str | None
    ssh_port: int | None

    @property
    def is_ssh_ready(self) -> bool:
        return self.public_ip is not None and self.ssh_port is not None


# ------------------------------
# USER SPECIFIED CONFIGS
# ------------------------------
@dataclass(frozen=True)
class PodSpec:
    name: str
    image_name: str
    gpu_type_id: str | None = None
    cpu_type_id: str | None = None
    gpu_count: int = 1
    cloud_type: str = "SECURE"
    container_disk_in_gb: int = 20
    volume_in_gb: int = 40
    ports: str = "22/tcp"
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("PodSpec.name must be non-empty")

        if not self.image_name.strip():
            raise ValueError("PodSpec.image_name must be non-empty")

        if self.gpu_type_id is None and self.cpu_type_id is None:
            raise ValueError("Need to specify at least one of cpu_type_id or gpu_type_id")

        if not self.ports.strip():
            raise ValueError("PodSpec.ports must be non-empty")

    @property
    def create_pod_sdk_args(self) -> dict[str, Any]:
        return {
            "name": self.name.strip(),
            "image_name": self.image_name.strip(),
            "instance_id": self.cpu_type_id,
            "gpu_type_id": self.gpu_type_id,
            "gpu_count": self.gpu_count,
            "cloud_type": self.cloud_type,
            "container_disk_in_gb": self.container_disk_in_gb,
            "volume_in_gb": self.volume_in_gb,
            "ports": self.ports.strip(),
            "env": {str(k): str(v) for k, v in self.env.items()},
        }


@dataclass(frozen=True)
class ProvisioningSpec:
    repo_dir: str
    repo_url: str
    repo_branch: str | None = None
    rclone_config_local: str | None = None
    rclone_config_remote: str = "/root/.config/rclone/rclone.conf"
    create_jupyter_kernel: bool = False
    kernel_name: str = "scaling-llms"
    kernel_display_name: str = "Python (scaling-llms)"
    poetry_install_args: list[str] = field(default_factory=list)
    env_file_local: str | None = None
    env_file_remote: str | None = None


@dataclass(frozen=True)
class CommandSpec:
    work_dir: str
    tmux_session_name: str
    log_path: str
    command: str
    gpu_count: int = 1
    stop_pod_at_success: bool = False
    stop_pod_at_failure: bool = False
    upload_files: tuple[tuple[str, str], ...] = ()  # [(local_path, remote_path)]


@dataclass(frozen=True)
class WorkflowOptions:
    reuse_if_exists: bool = True
    timeout_s: int = 900
    poll_s: int = 5
    terminate_after_launch: bool = False
    terminate_on_failure: bool = False
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

