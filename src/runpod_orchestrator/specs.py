from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    retry_base_s: float = 2.0


@dataclass(frozen=True)
class PodSpec:
    name: str
    image_name: str
    gpu_type_id: str
    cloud_type: str = "SECURE"
    container_disk_in_gb: int = 20
    volume_in_gb: int = 40
    ports: str = "22/tcp"
    identity_file: str = "~/.ssh/runpod_key"
    env: dict[str, str] = field(default_factory=dict)

    @property
    def expanded_identity_file(self) -> str:
        return str(Path(self.identity_file).expanduser())


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

    def ssh_command(self, identity_file: str) -> str:
        if not self.is_ssh_ready:
            raise RuntimeError("SSH is not ready yet.")
        identity_file = str(Path(identity_file).expanduser())
        return f"ssh root@{self.public_ip} -p {self.ssh_port} -i {identity_file}"


@dataclass(frozen=True)
class ProvisioningSpec:
    repo_dir: str
    repo_url: str
    repo_branch: str | None = None
    identity_file: str = "~/.ssh/runpod_key"
    rclone_config_local: str | None = None
    rclone_config_remote: str = "/root/.config/rclone/rclone.conf"
    create_jupyter_kernel: bool = False
    kernel_name: str = "scaling-llms"
    kernel_display_name: str = "Python (scaling-llms)"
    poetry_install_args: list[str] = field(default_factory=list)

    @property
    def expanded_identity_file(self) -> str:
        return str(Path(self.identity_file).expanduser())


@dataclass(frozen=True)
class JobLauncherSpec:
    command: str
    repo_dir: str
    identity_file: str = "~/.ssh/runpod_key"
    tmux_session_name: str = "tmux_job"
    log_path: str = "/workspace/runs/tmux_job.log"

    @property
    def expanded_identity_file(self) -> str:
        return str(Path(self.identity_file).expanduser())


@dataclass(frozen=True)
class WorkflowOptions:
    reuse_if_exists: bool = True
    timeout_s: int = 900
    poll_s: int = 5
    terminate_after_launch: bool = False
    terminate_on_failure: bool = False
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

# TODO: remove
# @dataclass(frozen=True)
# class RunResult:
#     pod: PodConnectionInfo
#     tmux_session_name: str
#     log_path: str
