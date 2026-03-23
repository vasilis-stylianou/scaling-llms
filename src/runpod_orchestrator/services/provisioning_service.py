# runpod_orchestrator/services/provision.py

from __future__ import annotations

import shlex

from runpod_orchestrator.clients.ssh import SSHClient
from runpod_orchestrator.exceptions import ProvisioningError
from runpod_orchestrator.specs import PodConnectionInfo, ProvisioningSpec


def _build_clone_or_update_command(repo_url: str, repo_dir: str, repo_branch: str | None) -> str:
    repo_url_q = shlex.quote(repo_url)
    repo_dir_q = shlex.quote(repo_dir)

    branch_cmd = ""
    if repo_branch:
        branch_q = shlex.quote(repo_branch)
        branch_cmd = (
            f"git -C {repo_dir_q} fetch origin {branch_q} && "
            f"git -C {repo_dir_q} checkout {branch_q} && "
            f"git -C {repo_dir_q} pull --ff-only origin {branch_q}"
        )
    else:
        branch_cmd = f"git -C {repo_dir_q} pull --ff-only"

    return (
        f"mkdir -p $(dirname {repo_dir_q}) && "
        f"if [ ! -d {repo_dir_q}/.git ]; then "
        f"git clone {repo_url_q} {repo_dir_q}; "
        f"fi && "
        f"{branch_cmd}"
    )


def _build_poetry_install_command(repo_dir: str, extra_args: list[str]) -> str:
    repo_dir_q = shlex.quote(repo_dir)
    extra = " ".join(shlex.quote(arg) for arg in extra_args)
    suffix = f" {extra}" if extra else ""
    return f"cd {repo_dir_q} && poetry install{suffix}"


def _build_kernel_command(repo_dir: str, kernel_name: str, display_name: str) -> str:
    repo_dir_q = shlex.quote(repo_dir)
    kernel_name_q = shlex.quote(kernel_name)
    display_name_q = shlex.quote(display_name)
    return (
        f"cd {repo_dir_q} && "
        f"poetry run python -m ipykernel install --user "
        f"--name {kernel_name_q} --display-name {display_name_q}"
    )


class ProvisioningService:
    """
    ProvisioningService is responsible for setting up a pod, including cloning or updating
    the repository, installing dependencies with Poetry, and optionally creating a Jupyter kernel.
    """
    def __init__(self, ssh_client: SSHClient) -> None:
        self.ssh = ssh_client

    def setup_pod(self, conn: PodConnectionInfo, spec: ProvisioningSpec) -> None:
        try:
            self.ssh.wait_until_ready(conn, timeout_s=300, poll_s=5)

            if spec.rclone_config_local:
                self.ssh.run(conn, "mkdir -p /root/.config/rclone")
                self.ssh.upload(conn, spec.rclone_config_local, spec.rclone_config_remote)

            self.ssh.run(conn, _build_clone_or_update_command(spec.repo_url, spec.repo_dir, spec.repo_branch))
            self.ssh.run(conn, _build_poetry_install_command(spec.repo_dir, spec.poetry_install_args))

            if spec.create_jupyter_kernel:
                self.ssh.run(
                    conn,
                    _build_kernel_command(
                        spec.repo_dir,
                        spec.kernel_name,
                        spec.kernel_display_name,
                    ),
                )
        except Exception as exc:
            raise ProvisioningError(f"Failed to provision pod {conn.pod_id}") from exc