from __future__ import annotations

import logging
import shlex
from pathlib import Path

from runpod_orch.clients import SSHClient
from runpod_orch.exceptions import ProvisioningError
from runpod_orch.specs import PodConnectionInfo, ProvisioningSpec

logger = logging.getLogger(__name__)


class PodProvisioner:
    """Sets up a pod: rclone config, repo clone/update, poetry install, optional kernel."""

    def __init__(self, ssh_client: SSHClient, spec: ProvisioningSpec) -> None:
        self.ssh_client = ssh_client
        self.spec = spec

    def copy_rclone_config(self, conn: PodConnectionInfo) -> None:
        """
        Copy a local rclone config into the remote pod when configured.

        Creates the remote parent directory and uses `scp` to transfer the file.
        """
        try:
            if self.spec.rclone_config_local is None:
                return

            local_rclone = Path(self.spec.rclone_config_local).expanduser().resolve()
            if not local_rclone.exists():
                raise FileNotFoundError(f"rclone config not found: {local_rclone}")

            remote_parent = str(Path(self.spec.rclone_config_remote).parent)
            self.ssh_client.run_command(
                conn,
                f"mkdir -p {shlex.quote(remote_parent)}",
            )
            logger.info("Copying rclone config to %s", self.spec.rclone_config_remote)
            self.ssh_client.scp_to_pod(
                conn,
                local_path=local_rclone,
                remote_path=self.spec.rclone_config_remote,
            )
        except Exception as exc:
            raise ProvisioningError(
                f"Failed to provision pod {conn.pod_id}"
            ) from exc

    def clone_or_update_repo(self, conn: PodConnectionInfo) -> None:
        """
        Ensure the repository is present on the remote pod, cloning or updating.

        Either clones the specified repo (and branch) or performs a fetch/pull to
        keep an existing checkout up to date.
        """
        try:
            repo_url_value = self.spec.repo_url.strip()
            if not repo_url_value:
                raise ValueError("ProvisioningSpec.repo_url must be provided")

            repo_dir = shlex.quote(self.spec.repo_dir)
            repo_url = shlex.quote(repo_url_value)
            repo_branch_value = (
                self.spec.repo_branch.strip() if self.spec.repo_branch else None
            )
            branch = (
                shlex.quote(repo_branch_value) if repo_branch_value else None
            )

            parent_dir = shlex.quote(str(Path(self.spec.repo_dir).parent))

            if branch:
                base_cmd = (
                    f"mkdir -p {parent_dir} && "
                    f"if [ -d {repo_dir}/.git ]; then "
                    f"git -C {repo_dir} fetch origin {branch} --prune && "
                    f"git -C {repo_dir} checkout {branch} && "
                    f"git -C {repo_dir} reset --hard origin/{branch}; "
                    f"else git clone --branch {branch} --single-branch "
                    f"{repo_url} {repo_dir}; fi"
                )
            else:
                base_cmd = (
                    f"mkdir -p {parent_dir} && "
                    f"if [ -d {repo_dir}/.git ]; then "
                    f"git -C {repo_dir} fetch --all --prune && "
                    "current_branch=$(git -C "
                    f"{repo_dir} rev-parse --abbrev-ref HEAD) && "
                    'if [ "$current_branch" = "HEAD" ]; then '
                    "default_branch=$(git -C "
                    f"{repo_dir} remote show origin | "
                    "sed -n 's/.*HEAD branch: //p') && "
                    f'git -C {repo_dir} checkout "$default_branch"; '
                    "fi && "
                    f"git -C {repo_dir} pull --ff-only; "
                    f"else git clone {repo_url} {repo_dir}; fi"
                )

            logger.info("Syncing repository at %s", self.spec.repo_dir)
            self.ssh_client.run_command(conn, base_cmd)
        except Exception as exc:
            raise ProvisioningError(
                f"Failed to provision pod {conn.pod_id}"
            ) from exc

    def poetry_install(self, conn: PodConnectionInfo) -> None:
        """
        Run `poetry install` in the checked-out repository on the pod.

        Invokes `poetry install` remotely with optional extra args supplied in the
        setup spec.
        """
        try:
            repo_dir = shlex.quote(self.spec.repo_dir)
            extra_args = " ".join(
                shlex.quote(arg) for arg in self.spec.poetry_install_args
            )
            cmd = f"cd {repo_dir} && poetry install"
            if extra_args:
                cmd = f"{cmd} {extra_args}"
            self.ssh_client.run_command(conn, cmd)
        except Exception as exc:
            raise ProvisioningError(
                f"Failed to provision pod {conn.pod_id}"
            ) from exc

    def create_jupyter_kernel(self, conn: PodConnectionInfo) -> None:
        """
        Install a Jupyter kernel in the remote environment when requested.

        When `create_jupyter_kernel` is True, installs an ipykernel entry for the
        project inside the pod's environment.
        """
        try:
            if not self.spec.create_jupyter_kernel:
                return

            repo_dir = shlex.quote(self.spec.repo_dir)
            kernel_name = shlex.quote(self.spec.kernel_name)
            kernel_display_name = shlex.quote(self.spec.kernel_display_name)
            cmd = (
                f"cd {repo_dir} && "
                "poetry run python -m ipykernel install --user "
                f"--name {kernel_name} --display-name {kernel_display_name}"
            )
            self.ssh_client.run_command(conn, cmd)
        except Exception as exc:
            raise ProvisioningError(
                f"Failed to provision pod {conn.pod_id}"
            ) from exc
