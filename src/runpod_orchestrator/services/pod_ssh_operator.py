from __future__ import annotations

import logging
import shlex
from pathlib import Path

from runpod_orchestrator.clients import SSHClient
from runpod_orchestrator.exceptions import CommandError, ProvisioningError
from runpod_orchestrator.specs import ProvisioningSpec, CommandSpec, PodConnectionInfo


logger = logging.getLogger(__name__)

def _build_tmux_job_command(spec: CommandSpec) -> str:
    """
    Build a shell command that launches the job inside a tmux session.

    Produces a single shell string that:
    - creates the log directory
    - kills any existing tmux session with the same name
    - starts a detached tmux session
    - runs the command from `work_dir` when provided
    - appends stdout/stderr to the configured log file
    """
    log_path = shlex.quote(spec.log_path)
    log_dir = shlex.quote(str(Path(spec.log_path).parent))
    session_name = shlex.quote(spec.tmux_session_name)

    command = spec.command.strip()
    if not command:
        raise ValueError("spec.command must be non-empty")

    work_dir_raw = spec.work_dir.strip()
    if work_dir_raw:
        work_dir = shlex.quote(work_dir_raw)
        job_cmd = (
            "set -euo pipefail; "
            f"if [ -d {work_dir} ]; then "
            f"cd {work_dir}; "
            "else "
            f'echo "Working directory does not exist: {work_dir_raw}" >&2; '
            "exit 1; "
            "fi; "
            f"{command} 2>&1 | tee -a {log_path}"
        )
    else:
        job_cmd = (
            "set -euo pipefail; "
            f"{command} 2>&1 | tee -a {log_path}"
        )

    job_cmd_quoted = shlex.quote(job_cmd)

    return (
        f"mkdir -p {log_dir} && "
        f"tmux kill-session -t {session_name} >/dev/null 2>&1 || true && "
        f"tmux new-session -d -s {session_name} bash -lc {job_cmd_quoted}"
    )



class PodSSHOperator:
    """Sets up a pod: rclone config, repo clone/update, poetry install, optional kernel."""
    def __init__(
        self,
        ssh_client: SSHClient | None = None,
    ) -> None:
        self.ssh = ssh_client

    def _validate_commands(
        self,
        conn: PodConnectionInfo,
        commands: list[str],
        *,
        error_prefix: str,
        error_cls: type[Exception] = ProvisioningError,
    ) -> None:
        for command in commands:
            result = self.ssh.run_command(conn, command, check=False)
            if result.returncode != 0:
                raise error_cls(
                    f"{error_prefix}\n"
                    f"Validation failed for command: {command}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

    def _validate_provisioning(
        self,
        conn: PodConnectionInfo,
        spec: ProvisioningSpec,
    ) -> None:
        error_prefix = f"Provisioning validation failed on pod {conn.pod_id}"
        repo_dir = shlex.quote(spec.repo_dir)

        commands = [
            f"test -d {repo_dir}",
            f"test -d {repo_dir}/.git",
            f"cd {repo_dir} && git rev-parse --is-inside-work-tree",
            f'export PATH="$HOME/.local/bin:$PATH" && cd {repo_dir} && poetry --version',
            (
                f'export PATH="$HOME/.local/bin:$PATH" && cd {repo_dir} && '
                "poetry run python -c 'import scaling_llms; print(\"success\")'"
            ),
        ]

        if spec.rclone_config_local is not None:
            commands.append(f"test -f {spec.rclone_config_remote}")

        if spec.env_file_local is not None and spec.env_file_remote is not None:
            commands.extend(
                [
                    f"test -f {spec.env_file_remote}",
                    f'test "$(stat -c %a {spec.env_file_remote})" = 600',
                ]
            )

        repo_branch_value = spec.repo_branch.strip() if spec.repo_branch else None
        if repo_branch_value:
            commands.append(
                f"cd {repo_dir} && test \"$(git rev-parse --abbrev-ref HEAD)\" = {shlex.quote(repo_branch_value)}"
            )

        self._validate_commands(
            conn,
            commands,
            error_prefix=error_prefix,
        )

    def set_ssh_client(self, ssh_client: SSHClient) -> None:
        self.ssh = ssh_client


    def copy_rclone_config(self, conn: PodConnectionInfo, spec: ProvisioningSpec) -> None:
        """
        Copy a local rclone config into the remote pod when configured.

        Creates the remote parent directory and uses `scp` to transfer the file.
        """
        error_prefix = f"Failed to copy rclone config to pod {conn.pod_id}"
        try:
            if spec.rclone_config_local is None:
                return

            local_rclone = Path(spec.rclone_config_local).expanduser().resolve()
            if not local_rclone.exists():
                raise FileNotFoundError(f"rclone config not found: {local_rclone}")

            remote_parent = str(Path(spec.rclone_config_remote).parent)
            self.ssh.run_command(
                conn,
                f"mkdir -p {shlex.quote(remote_parent)}",
            )
            logger.info("Copying rclone config to %s", spec.rclone_config_remote)
            self.ssh.scp_to_pod(
                conn,
                local_path=local_rclone,
                remote_path=spec.rclone_config_remote,
            )

        except Exception as exc:
            raise ProvisioningError(error_prefix) from exc

    def clone_or_update_repo(self, conn: PodConnectionInfo, spec: ProvisioningSpec) -> None:
        """
        Ensure the repository is present on the remote pod, cloning or updating.

        Either clones the specified repo (and branch) or performs a fetch/pull to
        keep an existing checkout up to date.
        """
        error_prefix = f"Failed to sync repository on pod {conn.pod_id}"
        try:
            repo_url_value = spec.repo_url.strip()
            if not repo_url_value:
                raise ValueError("ProvisioningSpec.repo_url must be provided")

            repo_dir = shlex.quote(spec.repo_dir)
            repo_url = shlex.quote(repo_url_value)
            repo_branch_value = (
                spec.repo_branch.strip() if spec.repo_branch else None
            )
            branch = (
                shlex.quote(repo_branch_value) if repo_branch_value else None
            )

            parent_dir = shlex.quote(str(Path(spec.repo_dir).parent))

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

            logger.info("Syncing repository at %s", spec.repo_dir)
            self.ssh.run_command(conn, base_cmd)

        except Exception as exc:
            raise ProvisioningError(error_prefix) from exc
        
        
    def copy_env_file(self, conn: PodConnectionInfo, spec: ProvisioningSpec) -> None:
        if spec.env_file_local is None:
            return
        
        error_prefix = f"Failed to copy env file on pod {conn.pod_id}"
        try:
            self.ssh.scp_to_pod(
                conn,
                local_path=spec.env_file_local,
                remote_path=spec.env_file_remote,
            )

            logger.info("Copying env file to %s", spec.env_file_remote)
            self.ssh.run_command(
                conn,
                f"chmod 600 {spec.env_file_remote}"
            )

        except Exception as exc:
            raise ProvisioningError(error_prefix) from exc

    def poetry_install(self, conn: PodConnectionInfo, spec: ProvisioningSpec) -> None:
        """
        Run `poetry install` in the checked-out repository on the pod.

        Invokes `poetry install` remotely with optional extra args supplied in the
        setup spec.
        """
        error_prefix = f"Failed to install poetry on pod {conn.pod_id}" 
        try:
            repo_dir = shlex.quote(spec.repo_dir)
            extra_args = " ".join(shlex.quote(arg) for arg in spec.poetry_install_args)

            poetry_install_cmd = "poetry install"
            if extra_args:
                poetry_install_cmd = f"{poetry_install_cmd} {extra_args}"

            cmd = (
                "set -euo pipefail; "
                f"cd {repo_dir}; "
                "python3 -m ensurepip --upgrade >/dev/null 2>&1 || true; "
                "python3 -m pip install --user poetry; "
                "export PATH=\"$HOME/.local/bin:$PATH\"; "
                "test -x \"$HOME/.local/bin/poetry\"; "
                "ln -sf \"$HOME/.local/bin/poetry\" /usr/local/bin/poetry; "
                "poetry --version; "
                f"{poetry_install_cmd}"
            )

            logger.info("Running poetry install in %s", spec.repo_dir)
            self.ssh.run_command(conn, cmd)

        except Exception as exc:
            raise ProvisioningError(error_prefix) from exc
        

    def create_jupyter_kernel(self, conn: PodConnectionInfo, spec: ProvisioningSpec) -> None:
        """
        Install a Jupyter kernel in the remote environment when requested.

        When `create_jupyter_kernel` is True, installs an ipykernel entry for the
        project inside the pod's environment.
        """
        try:
            if not spec.create_jupyter_kernel:
                return

            repo_dir = shlex.quote(spec.repo_dir)
            kernel_name = shlex.quote(spec.kernel_name)
            kernel_display_name = shlex.quote(spec.kernel_display_name)
            cmd = (
                f"cd {repo_dir} && "
                "poetry run python -m ipykernel install --user "
                f"--name {kernel_name} --display-name {kernel_display_name}"
            )
            logger.info("Creating Jupyter kernel %s", spec.kernel_name)
            self.ssh.run_command(conn, cmd)
        except Exception as exc:
            raise ProvisioningError(
                f"Failed to create Jupyter kernel on pod {conn.pod_id}"
            ) from exc
        
    def launch_tmux_job(self, conn: PodConnectionInfo, spec: CommandSpec) -> str:
        """Launch the job command on the pod inside a tmux session.

        Executes the tmux-start command remotely and returns the configured log
        path where the job output will be written.
        """
        error_prefix = f"Failed to launch job on pod {conn.pod_id}"
        try:
            remote_cmd = _build_tmux_job_command(spec)

            logger.info("Launching job with command: %s", spec.command)
            self.ssh.run_command(conn, remote_cmd)
            return spec.log_path
        except Exception as exc:
            raise CommandError(error_prefix) from exc
        