from __future__ import annotations

import logging
import shlex
import shutil
from pathlib import Path
from string import Template

from runpod_orchestrator.clients import SSHClient
import subprocess

from runpod_orchestrator.exceptions import CommandError, ProvisioningError
from runpod_orchestrator.specs import PodConnectionInfo


logger = logging.getLogger("PodSSH")

_TEMPLATE_PATH = Path("/Users/vasilis/Desktop/scaling-llms/scripts/tmux_job_template.sh")


def _build_tmux_job_command(
    command: str,
    work_dir: str,
    job_session_name: str,
    log_path: str,
    stop_pod_at_success: bool = False,
    stop_pod_at_failure: bool = False,
) -> str:
    command = command.strip()
    if not command:
        raise ValueError("command must be non-empty")

    work_dir = work_dir.strip()
    work_dir_block = (
        ""
        if not work_dir
        else (
            f"if [ -d {shlex.quote(work_dir)} ]; then\n"
            f"  cd {shlex.quote(work_dir)}\n"
            "else\n"
            f'  echo "Working directory does not exist: {work_dir}" >&2\n'
            "  exit 1\n"
            "fi"
        )
    )

    success_block = "  stop_current_pod" if stop_pod_at_success else "  true"
    failure_block = "  stop_current_pod || true" if stop_pod_at_failure else "  true"

    template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    script = template.substitute(
        log_path=shlex.quote(log_path),
        work_dir_block=work_dir_block,
        command=command,
        success_block=success_block,
        failure_block=failure_block,
    )
    script_quoted = shlex.quote(script)

    log_dir = shlex.quote(str(Path(log_path).parent))
    session_name = shlex.quote(job_session_name)

    return (
        f"mkdir -p {log_dir} && "
        f"tmux kill-session -t {session_name} >/dev/null 2>&1 || true && "
        f"tmux new-session -d -s {session_name} bash -lc {script_quoted}"
    )


class PodSSHOperator:
    def __init__(
        self,
        ssh_client: SSHClient | None = None,
    ) -> None:
        self.ssh = ssh_client

    def _validate_commands(
        self,
        conn: PodConnectionInfo,
        checks: list[tuple[str, str]],
        *,
        error_prefix: str,
        error_cls: type[Exception] = ProvisioningError,
    ) -> None:
        for command, success_message in checks:
            result = self.ssh.run_command(conn, command, check=False)
            if result.returncode != 0:
                raise error_cls(
                    f"{error_prefix}\n"
                    f"Validation failed for command: {command}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            logger.info(success_message)

    def _validate_provisioning(
        self,
        conn: PodConnectionInfo,
        repo_dir: str,
    ) -> None:
        error_prefix = f"Provisioning validation failed on pod {conn.pod_id}"
        repo_dir_q = shlex.quote(repo_dir)

        check_groups: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "repo",
                [
                    (f"test -d {repo_dir_q}", "[provisioning] Repo dir test passed"),
                    (f"test -f {repo_dir_q}/pyproject.toml", "[provisioning] pyproject.toml test passed"),
                    (f"test -d {repo_dir_q}/src", "[provisioning] src dir test passed"),
                ],
            ),
            (
                "python",
                [
                    ("command -v python", "[provisioning] Python binary test passed"),
                    ("python --version", "[provisioning] Python version test passed"),
                    (
                        f"cd {repo_dir_q} && python -c 'import scaling_llms; print(\"success\")'",
                        "[provisioning] Python import scaling_llms test passed",
                    ),
                    (
                        "python -c 'import sys; print(sys.executable)'",
                        "[provisioning] Python executable test passed",
                    ),
                ],
            ),
            (
                "poetry",
                [
                    ("command -v poetry", "[provisioning] Poetry binary test passed"),
                    ("poetry --version", "[provisioning] Poetry version test passed"),
                ],
            ),
            (
                "tools",
                [
                    ("command -v tmux", "[provisioning] tmux test passed"),
                    ("command -v rclone", "[provisioning] rclone binary test passed"),
                ],
            ),
            (
                "env",
                [
                    (f"test -f {repo_dir_q}/.env", "[provisioning] .env file test passed"),
                    (
                        f"grep -q '^DATABASE_URL=.' {repo_dir_q}/.env",
                        "[provisioning] DATABASE_URL test passed",
                    ),
                ],
            ),
            (
                "rclone",
                [
                    ("test -f /root/.config/rclone/rclone.conf", "[provisioning] rclone config file test passed"),
                    (
                        "grep -q '^\\[r2\\]' /root/.config/rclone/rclone.conf",
                        "[provisioning] r2 remote config test passed",
                    ),
                    (
                        "rclone listremotes | grep -q '^r2:$'",
                        "[provisioning] rclone remote listing test passed",
                    ),
                    (
                        "rclone ls r2:scaling-llms >/dev/null",
                        "[provisioning] r2 bucket access test passed",
                    ),
                ],
            ),
        ]

        for group_name, checks in check_groups:
            logger.info(f"[provisioning] Running {group_name} validation checks")
            self._validate_commands(
                conn,
                checks,
                error_prefix=error_prefix,
            )
            logger.info(f"[provisioning] {group_name.capitalize()} validation passed")

    def upload_files(self, conn: PodConnectionInfo, upload_files: list[tuple[str, str]] | None = None) -> None:
        """Upload local files to the remote pod before launching the job."""
        if upload_files is None or len(upload_files) == 0:
            return

        error_prefix = f"Failed to upload files to pod {conn.pod_id}"
        try:
            for local_path, remote_path in upload_files:
                local = Path(local_path).expanduser().resolve()
                if not local.exists():
                    raise FileNotFoundError(f"Upload file not found: {local}")

                remote_parent = str(Path(remote_path).parent)
                self.ssh.run_command(
                    conn,
                    f"mkdir -p {shlex.quote(remote_parent)}",
                )
                logger.info("[upload] %s -> %s", local, remote_path)
                self.ssh.scp_to_pod(
                    conn,
                    local_path=local,
                    remote_path=remote_path,
                )
        except Exception as exc:
            raise CommandError(error_prefix) from exc

    def upload_directory(
        self,
        conn: PodConnectionInfo,
        local_dir: str | Path,
        remote_parent: str,
    ) -> None:
        """
        Recursively upload a local directory into `remote_parent` on the pod.

        Creates `remote_parent` if missing, removes any pre-existing copy at
        `{remote_parent}/{local_dir.name}`, strips local `__pycache__` dirs to
        avoid shipping compiled artifacts, then runs `scp -r`.
        """
        local = Path(local_dir).expanduser().resolve()
        if not local.is_dir():
            raise FileNotFoundError(f"Upload directory not found: {local}")

        for pyc in local.rglob("__pycache__"):
            shutil.rmtree(pyc, ignore_errors=True)

        remote_parent = remote_parent.rstrip("/")
        remote_target = f"{remote_parent}/{local.name}"
        error_prefix = f"Failed to upload directory to pod {conn.pod_id}"
        try:
            self.ssh.run_command(
                conn,
                f"mkdir -p {shlex.quote(remote_parent)}",
            )
            self.ssh.run_command(
                conn,
                f"rm -rf {shlex.quote(remote_target)}",
            )
            logger.info("[upload] %s -> %s", local, remote_target)
            self.ssh.scp_dir_to_pod(
                conn,
                local_dir=local,
                remote_parent=remote_parent,
            )
        except Exception as exc:
            raise CommandError(error_prefix) from exc

    def set_ssh_client(self, ssh_client: SSHClient) -> None:
        self.ssh = ssh_client

    def launch_tmux_job(
        self,
        conn: PodConnectionInfo,
        command: str,
        work_dir: str,
        job_session_name: str,
        log_path: str,
        stop_pod_at_success: bool = False,
        stop_pod_at_failure: bool = False,
    ) -> str:
        """Launch the job command on the pod inside a tmux session.

        Executes the tmux-start command remotely and returns the configured log
        path where the job output will be written.
        """
        error_prefix = f"Failed to launch job on pod {conn.pod_id}"
        try:
            remote_cmd = _build_tmux_job_command(
                command=command,
                work_dir=work_dir,
                job_session_name=job_session_name,
                log_path=log_path,
                stop_pod_at_success=stop_pod_at_success,
                stop_pod_at_failure=stop_pod_at_failure,
            )

            logger.info("[job] Launching command with tmux: %s", command)
            logger.info("[job] Writing logs to: %s", log_path)
            self.ssh.run_command(conn, remote_cmd)

            # Return the command to stream logs from the job
            return self.ssh.make_shell_command(conn, f"tail -f {log_path}")
        except Exception as exc:
            raise CommandError(error_prefix) from exc

    def kill_job(
        self,
        conn: PodConnectionInfo,
        job_session_name: str | None = None,
        command_pattern: str | None = None,
    ) -> None:
        """Terminate a running job on the pod.

        Args:
            job_session_name: Name of the tmux session to kill. All child
                processes of the session are also sent SIGTERM.
            command_pattern: A substring of the command line to match with
                ``pkill -f``. Use this to kill a bare process launched via
                ``launch_job`` (no tmux). The pattern is matched against the
                full command line of every running process.

        At least one of the two arguments should be provided. Both can be
        supplied together, in which case the tmux session is killed first and
        then any remaining matching processes are cleaned up.
        """
        error_prefix = f"Failed to kill job on pod {conn.pod_id}"
        try:
            if job_session_name:
                session = shlex.quote(job_session_name)
                logger.info("[kill] Killing tmux session: %s", job_session_name)
                self.ssh.run_command(
                    conn,
                    f"tmux kill-session -t {session} >/dev/null 2>&1 || true",
                    check=False,
                )
                # Also kill any child processes spawned by the session that
                # may outlive the tmux server (e.g. the training script).
                self.ssh.run_command(
                    conn,
                    f"pkill -TERM -f {session} 2>/dev/null || true",
                    check=False,
                )
                logger.info("[kill] Tmux session '%s' terminated", job_session_name)

            if command_pattern:
                pattern = shlex.quote(command_pattern)
                logger.info("[kill] Killing processes matching pattern: %s", command_pattern)
                self.ssh.run_command(
                    conn,
                    f"pkill -TERM -f {pattern} 2>/dev/null || true",
                    check=False,
                )
                logger.info("[kill] Processes matching '%s' sent SIGTERM", command_pattern)

            if not job_session_name and not command_pattern:
                logger.info("[kill] No session name or pattern provided; killing all tmux sessions")
                self.ssh.run_command(
                    conn,
                    "tmux kill-server >/dev/null 2>&1 || true",
                    check=False,
                )
        except Exception as exc:
            raise CommandError(error_prefix) from exc

    def git_pull(self, conn: PodConnectionInfo, repo_dir: str) -> None:
        """Run `git pull` in the specified directory on the remote pod."""
        error_prefix = f"Failed to run git pull on pod {conn.pod_id}"
        try:
            repo_dir_quoted = shlex.quote(repo_dir)
            cmd = f"cd {repo_dir_quoted} && git pull"
            logger.info("[git] Running git pull in %s", repo_dir)
            self.ssh.run_command(conn, cmd)
        except Exception as exc:
            raise CommandError(error_prefix) from exc
        
    def install_ipykernel( self,
        conn: PodConnectionInfo,
        repo_dir: str = "/workspace/repos/scaling-llms",
    ) -> None:
        """
        Install a Jupyter kernel in the remote environment when requested.
        """
        try:
            repo_dir_q = shlex.quote(repo_dir)
            cmd = (
                f"cd {repo_dir_q} && "
                "poetry add ipykernel"
            )
            logger.info("Installing ipykernel")
            self.ssh.run_command(conn, cmd)
        except Exception as exc:
            raise ProvisioningError(
                f"Failed to install ipykernel on pod {conn.pod_id}"
            ) from exc