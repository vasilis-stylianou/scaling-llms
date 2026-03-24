from __future__ import annotations

import shlex
from pathlib import Path

from runpod_orch.clients import SSHClient
from runpod_orch.exceptions import JobError
from runpod_orch.specs import PodConnectionInfo, JobLauncherSpec


class PodJobLauncher:
    """Launches job commands on a pod inside a tmux session over SSH."""

    def __init__(self, ssh_client: SSHClient, spec: JobLauncherSpec) -> None:
        self.ssh = ssh_client
        self.spec = spec

    def launch_tmux_job(self, conn: PodConnectionInfo) -> str:
        """Launch the job command on the pod inside a tmux session.

        Executes the tmux-start command remotely and returns the configured log
        path where the job output will be written.
        """
        try:
            remote_cmd = self._build_tmux_job_command(self.spec)
            self.ssh.run_command(conn, remote_cmd)
            return self.spec.log_path
        except Exception as exc:
            raise JobError(f"Failed to launch job on pod {conn.pod_id}") from exc


    @staticmethod
    def _build_tmux_job_command(spec: JobLauncherSpec) -> str:
        """
        Build a shell command that launches the job inside a tmux session.
        
        Produces a single shell string that creates the log dir, starts a detached
        tmux session and appends output to the configured log path.
        """
        log_path = shlex.quote(spec.log_path)
        log_dir = shlex.quote(str(Path(spec.log_path).parent))
        repo_dir = shlex.quote(spec.repo_dir)
        session_name = shlex.quote(spec.tmux_session_name)

        if spec.repo_dir.strip():
            job_cmd = (
                f"if [ -d {repo_dir} ]; then "
                f"cd {repo_dir} && {spec.command}; "
                f"else {spec.command}; fi 2>&1 | tee -a {log_path}"
            )
        else:
            job_cmd = f"{spec.command} 2>&1 | tee -a {log_path}"

        job_cmd_quoted = shlex.quote(job_cmd)

        return (
            f"mkdir -p {log_dir} && "
            f"tmux kill-session -t {session_name} >/dev/null 2>&1 || true && "
            f"tmux new-session -d -s {session_name} bash -lc {job_cmd_quoted}"
        )