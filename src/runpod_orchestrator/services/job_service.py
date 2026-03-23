from __future__ import annotations

import shlex

from runpod_orchestrator.clients.ssh import SSHClient
from runpod_orchestrator.exceptions import JobLaunchError
from runpod_orchestrator.specs import PodConnectionInfo, JobSpec


def build_job_tmux_command(
    *,
    repo_dir: str,
    command: str,
    tmux_session_name: str,
    log_path: str,
) -> str:
    repo_dir_q = shlex.quote(repo_dir)
    session_q = shlex.quote(tmux_session_name)
    log_path_q = shlex.quote(log_path)

    inner = (
        f"mkdir -p $(dirname {log_path_q}) && "
        f"cd {repo_dir_q} && "
        f"{command} 2>&1 | tee -a {log_path_q}"
    )
    inner_q = shlex.quote(inner)

    return (
        f"tmux has-session -t {session_q} 2>/dev/null || "
        f"tmux new-session -d -s {session_q} {inner_q}"
    )


class JobService:
    """
    JobService is responsible for launching job sessions on a pod using SSH.
    It handles waiting for the pod to be ready, constructing the job command, and
    executing it within a tmux session.
    """
    def __init__(self, ssh_client: SSHClient) -> None:
        self.ssh = ssh_client

    def launch_job(self, conn: PodConnectionInfo, spec: JobSpec) -> str:
        try:
            self.ssh.wait_until_ready(conn, timeout_s=300, poll_s=5)
            cmd = build_job_tmux_command(
                repo_dir=spec.repo_dir,
                command=spec.command,
                tmux_session_name=spec.tmux_session_name,
                log_path=spec.log_path,
            )
            self.ssh.run(conn, cmd)
            return spec.tmux_session_name
        except Exception as exc:
            raise JobLaunchError(f"Failed to launch job on pod {conn.pod_id}") from exc