from __future__ import annotations

from pathlib import Path
import subprocess

from dotenv import load_dotenv
import re

from scaling_llms.utils.loggers import setup_console_logging
from runpod_orchestrator.clients.ssh import SSHClient
from runpod_orchestrator.config import (
    COMMAND_LOGS_DIR_REMOTE,
    EXPERIMENT_CONFIGS_PY_REMOTE,
    IDENTITY_FILE,
    REPO_DIR,
    RUNPOD_API_KEY,
    SCRIPT_YAML_REMOTE,
    PodOrchestratorConfig,
)
from runpod_orchestrator.services import (
    PodConnectionInfo,
    PodManager,
    PodSSHOperator,
)

setup_console_logging()

def _slugify(text: str) -> str:
    """Convert an arbitrary string into a safe identifier (underscores, no spaces)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", text).strip("_") or "job"


class PodOrchestrator(PodSSHOperator, PodManager):
    """
    Top-level API for pod lifecycle orchestration.

    Exposes:
      - create   : create a pod and wait for SSH
      - provision : run provisioning steps on an existing pod
      - submit    : launch a job on an existing pod
      - stop      : stop a pod
      - terminate : terminate (delete) a pod
    """

    def __init__(
        self,
        config: PodOrchestratorConfig,
    ) -> None:
        load_dotenv(Path(".env"), override=True)

        self._config = config
        self.pod_spec = config.pod_spec
        self.workflow = config.workflow
        self.repo_dir = REPO_DIR

        ssh = SSHClient(IDENTITY_FILE)

        PodSSHOperator.__init__(self, ssh_client=ssh)
        PodManager.__init__(self, api_key=RUNPOD_API_KEY, ssh_client=ssh)

        self.conn: PodConnectionInfo | None = None
        self.ssh = ssh

    # -- Pod Operations --
    def create(self) -> PodConnectionInfo:
        try:
            self.conn = self.create_and_wait(
                self.pod_spec,
                timeout_s=self.workflow.timeout_s,
                poll_s=self.workflow.poll_s,
                retry_policy=self.workflow.retry_policy,
            )
            return self.conn
        except Exception:
            if self.conn is not None and self.workflow.terminate_on_failure:
                try:
                    self.terminate(self.conn.pod_id)
                except Exception:
                    pass
            raise

    def stop(self, pod_id: str | None = None) -> None:
        pod_id = pod_id or (self.conn.pod_id if self.conn else None)
        if pod_id is None:
            raise ValueError("No pod_id provided and no active connection available.")
        self.stop_pod(pod_id, retry_policy=self.workflow.retry_policy)

    def terminate(self, pod_id: str | None = None) -> None:
        pod_id = pod_id or (self.conn.pod_id if self.conn else None)
        if pod_id is None:
            raise ValueError("No pod_id provided and no active connection available.")
        self.terminate_pod(pod_id, retry_policy=self.workflow.retry_policy)

    # --- SSH / job operations ---
    def validate_provisioning(
        self,
        conn: PodConnectionInfo | None = None,
    ) -> None:
        conn = conn or self.conn
        self._validate_provisioning(conn, self.repo_dir)

    def run(
        self,
        script_yaml: str | Path,
        experiment_configs_py: str | Path,
        *,
        script_rel_path: str = "scripts/run_experiments.py",
        job_session_name: str | None = None,
        stop_pod_at_success: bool = False,
        stop_pod_at_failure: bool = False,
        conn: PodConnectionInfo | None = None,
    ) -> str:
        conn = conn or self.conn

        # Prepare files to upload
        upload_files = [
            (str(script_yaml), SCRIPT_YAML_REMOTE),
            (str(experiment_configs_py), EXPERIMENT_CONFIGS_PY_REMOTE)

        ]

        # Prepare run command
        if self.pod_spec.gpu_count > 1:
            cmd = (
                f"poetry run python -m torch.distributed.run "
                f"--standalone --nnodes=1 --nproc_per_node={self.pod_spec.gpu_count} "
                f"{script_rel_path} {SCRIPT_YAML_REMOTE} --backend nccl"
            )
        else:
            cmd =f"poetry run python {script_rel_path} {SCRIPT_YAML_REMOTE}"
    
        # Prepare job session and log path
        job_session_name = job_session_name or _slugify(self.pod_spec.name)
        log_path = Path(COMMAND_LOGS_DIR_REMOTE) / f"{job_session_name}.log"
        
        # Upload necessary files and launch the job
        self.upload_files(conn, upload_files)
        return self.launch_tmux_job(
            conn,
            command=cmd,
            work_dir=self.repo_dir,
            job_session_name=job_session_name,
            log_path=str(log_path),
            stop_pod_at_success=stop_pod_at_success,
            stop_pod_at_failure=stop_pod_at_failure,
        )
    
    def open_remote_vscode(
        self, 
        conn: PodConnectionInfo | None = None,
        work_dir: str = "/workspace",
    ) -> None:
        conn = conn or self.conn
        if conn is None:
            raise ValueError("No active connection available.")

        # Install ipykernel to enable Jupyter support in VSCode remote sessions
        self.install_ipykernel(conn)

        # Add remote host to local SSH config
        local_repo_path = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=Path("."),
                text=True
            ).strip()
        )
        script_path = (local_repo_path / "scripts" / "setup_runpod_host.sh").as_posix()
        subprocess.run(
            ["bash", str(script_path), conn.public_ip, str(conn.ssh_port)],
            check=True
        )
        
        # Open VSCode connected to the pod
        subprocess.run(
            ["code", "--new-window", "--remote", "ssh-remote+running-runpod", work_dir],
            check=True,
        )