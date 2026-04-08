from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from scaling_llms.utils.loggers import setup_console_logging
from runpod_orchestrator.specs import ProvisioningSpec
from runpod_orchestrator.clients.ssh import SSHClient
from runpod_orchestrator.config import PodOrchestratorConfig
from runpod_orchestrator.services import (
    PodConnectionInfo,
    PodManager,
    PodSSHOperator,
)

setup_console_logging()


class PodOrchestrator(PodSSHOperator, PodManager):
    """
    Top-level API for pod lifecycle orchestration.

    Exposes:
      - create   : create (or reuse) a pod and wait for SSH
      - provision : run provisioning steps on an existing pod
      - submit    : launch a job on an existing pod
      - stop      : stop a pod
      - terminate : terminate (delete) a pod
      - run       : full create -> provision -> submit workflow
      - git_pull   : convenience method to git pull on the pod
    """

    def __init__(self, config: PodOrchestratorConfig, config_path: Path | None = None) -> None:
        load_dotenv(Path(".env"), override=True)

        self._config_path = config_path
        self._apply_config(config)

        ssh = SSHClient(config.identity_file)

        PodSSHOperator.__init__(self, ssh_client=ssh)
        PodManager.__init__(self, api_key=config.runpod_api_key, ssh_client=ssh)

        self.conn: PodConnectionInfo | None = None
        self.ssh = ssh

    def _apply_config(self, config: PodOrchestratorConfig) -> None:
        self._config = config
        self.pod_spec = config.pod_spec
        self.provisioning_spec = config.provisioning
        self.command_spec = config.command_spec
        self.identity_file = config.identity_file
        self.workflow = config.workflow

    def reload_config(self, path: str | Path | None = None) -> None:
        """Re-read the YAML config and refresh all config-derived attributes.

        ``self.conn`` and the active SSH client are preserved.
        If ``path`` is omitted, the path used at construction time is reused.
        """
        source = Path(path) if path is not None else self._config_path
        if source is None:
            raise ValueError(
                "No config path available. Pass a path explicitly or "
                "construct via PodOrchestrator.from_yaml()."
            )
        load_dotenv(Path(".env"), override=True)
        new_config = PodOrchestratorConfig.from_yaml(source)
        self._config_path = Path(source)
        self._apply_config(new_config)

    # -- high-level operations --
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

    def provision(
        self,
        conn: PodConnectionInfo | None = None,
        spec: ProvisioningSpec | None = None,
    ) -> None:
        conn = conn or self.conn
        spec = spec or self.provisioning_spec
        self.copy_rclone_config(conn, spec)
        self.clone_or_update_repo(conn, spec)
        self.copy_env_file(conn, spec)
        self.poetry_install(conn, spec)
        self.install_tmux(conn)

        # self.create_jupyter_kernel(conn, spec)

    def validate_provisioning(
        self,
        conn: PodConnectionInfo | None = None,
        spec: ProvisioningSpec | None = None,
    ) -> None:
        conn = conn or self.conn
        spec = spec or self.provisioning_spec
        self._validate_provisioning(conn, spec)

    def submit_job(
        self,
        *,
        conn: PodConnectionInfo | None = None,
        command: str | None = None,
        work_dir: str | None = None,
        job_session_name: str | None = None,
        log_path: str | None = None,
        stop_pod_at_success: bool | None = None,
        stop_pod_at_failure: bool | None = None,
        upload_files: tuple[tuple[str, str], ...] | None = None,
        git_pull_first: bool = True,
        as_tmux_job: bool = True,
        repo_dir: str | None = None,
    ) -> str:
        conn = conn or self.conn
        cs = self.command_spec
        _command = command if command is not None else cs.command
        _work_dir = work_dir if work_dir is not None else cs.work_dir
        _job_session_name = job_session_name if job_session_name is not None else cs.job_session_name
        _log_path = log_path if log_path is not None else cs.log_path
        _stop_pod_at_success = stop_pod_at_success if stop_pod_at_success is not None else cs.stop_pod_at_success
        _stop_pod_at_failure = stop_pod_at_failure if stop_pod_at_failure is not None else cs.stop_pod_at_failure
        _upload_files = upload_files if upload_files is not None else cs.upload_files
        _repo_dir = repo_dir if repo_dir is not None else self.provisioning_spec.repo_dir
        if git_pull_first:
            self.ssh_git_pull(conn, _repo_dir)
        if as_tmux_job:
            return self.launch_tmux_job(
                conn,
                command=_command,
                work_dir=_work_dir,
                job_session_name=_job_session_name,
                log_path=_log_path,
                stop_pod_at_success=_stop_pod_at_success,
                stop_pod_at_failure=_stop_pod_at_failure,
                upload_files=_upload_files,
            )
        else:
            self.ssh.run_command(conn, _command)
            return self.ssh.make_shell_command(conn, f"tail -f {_log_path}")

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

    def run(self) -> PodConnectionInfo:
        try:
            self.conn = self.create()
            self.provision(self.conn)
            self.submit_job(self.conn)

            return self.conn
        except Exception:
            if self.conn is not None and self.workflow.terminate_on_failure:
                self.terminate(self.conn.pod_id)
            raise

    # -- convenience --
    def ssh_git_pull(
        self,
        conn: PodConnectionInfo | None = None,
        repo_dir: str | None = None,
    ) -> None:
        conn = conn or self.conn
        repo_dir = repo_dir or self.provisioning_spec.repo_dir
        if conn is None:
            raise ValueError("No active connection available.")
        self.git_pull(conn, repo_dir)

    def set_ssh_client(self) -> None:
        self.ssh = SSHClient(self.identity_file)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PodOrchestrator:
        resolved = Path(path).expanduser().resolve()
        config = PodOrchestratorConfig.from_yaml(resolved)
        return cls(config=config, config_path=resolved)
