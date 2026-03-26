from __future__ import annotations

from pathlib import Path

from runpod_orchestrator.specs import CommandSpec, ProvisioningSpec
from runpod_orchestrator.clients.ssh import SSHClient
from runpod_orchestrator.config import PodOrchestratorConfig
from runpod_orchestrator.services import (
    PodConnectionInfo,
    PodManager,
    PodSSHOperator,
)

class PodOrchestrator(PodSSHOperator, PodManager):
    """
    Top-level API for pod lifecycle orchestration.

    Exposes:
      - create   : create (or reuse) a pod and wait for SSH
      - provision : run provisioning steps on an existing pod
      - submit    : launch a job on an existing pod
      - stop      : stop a pod
      - resume    : resume a stopped pod
      - terminate : terminate (delete) a pod
      - run       : full create -> provision -> submit workflow
    """

    def __init__(self, config: PodOrchestratorConfig) -> None:
        self._config = config
        self.pod_spec = config.pod_spec
        self.provisioning_spec = config.provisioning
        self.command_spec = config.command_spec
        self.identity_file = config.identity_file
        self.workflow = config.workflow
        
        ssh = SSHClient(config.identity_file)

        PodSSHOperator.__init__(self, ssh_client=ssh)
        PodManager.__init__(self, api_key=config.runpod_api_key, ssh_client=ssh)

        self.conn: PodConnectionInfo | None = None
        self.ssh = ssh

    # -- high-level operations --
    def create(self) -> PodConnectionInfo:
        try:
            self.conn = self.resolve_or_create_pod(
                self.pod_spec,
                reuse_if_exists=self.workflow.reuse_if_exists,
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
        spec: ProvisioningSpec | None = None
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
        spec: ProvisioningSpec | None = None
    ) -> None:
        conn = conn or self.conn
        spec = spec or self.provisioning_spec
        self._validate_provisioning(conn, spec)
        
        
    def submit_job(self, conn: PodConnectionInfo | None = None, spec: CommandSpec | None = None) -> str:
        conn = conn or self.conn
        spec = spec or self.command_spec
        return self.launch_tmux_job(conn, spec)

    def stop(self, pod_id: str | None = None) -> None:
        pod_id = pod_id or (self.conn.pod_id if self.conn else None)
        if pod_id is None:
            raise ValueError("No pod_id provided and no active connection available.")
        self.stop_pod(
            pod_id, retry_policy=self.workflow.retry_policy
        )

    def resume(self, pod_id: str | None = None) -> PodConnectionInfo:
        pod_id = pod_id or (self.conn.pod_id if self.conn else None)
        if pod_id is None:
            raise ValueError("No pod_id provided and no active connection available.")
        self.resume_pod(pod_id)
        return self.wait_for_ssh_ready(
            pod_id,
            timeout_s=self.workflow.timeout_s,
            poll_s=self.workflow.poll_s,
            retry_policy=self.workflow.retry_policy,
        )

    def terminate(self, pod_id: str | None = None) -> None:
        pod_id = pod_id or (self.conn.pod_id if self.conn else None)
        if pod_id is None:
            raise ValueError("No pod_id provided and no active connection available.")
        self.terminate_pod(
            pod_id, retry_policy=self.workflow.retry_policy
        )

    def run(self) -> PodConnectionInfo:
        try:
            self.conn = self.create()
            self.provision(self.conn)
            self.submit_job(self.conn)

            if self.workflow.terminate_after_launch:
                self.terminate(self.conn.pod_id)

            return self.conn
        except Exception:
            if self.conn is not None and self.workflow.terminate_on_failure:
                self.terminate(self.conn.pod_id)
            raise

    # -- convenience --
    def set_ssh_client(self) -> None:
        self.ssh = SSHClient(self.identity_file)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> PodOrchestrator:
        config = PodOrchestratorConfig.from_yaml(path)
        return cls(config=config)