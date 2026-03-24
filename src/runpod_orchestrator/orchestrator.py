from __future__ import annotations

import base64
from pathlib import Path

from runpod_orch.clients.runpod import RunPodClient
from runpod_orch.clients.ssh import SSHClient
from runpod_orch.config import PodOrchestratorConfig
from runpod_orch.specs import PodConnectionInfo
from runpod_orch.services import (
    PodJobLauncher,
    PodManager,
    PodProvisioner,
)

class PodOrchestrator:
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

    def __init__(
        self,
        config: PodOrchestratorConfig,
        runpod_client: RunPodClient | None = None,
        pod_manager: PodManager | None = None,
    ) -> None:
        self.config = config
        self._runpod_client = runpod_client or RunPodClient()
        self.pod_manager = pod_manager or PodManager(self._runpod_client, self.config.pod_spec)

        self.conn: PodConnectionInfo | None = None

    # -- high-level operations --
    def create(self) -> PodConnectionInfo:
        docker_args = self.load_bootstrap_as_docker_args(self.config.bootstrap_script_path)
        try:
            self.conn = self.pod_manager.resolve_or_create_pod(
                docker_args=docker_args,
                reuse_if_exists=self.config.workflow.reuse_if_exists,
                timeout_s=self.config.workflow.timeout_s,
                poll_s=self.config.workflow.poll_s,
                retry_policy=self.config.workflow.retry_policy,
            )
            return self.conn
        except Exception:
            if self.conn is not None and self.config.workflow.terminate_on_failure:
                try:
                    self.terminate(self.conn.pod_id)
                except Exception:
                    pass
            raise

    def provision(self, conn: PodConnectionInfo | None = None) -> None:
        conn = conn or self.conn
        ssh_client = self.make_ssh_client()
        pod_provisioner = PodProvisioner(ssh_client, self.config.provisioning)
        pod_provisioner.copy_rclone_config(conn)
        pod_provisioner.clone_or_update_repo(conn)
        pod_provisioner.poetry_install(conn)
        pod_provisioner.create_jupyter_kernel(conn)
        
        
    def submit(self, conn: PodConnectionInfo | None = None) -> str:
        conn = conn or self.conn
        ssh_client = self.make_ssh_client()
        return PodJobLauncher(ssh_client, self.config.job_launcher_spec).launch_tmux_job(conn)

    def stop(self, pod_id: str) -> None:
        self.pod_manager.stop_pod(
            pod_id, retry_policy=self.config.workflow.retry_policy
        )

    def resume(self, pod_id: str) -> PodConnectionInfo:
        self.pod_manager.resume_pod(pod_id)
        return self.pod_manager.wait_for_ssh_ready(
            pod_id,
            timeout_s=self.config.workflow.timeout_s,
            poll_s=self.config.workflow.poll_s,
            retry_policy=self.config.workflow.retry_policy,
            identity_file=self.config.pod_spec.expanded_identity_file,
        )

    def terminate(self, pod_id: str) -> None:
        self.pod_manager.terminate_pod(
            pod_id, retry_policy=self.config.workflow.retry_policy
        )

    def run(self) -> PodConnectionInfo:
        try:
            self.conn = self.create()
            self.provision(self.conn)
            self.submit(self.conn)

            if self.config.workflow.terminate_after_launch:
                self.terminate(self.conn.pod_id)

            return self.conn
        except Exception:
            if self.conn is not None and self.config.workflow.terminate_on_failure:
                self.terminate(self.conn.pod_id)
            raise

    # -- convenience --
    def make_ssh_client(self) -> SSHClient:
        return SSHClient(self.config.provisioning.expanded_identity_file)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> PodOrchestrator:
        config = PodOrchestratorConfig.from_yaml(path)
        return cls(config=config)

    @staticmethod
    def load_bootstrap_as_docker_args(path: str | Path) -> str:
        """
        Encode a bootstrap script file as docker `docker_args` for pod creation.

        The script is base64-encoded and returned as a shell command string
        suitable for passing to the pod's docker args.
        """
        script = Path(path).expanduser().resolve().read_text(encoding="utf-8")
        encoded = base64.b64encode(script.encode("utf-8")).decode("utf-8")
        return (
            "bash -lc "
            f"'echo {encoded} | base64 --decode > /tmp/bootstrap.sh "
            "&& bash /tmp/bootstrap.sh'"
        )