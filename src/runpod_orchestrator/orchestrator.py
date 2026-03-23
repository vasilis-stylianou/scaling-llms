# runpod_orchestrator/orchestrator.py

from __future__ import annotations

from pathlib import Path

from runpod_orchestrator.clients.runpod import RunPodClient
from runpod_orchestrator.clients.ssh import SSHClient
from runpod_orchestrator.config import OrchestratorConfig
from runpod_orchestrator.specs import RunResult
from runpod_orchestrator.services.job_service import JobService
from runpod_orchestrator.services.pod_service import PodService, load_bootstrap_as_docker_args
from runpod_orchestrator.services.provisioning_service import ProvisioningService


class OrchestratorService:
    """
    Main public API for pod lifecycle orchestration.

    Exposes a scheduler-like interface:
      - create
      - provision
      - submit
      - stop
      - resume
      - terminate
      - run
    """

    def __init__(self, runpod_client: RunPodClient | None = None) -> None:
        self.runpod_client = runpod_client or RunPodClient()
        self.pods = PodService(self.runpod_client)

    def create(self, config: OrchestratorConfig) -> RunResult:
        """
        Creates a pod according to the given config and waits until it's ready for SSH connections.
        """
        docker_args = load_bootstrap_as_docker_args(config.bootstrap_script_path)
        pod_id = self.pods.resolve_or_create_pod(
            config.pod_spec,
            docker_args=docker_args,
            reuse_if_exists=config.workflow.reuse_if_exists,
        )
        conn = self.pods.wait_for_connection_info(
            pod_id,
            timeout_s=config.workflow.timeout_s,
            poll_s=config.workflow.poll_s,
            retry_policy=config.workflow.retry_policy,
        )
        return self._make_run_result(config, conn.pod_id)

    def provision(self, config: OrchestratorConfig, *, pod_id: str) -> RunResult:
        """
        Runs provisioning steps on the given pod according to the config.
        """
        conn = self.pods.wait_for_connection_info(
            pod_id,
            timeout_s=config.workflow.timeout_s,
            poll_s=config.workflow.poll_s,
            retry_policy=config.workflow.retry_policy,
        )
        ssh = SSHClient(config.provisioning.expanded_identity_file)
        ProvisioningService(ssh).setup_pod(conn, config.provisioning)
        return self._make_run_result(config, pod_id)

    def submit(self, config: OrchestratorConfig, *, pod_id: str) -> RunResult:
        """
        Submits the run job to the given pod according to the config.
        """
        conn = self.pods.wait_for_connection_info(
            pod_id,
            timeout_s=config.workflow.timeout_s,
            poll_s=config.workflow.poll_s,
            retry_policy=config.workflow.retry_policy,
        )
        ssh = SSHClient(config.job_spec.expanded_identity_file)
        JobService(ssh).launch_job(conn, config.job_spec)
        return self._make_run_result(config, pod_id)

    def stop(self, pod_id: str) -> None:
        """
        Stops the given pod.
        """
        self.pods.stop_pod(pod_id)

    def resume(self, config: OrchestratorConfig, *, pod_id: str) -> RunResult:
        """
        Resumes the given pod and waits until it's ready for SSH connections.
        """
        self.pods.resume_pod(pod_id)
        return self._make_run_result(
            config,
            self.pods.wait_for_connection_info(
                pod_id,
                timeout_s=config.workflow.timeout_s,
                poll_s=config.workflow.poll_s,
                retry_policy=config.workflow.retry_policy,
            ).pod_id,
        )

    def terminate(self, pod_id: str) -> None:
        """
        Terminates the given pod.
        """
        self.pods.terminate_pod(pod_id)

    def run(self, config: OrchestratorConfig) -> RunResult:
        """
        Runs the full create+provision+submit workflow according to the config and returns the result.
        """
        created = self.create(config)
        try:
            self.provision(config, pod_id=created.pod.pod_id)
            result = self.submit(config, pod_id=created.pod.pod_id)

            if config.workflow.terminate_after_launch:
                self.terminate(result.pod.pod_id)

            return result
        except Exception:
            if config.workflow.terminate_on_failure:
                self.terminate(created.pod.pod_id)
            raise

    def _make_run_result(self, config: OrchestratorConfig, pod_id: str) -> RunResult:
        """
        Creates a RunResult object for the given pod and config.
        """
        conn = self.pods.get_connection_info(pod_id)
        return RunResult(
            pod=conn,
            tmux_session_name=config.job_spec.tmux_session_name,
            log_path=config.job_spec.log_path,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> tuple["OrchestratorService", OrchestratorConfig]:
        """
        Creates an OrchestratorService instance and loads the configuration from a YAML file.
        """
        config = OrchestratorConfig.from_yaml(path)
        return cls(), config