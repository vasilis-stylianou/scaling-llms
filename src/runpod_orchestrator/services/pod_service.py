# runpod_orchestrator/services/pods.py

from __future__ import annotations

import time
from pathlib import Path

from runpod_orchestrator.clients.runpod import RunPodClient
from runpod_orchestrator.exceptions import PodNotReadyError, RunPodError
from runpod_orchestrator.specs import PodConnectionInfo, PodSpec, RetryPolicy


def load_bootstrap_as_docker_args(path: str | Path) -> str:
    script_path = Path(path).expanduser().resolve()
    bootstrap = script_path.read_text(encoding="utf-8")
    escaped = bootstrap.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'/bin/bash -lc "{escaped}"'


class PodService:
    """
    PodService is responsible for managing pods, including creating, resolving, and terminating them.
    It interacts with the RunPodClient to perform these operations and provides methods to wait for
    pod readiness and retrieve connection information.
    """
    def __init__(self, client: RunPodClient) -> None:
        self.client = client

    def resolve_or_create_pod(
        self,
        spec: PodSpec,
        *,
        docker_args: str,
        reuse_if_exists: bool,
    ) -> str:
        if reuse_if_exists:
            existing = self.client.find_pod_by_name(spec.name)
            if existing and existing.get("id"):
                return str(existing["id"])

        created = self.client.create_pod(
            name=spec.name,
            image_name=spec.image_name,
            gpu_type_id=spec.gpu_type_id,
            cloud_type=spec.cloud_type,
            container_disk_in_gb=spec.container_disk_in_gb,
            volume_in_gb=spec.volume_in_gb,
            ports=spec.ports,
            env=spec.env,
            docker_args=docker_args,
        )
        return str(created["id"])

    def wait_for_connection_info(
        self,
        pod_id: str,
        *,
        timeout_s: int,
        poll_s: int,
        retry_policy: RetryPolicy,
    ) -> PodConnectionInfo:
        deadline = time.time() + timeout_s
        attempts = 0
        last_error: Exception | None = None

        while time.time() < deadline:
            attempts += 1
            try:
                pod = self.client.get_pod(pod_id)
                conn = self._to_connection_info(pod)
                if conn.is_ssh_ready:
                    return conn
            except Exception as exc:
                last_error = exc

            if attempts >= retry_policy.max_attempts:
                attempts = 0
            time.sleep(poll_s)

        raise PodNotReadyError(
            f"Pod {pod_id} was not SSH-ready within {timeout_s}s. Last error: {last_error}"
        )

    def get_connection_info(self, pod_id: str) -> PodConnectionInfo:
        pod = self.client.get_pod(pod_id)
        return self._to_connection_info(pod)

    def stop_pod(self, pod_id: str) -> None:
        try:
            self.client.stop_pod(pod_id)
        except Exception as exc:
            raise RunPodError(f"Failed to stop pod {pod_id}") from exc

    def resume_pod(self, pod_id: str) -> None:
        try:
            self.client.resume_pod(pod_id)
        except Exception as exc:
            raise RunPodError(f"Failed to resume pod {pod_id}") from exc

    def terminate_pod(self, pod_id: str) -> None:
        try:
            self.client.terminate_pod(pod_id)
        except Exception as exc:
            raise RunPodError(f"Failed to terminate pod {pod_id}") from exc

    @staticmethod
    def _to_connection_info(pod: dict) -> PodConnectionInfo:
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        public_ip = runtime.get("publicIp")
        ssh_port = None

        for port in ports:
            private_port = port.get("privatePort")
            if private_port == 22 and port.get("publicPort") is not None:
                ssh_port = int(port["publicPort"])
                break

        return PodConnectionInfo(
            pod_id=str(pod["id"]),
            name=str(pod.get("name", "")),
            public_ip=str(public_ip) if public_ip else "",
            ssh_port=int(ssh_port) if ssh_port is not None else 0,
            desired_status=pod.get("desiredStatus"),
            runtime_status=str(runtime.get("status") or pod.get("status") or ""),
        )