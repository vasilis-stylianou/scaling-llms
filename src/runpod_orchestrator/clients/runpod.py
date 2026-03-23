from __future__ import annotations

import os
from typing import Any

import runpod

from runpod_orchestrator.exceptions import RunPodError


class RunPodClient:
    """
    RunPodClient is a wrapper around the runpod Python SDK that provides methods for managing pods. 
    It handles API key configuration and provides error handling for unexpected responses.
    """
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise RunPodError("RUNPOD_API_KEY is not set")
        runpod.api_key = self.api_key

    def list_pods(self) -> list[dict[str, Any]]:
        result = runpod.get_pods()
        if not isinstance(result, list):
            raise RunPodError(f"Unexpected get_pods() response: {result!r}")
        return result

    def find_pod_by_name(self, name: str) -> dict[str, Any] | None:
        for pod in self.list_pods():
            if pod.get("name") == name:
                return pod
        return None

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        for pod in self.list_pods():
            if pod.get("id") == pod_id:
                return pod
        raise RunPodError(f"Pod not found: {pod_id}")

    def create_pod(
        self,
        *,
        name: str,
        image_name: str,
        gpu_type_id: str,
        cloud_type: str,
        container_disk_in_gb: int,
        volume_in_gb: int,
        ports: str,
        env: dict[str, str],
        docker_args: str,
    ) -> dict[str, Any]:
        result = runpod.create_pod(
            name=name,
            image_name=image_name,
            gpu_type_id=gpu_type_id,
            cloud_type=cloud_type,
            container_disk_in_gb=container_disk_in_gb,
            volume_in_gb=volume_in_gb,
            ports=ports,
            env=env,
            docker_args=docker_args,
        )
        if not isinstance(result, dict):
            raise RunPodError(f"Unexpected create_pod() response: {result!r}")
        if result.get("id") is None:
            raise RunPodError(f"RunPod create_pod() failed: {result!r}")
        return result

    def stop_pod(self, pod_id: str) -> Any:
        return runpod.stop_pod(pod_id)

    def resume_pod(self, pod_id: str, gpu_count: int = 1) -> Any:
        if hasattr(runpod, "resume_pod"):
            return runpod.resume_pod(pod_id, gpu_count=gpu_count)
        if hasattr(runpod, "start_pod"):
            return runpod.start_pod(pod_id, gpu_count=gpu_count)
        raise RunPodError("RunPod SDK does not expose resume_pod/start_pod")

    def terminate_pod(self, pod_id: str) -> Any:
        return runpod.terminate_pod(pod_id)