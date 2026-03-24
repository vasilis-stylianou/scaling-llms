from __future__ import annotations


import logging
import time
from typing import Any, Callable, TypeVar

from runpod_orch.clients import RunPodClient
from runpod_orch.clients import SSHClient
from runpod_orch.exceptions import PodNotReadyError, RunPodError
from runpod_orch.specs import PodConnectionInfo, PodSpec, RetryPolicy


T = TypeVar("T")
logger = logging.getLogger(__name__)



def with_retries(
    fn: Callable[..., T],
    *args: Any,
    retry_policy: RetryPolicy,
    **kwargs: Any,
) -> T:
    """
    Execute `fn` with retry/backoff according to the provided policy.

    Retries the callable on exception, applying exponential backoff until
    `retry_policy.max_attempts` is exhausted.
    """
    last_err: Exception | None = None
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            if attempt == retry_policy.max_attempts:
                break
            sleep_s = retry_policy.retry_base_s * (2 ** (attempt - 1))
            logger.warning(
                "Retrying %s after failure (%s/%s): %s",
                fn.__name__,
                attempt,
                retry_policy.max_attempts,
                exc,
            )
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


class PodManager:
    """Manages pod lifecycle: create, stop, resume, terminate, and readiness polling."""

    def __init__(self, client: RunPodClient, spec: PodSpec) -> None:
        self.runpod_client = client
        self.spec = spec

    # --- POD READINESS ---
    def wait_for_ssh_ready(
        self,
        pod_id: str,
        *,
        timeout_s: int,
        poll_s: int,
        retry_policy: RetryPolicy,
        identity_file: str | None = None,
    ) -> PodConnectionInfo:
        """
        Wait until the pod's SSH becomes reachable and return its connection info.

        Polls the RunPod API until the pod reports SSH-ready ports and an SSH
        probe succeeds within the provided timeout.
        """
        
        deadline = time.time() + timeout_s
        last_info: PodConnectionInfo | None = None
        identity = identity_file or "~/.ssh/runpod_key"

        while time.time() < deadline:
            pod = self._get_pod_by_id(pod_id, retry_policy=retry_policy)
            info = self._extract_connection_info(pod)
            last_info = info

            if info.is_ssh_ready:
                ssh = SSHClient(identity)
                if ssh.probe_connectivity(info):
                    return info

            time.sleep(poll_s)

        raise PodNotReadyError(
            f"Timed out waiting for pod SSH readiness. Last info: {last_info}"
        )

    # --- LIFECYCLE ---
    def create_pod(
        self,
        docker_args: str,
        *,
        retry_policy: RetryPolicy,
    ) -> dict[str, Any]:
        """
        Create a new pod via the RunPod API and return the created pod object.

        Applies the provided retry policy and validates the returned payload.
        """
        pod = with_retries(
            self.runpod_client.create_pod,
            name=self.spec.name,
            image_name=self.spec.image_name,
            gpu_type_id=self.spec.gpu_type_id,
            cloud_type=self.spec.cloud_type,
            container_disk_in_gb=self.spec.container_disk_in_gb,
            volume_in_gb=self.spec.volume_in_gb,
            ports=self.spec.ports,
            env=self.spec.env,
            docker_args=docker_args,
            retry_policy=retry_policy,
        )
        if not isinstance(pod, dict):
            raise RuntimeError(f"Unexpected create_pod() response: {pod}")
        return pod
    
    def stop_pod(
        self,
        pod_id: str,
        retry_policy: RetryPolicy = RetryPolicy(),
    ) -> None:
        """
        Request a pod stop operation via the RunPod API with retries.

        Delegates to the RunPod `stop_pod` API and applies the retry policy.
        """

        with_retries(self.runpod_client.stop_pod, pod_id, retry_policy=retry_policy)

    def resume_pod(self, pod_id: str) -> None:
        try:
            self.runpod_client.resume_pod(pod_id)
        except Exception as exc:
            raise RunPodError(f"Failed to resume pod {pod_id}") from exc

    def terminate_pod(
        self,
        pod_id: str,
        retry_policy: RetryPolicy = RetryPolicy(),
    ) -> None:
        """
        Request pod termination and verify the pod becomes terminal or absent.

        Sends the terminate request and polls the API until the pod is gone or in a
        terminal state, applying the provided retry policy.
        """
            
        last_exc: Exception | None = None

        # Phase 1: send terminate request
        for attempt in range(1, retry_policy.max_attempts + 1):
            try:
                response = self.runpod_client.terminate_pod(pod_id)
                api_error = self._extract_api_error(response)

                if api_error:
                    msg = api_error.lower()
                    if "not found" in msg or "does not exist" in msg:
                        logger.info(
                            "Terminate API says pod %s is already absent: %s",
                            pod_id,
                            api_error,
                        )
                        break
                    raise RuntimeError(api_error)

                logger.info(
                    "Terminate request accepted for pod %s: %r",
                    pod_id,
                    response,
                )
                break

            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()

                if "not found" in message or "does not exist" in message:
                    logger.info(
                        "Terminate call says pod %s is already absent: %s",
                        pod_id,
                        exc,
                    )
                    break

                if attempt == retry_policy.max_attempts:
                    raise

                sleep_s = retry_policy.retry_base_s * (2 ** (attempt - 1))
                logger.warning(
                    "Terminate retry for pod %s (%s/%s): %s",
                    pod_id,
                    attempt,
                    retry_policy.max_attempts,
                    exc,
                )
                time.sleep(sleep_s)

        # Phase 2: verify pod is terminal or gone
        verify_deadline = time.time() + max(
            60.0,
            float(retry_policy.max_attempts)
            * max(1.0, retry_policy.retry_base_s)
            * 4.0,
        )
        no_retry = RetryPolicy(max_attempts=1, retry_base_s=0)

        while time.time() < verify_deadline:
            try:
                pod = self._get_pod_by_id(pod_id, retry_policy=no_retry)
                if self._is_terminal_pod_state(pod):
                    logger.info("Pod %s reached terminal state", pod_id)
                    return

                if self._pod_missing_from_list(pod_id, retry_policy=no_retry):
                    logger.info(
                        "Pod %s no longer appears in pod list", pod_id
                    )
                    return

            except Exception as exc:
                message = str(exc).lower()
                if "not found" in message or "does not exist" in message:
                    logger.info(
                        "Pod %s no longer retrievable after terminate", pod_id
                    )
                    return
                last_exc = exc

            time.sleep(max(1.0, retry_policy.retry_base_s))

        if last_exc is not None:
            raise RuntimeError(
                f"Terminate was requested for pod {pod_id}, but it never became "
                f"absent or terminal before timeout. Last error: {last_exc}"
            )

        raise TimeoutError(
            f"Terminate was requested for pod {pod_id}, but it never became "
            "absent or terminal before timeout."
        )
    
    def resolve_or_create_pod(
        self,
        *,
        docker_args: str,
        reuse_if_exists: bool,
        timeout_s: int = 60,
        poll_s: int = 5,
        retry_policy: RetryPolicy = RetryPolicy(),
    ) -> PodConnectionInfo:
        """
        Create a pod from `spec` or reuse an existing one, returning SSH info.

        Either reuses a visible pod matching the name (and waits for SSH) or
        creates a new pod and waits for SSH readiness before returning connection
        details.
        """
        existing: dict[str, Any] | None = None
        if reuse_if_exists:
            existing = self._wait_until_pod_visible_by_name(
                self.spec.name,
                timeout_s=min(timeout_s, 60),
                poll_s=max(1, poll_s),
                retry_policy=retry_policy,
            )

        if existing and reuse_if_exists:
            pod_id = existing.get("id") or existing.get("_id")
            if not pod_id:
                raise RuntimeError(f"Existing pod missing id: {existing}")
            info = self._extract_connection_info(existing)
            if info.is_ssh_ready:
                return info

            return self.wait_for_ssh_ready(
                str(pod_id),
                timeout_s=timeout_s,
                poll_s=poll_s,
                retry_policy=retry_policy,
                identity_file=self.spec.expanded_identity_file,
            )

        created = self.create_pod(
            docker_args=docker_args,
            retry_policy=retry_policy,
        )
        pod_id = created.get("id") or created.get("_id")

        if not pod_id:
            found = self._find_pod_by_name(self.spec.name, retry_policy=retry_policy)
            if not found:
                raise RuntimeError(
                    f"Could not determine pod id after create: {created}"
                )
            pod_id = found.get("id") or found.get("_id")

        return self.wait_for_ssh_ready(
            str(pod_id),
            timeout_s=timeout_s,
            poll_s=poll_s,
            retry_policy=retry_policy,
            identity_file=self.spec.expanded_identity_file,
        )

    
    # TODO
    # def get_connection_info(self, pod_id: str) -> PodConnectionInfo:
    #     pod = self.client.get_pod(pod_id)
    #     return self._extract_connection_info(pod)

    # --- INTERNAL HELPERS ---

    def _find_pod_by_name(
        self,
        name: str,
        *,
        retry_policy: RetryPolicy,
    ) -> dict[str, Any] | None:
        """
        Locate a pod by its name by scanning the pod list.

        Returns the matching pod dict when found, or `None` if no match exists.
        """
        pods = with_retries(
            self.runpod_client.list_pods, retry_policy=retry_policy
        )
        for pod in pods:
            if pod.get("name") == name:
                return pod
        return None
    
    def _get_pod_by_id(self, pod_id: str, *, retry_policy: RetryPolicy) -> dict[str, Any]:
        """Fetch a single pod by its id from the RunPod API with retries.

        Raises a `RuntimeError` if the response is missing or malformed.
        """

        pod = with_retries(self.runpod_client.get_pod, pod_id, retry_policy=retry_policy)

        if pod is None:
            raise RuntimeError(f"Pod {pod_id} not found")

        if not isinstance(pod, dict):
            raise RuntimeError(f"Unexpected get_pod() response: {pod}")

        return pod
    
    def _wait_until_pod_visible_by_name(
        self,
        pod_name: str,
        *,
        timeout_s: int,
        poll_s: int,
        retry_policy: RetryPolicy,
    ) -> dict[str, Any] | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            pod = self._find_pod_by_name(pod_name, retry_policy=retry_policy)
            if pod is not None:
                return pod
            time.sleep(poll_s)
        return None

    def _pod_missing_from_list(
        self,
        pod_id: str,
        *,
        retry_policy: RetryPolicy,
    ) -> bool:
        pods = with_retries(
            self.runpod_client.list_pods, retry_policy=retry_policy
        )
        return not any(
            (p.get("id") or p.get("_id")) == pod_id for p in pods
        )

    @staticmethod
    def _extract_api_error(response: Any) -> str | None:
        """
        Return an extracted error message from a RunPod API response, if any.

        Examines common keys (`error`, `errors`, `message`) and returns a string
        when an error-like field is present, otherwise `None`.
        """
        if isinstance(response, dict):
            if response.get("error"):
                return str(response["error"])
            if response.get("errors"):
                return str(response["errors"])
            if response.get("message") and str(
                response["message"]
            ).lower().startswith("error"):
                return str(response["message"])
        return None

    @staticmethod
    def _is_terminal_pod_state(pod: dict[str, Any]) -> bool:
        """
        Return whether the provided pod dict reflects a terminal lifecycle state.

        Considers both desired status and runtime status when determining terminality.
        """
        terminal_states = {"exited", "stopped", "terminated"}
        
        info = PodManager._extract_connection_info(pod)
        desired = (info.desired_status or "").lower()
        runtime = (info.runtime_status or "").lower()
        
        return desired in terminal_states or runtime in terminal_states

    @staticmethod
    def _extract_runtime_status(pod: dict[str, Any]) -> str | None:
        """
        Extract a runtime status string from a pod response when available.

        Prefers the runtime `status` field but falls back to `desiredStatus` when
        runtime status is not present.
        """
        runtime = pod.get("runtime") or {}
        if isinstance(runtime, dict) and runtime.get("status"):
            return str(runtime["status"])
        desired = pod.get("desiredStatus") or pod.get("desired_status")
        return str(desired) if desired is not None else None

    @staticmethod
    def _extract_connection_info(pod: dict[str, Any]) -> PodConnectionInfo:
        """
        Convert a raw pod response dict into a `PodConnectionInfo` object.

        Extracts id, name, desired/runtime status, public IP and SSH port.
        """
        
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []

        return PodConnectionInfo(
            pod_id=str(pod.get("id") or pod.get("_id") or ""),
            name=str(pod.get("name") or ""),
            desired_status=(
                str(pod.get("desiredStatus") or pod.get("desired_status"))
                if (
                    pod.get("desiredStatus") or pod.get("desired_status")
                )
                is not None
                else None
            ),
            runtime_status=PodManager._extract_runtime_status(pod),
            public_ip=PodManager._extract_ip(pod),
            ssh_port=PodManager._extract_ssh_port(ports),
        )

    @staticmethod
    def _extract_ip(pod: dict[str, Any]) -> str | None:
        """
        Derive a public IP address for a pod from several possible response fields.

        Examines runtime ports and common top-level keys to locate a public IP.
        """
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []

        if isinstance(ports, list):
            for entry in ports:
                if not isinstance(entry, dict):
                    continue
                if entry.get("isIpPublic") is True and entry.get("ip"):
                    return str(entry["ip"])

        for key in ("ip", "publicIp", "public_ip"):
            if pod.get(key):
                return str(pod[key])

        return None

    @staticmethod
    def _extract_ssh_port(ports: list[Any]) -> int | None:
        """
        Find the SSH public port (host/public) corresponding to private port 22.

        Scans a list of port entries and returns the host/public port mapped to
        the container's port 22 when present.
        """
        
        for entry in ports:
            if not isinstance(entry, dict):
                continue

            private_port = entry.get("privatePort") or entry.get("private_port")
            if str(private_port) != "22":
                continue

            port = (
                entry.get("publicPort")
                or entry.get("hostPort")
                or entry.get("public_port")
                or entry.get("port")
            )
            if port is not None:
                return int(port)

        return None
