from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import runpod
import yaml

from runpod_orchestrator.models import PodConnectionInfo, PodSpec, RetryPolicy
from runpod_orchestrator import ssh as ssh_utils

T = TypeVar("T")
logger = logging.getLogger(__name__)


def configure_api_key_from_env() -> None:
    """Set the RunPod API key from the environment into the RunPod client.

    This reads `RUNPOD_API_KEY` and configures the `runpod` client accordingly.
    """

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is not set.")
    runpod.api_key = api_key


def load_pod_spec(path: Path) -> PodSpec:
    """Load a pod specification YAML file and return a populated PodSpec.

    The function parses the given YAML file and maps keys to a `PodSpec`.
    """

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PodSpec(
        name=data["name"],
        image_name=data["image_name"],
        gpu_type_id=data["gpu_type_id"],
        cloud_type=data.get("cloud_type", "SECURE"),
        container_disk_in_gb=data.get("container_disk_in_gb", 20),
        volume_in_gb=data.get("volume_in_gb", 40),
        ports=data.get("ports", "22/tcp"),
        identity_file=data.get("identity_file", "~/.ssh/runpod_key"),
        env={str(k): str(v) for k, v in (data.get("env", {}) or {}).items()},
    )


def load_bootstrap_as_docker_args(path: Path) -> str:
    """Encode a bootstrap script file as docker `docker_args` for pod creation.

    The script is base64-encoded and returned as a shell command string
    suitable for passing to the pod's docker args.
    """

    script = path.read_text(encoding="utf-8")
    encoded = base64.b64encode(script.encode("utf-8")).decode("utf-8")
    return (
        "bash -lc "
        f"'echo {encoded} | base64 --decode > /tmp/bootstrap.sh "
        "&& bash /tmp/bootstrap.sh'"
    )


def with_retries(
    fn: Callable[..., T],
    *args: Any,
    retry_policy: RetryPolicy,
    **kwargs: Any,
) -> T:
    """Execute `fn` with retry/backoff according to the provided policy.

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
            logger.info("Sleeping %.1fs before retry", sleep_s)
            time.sleep(sleep_s)

    assert last_err is not None
    raise last_err


def list_pods(*, retry_policy: RetryPolicy) -> list[dict[str, Any]]:
    """Return the list of pods from the RunPod API, honoring retries.

    Wraps the API call with retry semantics and validates the response shape.
    """

    pods = with_retries(runpod.get_pods, retry_policy=retry_policy)
    if not isinstance(pods, list):
        raise RuntimeError(f"Unexpected get_pods() response: {pods}")
    return pods


def get_pod_by_id(pod_id: str, *, retry_policy: RetryPolicy) -> dict[str, Any]:
    """Fetch a single pod by its id from the RunPod API with retries.

    Raises a `RuntimeError` if the response is missing or malformed.
    """

    pod = with_retries(runpod.get_pod, pod_id, retry_policy=retry_policy)

    if pod is None:
        raise RuntimeError(f"Pod {pod_id} not found")

    if not isinstance(pod, dict):
        raise RuntimeError(f"Unexpected get_pod() response: {pod}")

    return pod


def find_pod_by_name(name: str, *, retry_policy: RetryPolicy) -> dict[str, Any] | None:
    """Locate a pod by its name by scanning the pod list.

    Returns the matching pod dict when found, or `None` if no match exists.
    """

    for pod in list_pods(retry_policy=retry_policy):
        if pod.get("name") == name:
            return pod
    return None


def create_pod(
    spec: PodSpec,
    docker_args: str,
    *,
    retry_policy: RetryPolicy,
) -> dict[str, Any]:
    """Create a new pod via the RunPod API and return the created pod object.

    Applies the provided retry policy and validates the returned payload.
    """

    pod = with_retries(
        runpod.create_pod,
        name=spec.name,
        image_name=spec.image_name,
        gpu_type_id=spec.gpu_type_id,
        cloud_type=spec.cloud_type,
        container_disk_in_gb=spec.container_disk_in_gb,
        volume_in_gb=spec.volume_in_gb,
        ports=spec.ports,
        docker_args=docker_args,
        env={str(k): str(v) for k, v in spec.env.items()},
        retry_policy=retry_policy,
    )
    if not isinstance(pod, dict):
        raise RuntimeError(f"Unexpected create_pod() response: {pod}")
    return pod


def stop_pod(pod_id: str, *, retry_policy: RetryPolicy) -> None:
    """Request a pod stop operation via the RunPod API with retries.

    Delegates to the RunPod `stop_pod` API and applies the retry policy.
    """

    with_retries(runpod.stop_pod, pod_id, retry_policy=retry_policy)


def _extract_api_error(response: Any) -> str | None:
    """Return an extracted error message from a RunPod API response, if any.

    Examines common keys (`error`, `errors`, `message`) and returns a string
    when an error-like field is present, otherwise `None`.
    """

    if isinstance(response, dict):
        if response.get("error"):
            return str(response.get("error"))
        if response.get("errors"):
            return str(response.get("errors"))
        if response.get("message") and str(response.get("message")).lower().startswith("error"):
            return str(response.get("message"))

    # RunPod terminate may return None / empty-ish response even when the request
    # has been accepted or the pod is already disappearing.
    return None


def _is_terminal_pod_state(pod: dict[str, Any]) -> bool:
    """Return whether the provided pod dict reflects a terminal lifecycle state.

    Considers both desired status and runtime status when determining terminality.
    """

    terminal_states = {"exited", "stopped", "terminated"}

    info = extract_connection_info(pod)
    desired = (info.desired_status or "").lower()
    runtime = (info.runtime_status or "").lower()

    return desired in terminal_states or runtime in terminal_states


def _pod_missing_from_list(
    pod_id: str,
    *,
    retry_policy: RetryPolicy,
) -> bool:
    """Check whether the pod id is absent from the RunPod pod listing.

    Returns True when the pod id does not appear in the list returned by the API.
    """

    pods = list_pods(retry_policy=retry_policy)
    return not any((p.get("id") or p.get("_id")) == pod_id for p in pods)


def terminate_pod(pod_id: str, *, retry_policy: RetryPolicy) -> None:
    """Request pod termination and verify the pod becomes terminal or absent.

    Sends the terminate request and polls the API until the pod is gone or in a
    terminal state, applying the provided retry policy.
    """

    last_exc: Exception | None = None

    # Phase 1: send terminate request
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            response = runpod.terminate_pod(pod_id)
            api_error = _extract_api_error(response)

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

            logger.info("Terminate request accepted for pod %s: %r", pod_id, response)
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
        float(retry_policy.max_attempts) * max(1.0, retry_policy.retry_base_s) * 4.0,
    )

    while time.time() < verify_deadline:
        try:
            pod = get_pod_by_id(
                pod_id,
                retry_policy=RetryPolicy(max_attempts=1, retry_base_s=0),
            )

            if _is_terminal_pod_state(pod):
                logger.info("Pod %s reached terminal state", pod_id)
                return

            if _pod_missing_from_list(
                pod_id,
                retry_policy=RetryPolicy(max_attempts=1, retry_base_s=0),
            ):
                logger.info("Pod %s no longer appears in pod list", pod_id)
                return

        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "does not exist" in message:
                logger.info("Pod %s no longer retrievable after terminate", pod_id)
                return
            last_exc = exc

        time.sleep(max(1.0, retry_policy.retry_base_s))

    if last_exc is not None:
        raise RuntimeError(
            f"Terminate was requested for pod {pod_id}, but it never became absent "
            f"or terminal before timeout. Last error: {last_exc}"
        )

    raise TimeoutError(
        f"Terminate was requested for pod {pod_id}, but it never became absent "
        "or terminal before timeout."
    )


def _extract_runtime_status(pod: dict[str, Any]) -> str | None:
    """Extract a runtime status string from a pod response when available.

    Prefers the runtime `status` field but falls back to `desiredStatus` when
    runtime status is not present.
    """

    runtime = pod.get("runtime") or {}
    if isinstance(runtime, dict) and runtime.get("status"):
        return str(runtime["status"])
    desired = pod.get("desiredStatus") or pod.get("desired_status")
    return str(desired) if desired is not None else None


def _extract_ip(pod: dict[str, Any]) -> str | None:
    """Derive a public IP address for a pod from several possible response fields.

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


def _extract_ssh_port_from_ports(ports: list[Any]) -> int | None:
    """Find the SSH public port (host/public) corresponding to private port 22.

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


def extract_connection_info(pod: dict[str, Any]) -> PodConnectionInfo:
    """Convert a raw pod response dict into a `PodConnectionInfo` object.

    Extracts id, name, desired/runtime status, public IP and SSH port.
    """

    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []

    return PodConnectionInfo(
        pod_id=str(pod.get("id") or pod.get("_id") or ""),
        name=str(pod.get("name") or ""),
        desired_status=(
            str(pod.get("desiredStatus") or pod.get("desired_status"))
            if (pod.get("desiredStatus") or pod.get("desired_status")) is not None
            else None
        ),
        runtime_status=_extract_runtime_status(pod),
        public_ip=_extract_ip(pod),
        ssh_port=_extract_ssh_port_from_ports(ports),
    )


def wait_for_ssh_ready(
    pod_id: str,
    *,
    timeout_s: int,
    poll_s: int,
    retry_policy: RetryPolicy,
    identity_file: str,
) -> PodConnectionInfo:
    """Wait until the pod's SSH becomes reachable and return its connection info.

    Polls the RunPod API until the pod reports SSH-ready ports and an SSH
    probe succeeds within the provided timeout.
    """

    deadline = time.time() + timeout_s
    last_info: PodConnectionInfo | None = None

    while time.time() < deadline:
        pod = get_pod_by_id(pod_id, retry_policy=retry_policy)
        info = extract_connection_info(pod)
        last_info = info

        if info.is_ssh_ready and ssh_utils.probe_ssh_connectivity(
            info,
            identity_file=identity_file,
        ):
            return info

        time.sleep(poll_s)

    raise TimeoutError(f"Timed out waiting for pod SSH readiness. Last info: {last_info}")


def resolve_or_create_pod(
    spec: PodSpec,
    docker_args: str,
    *,
    reuse_if_exists: bool,
    timeout_s: int,
    poll_s: int,
    retry_policy: RetryPolicy,
) -> PodConnectionInfo:
    """Create a pod from `spec` or reuse an existing one, returning SSH info.

    Either reuses a visible pod matching the name (and waits for SSH) or
    creates a new pod and waits for SSH readiness before returning connection
    details.
    """

    existing = None
    if reuse_if_exists:
        existing = wait_until_pod_visible_by_name(
            spec.name,
            timeout_s=min(timeout_s, 60),
            poll_s=max(1, poll_s),
            retry_policy=retry_policy,
        )

    if existing and reuse_if_exists:
        pod_id = existing.get("id") or existing.get("_id")
        if not pod_id:
            raise RuntimeError(f"Existing pod missing id: {existing}")

        info = extract_connection_info(existing)
        if info.is_ssh_ready:
            return info

        return wait_for_ssh_ready(
            str(pod_id),
            timeout_s=timeout_s,
            poll_s=poll_s,
            retry_policy=retry_policy,
            identity_file=spec.expanded_identity_file,
        )

    created = create_pod(spec, docker_args=docker_args, retry_policy=retry_policy)
    pod_id = created.get("id") or created.get("_id")

    if not pod_id:
        found = find_pod_by_name(spec.name, retry_policy=retry_policy)
        if not found:
            raise RuntimeError(f"Could not determine pod id after create: {created}")
        pod_id = found.get("id") or found.get("_id")

    return wait_for_ssh_ready(
        str(pod_id),
        timeout_s=timeout_s,
        poll_s=poll_s,
        retry_policy=retry_policy,
        identity_file=spec.expanded_identity_file,
    )


def wait_until_pod_visible_by_name(
    pod_name: str,
    *,
    timeout_s: int,
    poll_s: int,
    retry_policy: RetryPolicy,
) -> dict[str, Any] | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = find_pod_by_name(pod_name, retry_policy=retry_policy)
        if pod is not None:
            return pod
        time.sleep(poll_s)
    return None


def get_connection_info_for_pod(
    pod_id: str,
    *,
    timeout_s: int,
    poll_s: int,
    retry_policy: RetryPolicy,
    identity_file: str,
) -> PodConnectionInfo:
    return wait_for_ssh_ready(
        pod_id,
        timeout_s=timeout_s,
        poll_s=poll_s,
        retry_policy=retry_policy,
        identity_file=identity_file,
    )
