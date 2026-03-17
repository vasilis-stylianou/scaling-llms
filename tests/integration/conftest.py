from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from runpod_orchestrator.lifecycle import (
    configure_api_key_from_env,
    get_pod_by_id,
    load_bootstrap_as_docker_args,
    load_pod_spec,
    terminate_pod,
)
from runpod_orchestrator.models import PodSpec, RetryPolicy


logger = logging.getLogger(__name__)


def _pod_state_summary(pod: dict) -> str:
    """Return a short summary string describing a pod's desired and runtime state.

    Produces a compact textual summary useful for debugging cleanup visibility.
    """

    desired = str(pod.get("desiredStatus") or pod.get("desired_status") or "")
    runtime = pod.get("runtime") or {}
    runtime_status = str(runtime.get("status") or pod.get("status") or "")
    return f"desired={desired or 'unknown'}, runtime={runtime_status or 'unknown'}"


class IntegrationConfig:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.repo_root = repo_root
        self.pod_spec_path = Path(
            os.getenv(
                "RUNPOD_TEST_POD_SPEC",
                str(repo_root / "src" / "runpod_orchestrator" / "pod_spec.yaml"),
            )
        )
        self.bootstrap_script_path = Path(
            os.getenv(
                "RUNPOD_TEST_BOOTSTRAP_SCRIPT",
                str(repo_root / "src" / "runpod_orchestrator" / "bootstrap.sh"),
            )
        )
        self.timeout_s = int(os.getenv("RUNPOD_TEST_TIMEOUT_S", "900"))
        self.poll_s = int(os.getenv("RUNPOD_TEST_POLL_S", "5"))
        self.max_attempts = int(os.getenv("RUNPOD_TEST_MAX_ATTEMPTS", "5"))
        self.retry_base_s = float(os.getenv("RUNPOD_TEST_RETRY_BASE_S", "2.0"))
        self.run_up_workflow = os.getenv("RUNPOD_E2E_RUN_UP", "0") == "1"
        self.up_repo_url = os.getenv(
            "RUNPOD_TEST_REPO_URL", 
            "https://github.com/vasilis-stylianou/scaling-llms.git"
        ).strip()
        self.rclone_config_local = os.getenv(
            "RUNPOD_TEST_RCLONE_CONFIG",
            str(Path.home() / ".config" / "rclone" / "rclone.conf"),
        ).strip()

    @property
    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_attempts=self.max_attempts,
            retry_base_s=self.retry_base_s,
        )


@pytest.fixture(scope="session", autouse=True)
def _gate_integration_tests() -> None:
    enabled = os.getenv("RUNPOD_E2E") == "1"
    if not enabled:
        pytest.skip("Integration tests are disabled. Set RUNPOD_E2E=1 to enable.")

    if not os.getenv("RUNPOD_API_KEY"):
        pytest.skip("RUNPOD_API_KEY is required for integration tests.")

    configure_api_key_from_env()


@pytest.fixture(scope="session")
def integration_config() -> IntegrationConfig:
    cfg = IntegrationConfig()
    if not cfg.pod_spec_path.exists():
        pytest.skip(f"Pod spec file not found: {cfg.pod_spec_path}")
    if not cfg.bootstrap_script_path.exists():
        pytest.skip(f"Bootstrap script not found: {cfg.bootstrap_script_path}")
    return cfg


@pytest.fixture(scope="session")
def base_pod_spec(integration_config: IntegrationConfig) -> PodSpec:
    return load_pod_spec(integration_config.pod_spec_path)


@pytest.fixture(scope="session")
def docker_args(integration_config: IntegrationConfig) -> str:
    return load_bootstrap_as_docker_args(integration_config.bootstrap_script_path)


@pytest.fixture
def unique_pod_spec(base_pod_spec: PodSpec) -> PodSpec:
    suffix = f"e2e-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    return replace(base_pod_spec, name=f"{base_pod_spec.name}-{suffix}")


@pytest.fixture
def cleanup_pod(integration_config: IntegrationConfig):
    terminated: set[str] = set()

    def _cleanup(pod_id: str) -> None:
        if pod_id in terminated:
            return
        try:
            terminate_pod(pod_id, retry_policy=integration_config.retry_policy)
            try:
                pod = get_pod_by_id(
                    pod_id,
                    retry_policy=RetryPolicy(max_attempts=1, retry_base_s=0),
                )
                logger.info(
                    "Cleanup terminate observed pod still visible: id=%s %s",
                    pod_id,
                    _pod_state_summary(pod),
                )
            except Exception as verify_exc:
                verify_message = str(verify_exc).lower()
                if "not found" in verify_message or "does not exist" in verify_message:
                    logger.info("Cleanup terminate confirmed pod absent: id=%s", pod_id)
                else:
                    logger.warning(
                        "Cleanup terminate verification ambiguous for pod %s: %s",
                        pod_id,
                        verify_exc,
                    )
            terminated.add(pod_id)
        except Exception as exc:  # pragma: no cover - explicit cleanup safeguard
            message = str(exc).lower()
            if "not found" in message or "does not exist" in message:
                terminated.add(pod_id)
                return
            raise AssertionError(f"Failed to terminate pod {pod_id}: {exc}") from exc

    return _cleanup


@pytest.fixture
def pod_tracker(cleanup_pod):
    created: list[str] = []

    def _track(pod_id: str) -> None:
        if pod_id and pod_id not in created:
            created.append(pod_id)

    yield _track

    for pod_id in reversed(created):
        cleanup_pod(pod_id)


@pytest.fixture
def get_pod(integration_config: IntegrationConfig):
    def _get_pod(pod_id: str) -> dict:
        return get_pod_by_id(pod_id, retry_policy=integration_config.retry_policy)

    return _get_pod
