from __future__ import annotations

import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from runpod_orchestrator.config import OrchestratorConfig
from runpod_orchestrator.orchestrator import OrchestratorService


class IntegrationConfig:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.repo_root = repo_root
        self.config_path = Path(
            os.getenv(
                "RUNPOD_TEST_CONFIG",
                str(repo_root / "configs" / "runpod" / "runtime.yaml"),
            )
        )

    @property
    def has_config(self) -> bool:
        return self.config_path.exists()


@pytest.fixture(scope="session")
def require_runpod_e2e() -> None:
    if os.getenv("RUNPOD_E2E") != "1":
        pytest.skip("Set RUNPOD_E2E=1 to enable RunPod integration tests.")
    if not os.getenv("RUNPOD_API_KEY"):
        pytest.skip("RUNPOD_API_KEY is required for RunPod integration tests.")


@pytest.fixture(scope="session")
def integration_config(require_runpod_e2e: None) -> IntegrationConfig:
    cfg = IntegrationConfig()
    if not cfg.has_config:
        pytest.skip(f"Orchestrator config file not found: {cfg.config_path}")
    return cfg


@pytest.fixture(scope="session")
def base_config(integration_config: IntegrationConfig) -> OrchestratorConfig:
    return OrchestratorConfig.from_yaml(integration_config.config_path)


@pytest.fixture(scope="session")
def orchestrator() -> OrchestratorService:
    return OrchestratorService()


@pytest.fixture
def unique_config(base_config: OrchestratorConfig) -> OrchestratorConfig:
    suffix = f"e2e-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    pod_spec = replace(
        base_config.pod_spec,
        name=f"{base_config.pod_spec.name}-{suffix}",
    )
    workflow = replace(base_config.workflow, reuse_if_exists=False)
    return replace(base_config, pod_spec=pod_spec, workflow=workflow)


@pytest.fixture
def pod_tracker(orchestrator: OrchestratorService):
    created: list[str] = []
    terminated: set[str] = set()

    def _track(pod_id: str) -> None:
        if pod_id and pod_id not in created:
            created.append(pod_id)

    yield _track

    errors: list[str] = []
    for pod_id in reversed(created):
        if pod_id in terminated:
            continue
        try:
            orchestrator.terminate(pod_id)
            terminated.add(pod_id)
        except Exception as exc:
            errors.append(f"{pod_id}: {exc}")

    if errors:
        raise AssertionError(
            "Failed to strictly terminate all pods used by test:\n" + "\n".join(errors)
        )