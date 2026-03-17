"""RunPod integration tests (real cloud resources).

Required env vars:
- RUNPOD_E2E=1
- RUNPOD_API_KEY=...

Optional env vars:
- RUNPOD_TEST_POD_SPEC=...
- RUNPOD_TEST_BOOTSTRAP_SCRIPT=...
- RUNPOD_TEST_TIMEOUT_S=900
- RUNPOD_TEST_POLL_S=5
- RUNPOD_TEST_MAX_ATTEMPTS=5
- RUNPOD_TEST_RETRY_BASE_S=2.0
- RUNPOD_E2E_RUN_UP=1 (enable optional up_workflow test)
- RUNPOD_TEST_REPO_URL=... (optional, only used when RUNPOD_E2E_RUN_UP=1)

Example:
  RUNPOD_E2E=1 RUNPOD_API_KEY=... poetry run pytest -m "integration and runpod" tests/integration -vv

Warning: these tests create real RunPod resources and may incur cost.
"""

from __future__ import annotations

import time

import pytest

from runpod_orchestrator.lifecycle import (
    get_pod_by_id,
    resolve_or_create_pod,
    stop_pod,
    terminate_pod,
    wait_until_pod_visible_by_name,
)
from runpod_orchestrator.models import RetryPolicy, SetupSpec
from runpod_orchestrator.provision import setup_pod
from runpod_orchestrator.ssh import run_ssh_command
from runpod_orchestrator.workflows import up_workflow


def _is_terminal_pod_state(pod: dict) -> bool:
    """Checks whether a pod is in a terminal lifecycle state."""

    desired = str(pod.get("desiredStatus") or pod.get("desired_status") or "").lower()
    if desired in {"exited", "stopped", "terminated"}:
        return True

    runtime = pod.get("runtime") or {}
    runtime_status = str(runtime.get("status") or pod.get("status") or "").lower()
    return runtime_status in {"exited", "stopped", "terminated"}


@pytest.mark.integration
@pytest.mark.runpod
def test_create_reconnect_stop_terminate_e2e(
    integration_config,
    unique_pod_spec,
    docker_args,
    pod_tracker,
    get_pod,
) -> None:
    """Verifies that a pod can be created, reused, stopped, and terminated end to end."""

    first = resolve_or_create_pod(
        unique_pod_spec,
        docker_args,
        reuse_if_exists=False,
        timeout_s=integration_config.timeout_s,
        poll_s=integration_config.poll_s,
        retry_policy=integration_config.retry_policy,
    )
    pod_tracker(first.pod_id)

    assert first.public_ip
    assert first.ssh_port

    visible = wait_until_pod_visible_by_name(
        unique_pod_spec.name,
        timeout_s=min(120, integration_config.timeout_s),
        poll_s=max(1, integration_config.poll_s),
        retry_policy=integration_config.retry_policy,
    )
    assert visible is not None

    second = resolve_or_create_pod(
        unique_pod_spec,
        docker_args,
        reuse_if_exists=True,
        timeout_s=integration_config.timeout_s,
        poll_s=integration_config.poll_s,
        retry_policy=integration_config.retry_policy,
    )
    pod_tracker(second.pod_id)
    assert second.pod_id == first.pod_id

    stop_pod(first.pod_id, retry_policy=integration_config.retry_policy)

    post_stop = get_pod(first.pod_id)
    assert (post_stop.get("id") or post_stop.get("_id")) == first.pod_id

    terminate_pod(first.pod_id, retry_policy=integration_config.retry_policy)


@pytest.mark.integration
@pytest.mark.runpod
def test_create_and_terminate_e2e(
    integration_config,
    unique_pod_spec,
    docker_args,
    pod_tracker,
) -> None:
    """Verifies that a pod can be created, become reachable, and terminate successfully."""

    conn = resolve_or_create_pod(
        unique_pod_spec,
        docker_args,
        reuse_if_exists=False,
        timeout_s=integration_config.timeout_s,
        poll_s=integration_config.poll_s,
        retry_policy=integration_config.retry_policy,
    )
    pod_tracker(conn.pod_id)

    assert conn.pod_id
    assert conn.public_ip
    assert conn.ssh_port

    terminate_pod(conn.pod_id, retry_policy=integration_config.retry_policy)


@pytest.mark.integration
@pytest.mark.runpod
def test_up_workflow_e2e_optional(
    integration_config,
    unique_pod_spec,
    pod_tracker,
) -> None:
    """Verifies that the full up workflow provisions a pod and leaves it ready for use."""

    if not integration_config.run_up_workflow:
        pytest.skip("Optional up_workflow integration test disabled. Set RUNPOD_E2E_RUN_UP=1.")
    if not integration_config.up_repo_url:
        pytest.skip("RUNPOD_TEST_REPO_URL is required when RUNPOD_E2E_RUN_UP=1.")

    setup_spec = SetupSpec(
        identity_file=unique_pod_spec.expanded_identity_file,
        repo_url=integration_config.up_repo_url,
        repo_dir="/workspace/repos/scaling-llms",
        rclone_config_local=None,
        create_jupyter_kernel=False,
        poetry_install_args=[],
    )

    try:
        conn = up_workflow(
            pod_spec=unique_pod_spec,
            bootstrap_script=integration_config.bootstrap_script_path,
            setup_spec=setup_spec,
            reuse_if_exists=False,
            timeout_s=integration_config.timeout_s,
            poll_s=integration_config.poll_s,
            retry_policy=integration_config.retry_policy,
        )
    except Exception:
        maybe_pod = wait_until_pod_visible_by_name(
            unique_pod_spec.name,
            timeout_s=min(120, integration_config.timeout_s),
            poll_s=max(1, integration_config.poll_s),
            retry_policy=integration_config.retry_policy,
        )
        if maybe_pod is not None:
            maybe_pod_id = maybe_pod.get("id") or maybe_pod.get("_id")
            if maybe_pod_id:
                terminate_pod(str(maybe_pod_id), retry_policy=integration_config.retry_policy)
        raise

    pod_tracker(conn.pod_id)
    assert conn.pod_id
    assert conn.public_ip
    assert conn.ssh_port


@pytest.mark.integration
@pytest.mark.runpod
def test_setup_pod_e2e(
    integration_config,
    unique_pod_spec,
    docker_args,
    pod_tracker,
) -> None:
    """Verifies that pod setup prepares the remote workspace and runtime dependencies."""

    if not integration_config.up_repo_url:
        pytest.skip("RUNPOD_TEST_REPO_URL is required for test_setup_pod_e2e.")

    try:
        conn = resolve_or_create_pod(
            unique_pod_spec,
            docker_args,
            reuse_if_exists=False,
            timeout_s=integration_config.timeout_s,
            poll_s=integration_config.poll_s,
            retry_policy=integration_config.retry_policy,
        )
    except Exception:
        maybe_pod = wait_until_pod_visible_by_name(
            unique_pod_spec.name,
            timeout_s=min(120, integration_config.timeout_s),
            poll_s=max(1, integration_config.poll_s),
            retry_policy=integration_config.retry_policy,
        )
        if maybe_pod is not None:
            maybe_pod_id = maybe_pod.get("id") or maybe_pod.get("_id")
            if maybe_pod_id:
                terminate_pod(str(maybe_pod_id), retry_policy=integration_config.retry_policy)
        raise

    pod_tracker(conn.pod_id)

    setup_spec = SetupSpec(
        identity_file=unique_pod_spec.expanded_identity_file,
        repo_url=integration_config.up_repo_url,
        repo_dir="/workspace/repos/scaling-llms",
        rclone_config_local=None,
        create_jupyter_kernel=False,
        poetry_install_args=[],
    )

    setup_pod(conn, setup_spec)

    checks = [
        "test -d /workspace/repos/scaling-llms/.git",
        "cd /workspace/repos/scaling-llms && poetry --version",
        "cd /workspace/repos/scaling-llms && poetry run python -c \"print('ok')\"",
        "test -d /workspace/runs",
    ]

    for command in checks:
        result = run_ssh_command(
            conn,
            command,
            identity_file=setup_spec.expanded_identity_file,
            check=False,
        )
        assert result.returncode == 0, (
            f"Remote validation failed for command: {command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


@pytest.mark.integration
@pytest.mark.runpod
def test_terminate_eventual_consistency_e2e(
    integration_config,
    unique_pod_spec,
    docker_args,
    pod_tracker,
) -> None:
    """Verifies that termination eventually makes the pod disappear or become terminal."""

    conn = resolve_or_create_pod(
        unique_pod_spec,
        docker_args,
        reuse_if_exists=False,
        timeout_s=integration_config.timeout_s,
        poll_s=integration_config.poll_s,
        retry_policy=integration_config.retry_policy,
    )
    pod_tracker(conn.pod_id)

    terminate_pod(conn.pod_id, retry_policy=integration_config.retry_policy)

    verify_policy = integration_config.retry_policy
    verify_deadline = time.time() + max(
        60.0,
        float(verify_policy.max_attempts) * max(1.0, verify_policy.retry_base_s) * 4.0,
    )
    pod_inactive_or_missing = False
    while time.time() < verify_deadline:
        try:
            pod = get_pod_by_id(conn.pod_id, retry_policy=RetryPolicy(max_attempts=1, retry_base_s=0))
            if _is_terminal_pod_state(pod):
                pod_inactive_or_missing = True
                break
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "does not exist" in message:
                pod_inactive_or_missing = True
                break
        time.sleep(max(1.0, verify_policy.retry_base_s))

    assert pod_inactive_or_missing, (
        f"Pod {conn.pod_id} should be absent or terminal after terminate"
    )