from __future__ import annotations

import pytest

from runpod_orchestrator.lifecycle import (
    resolve_or_create_pod,
    terminate_pod,
    wait_until_pod_visible_by_name,
)
from runpod_orchestrator.models import SetupSpec
from runpod_orchestrator.provision import setup_pod
from runpod_orchestrator.ssh import run_ssh_command


@pytest.mark.integration
@pytest.mark.runpod
def test_run_experiment_group_script_e2e(
    integration_config,
    unique_pod_spec,
    docker_args,
    pod_tracker,
) -> None:
    if not integration_config.up_repo_url:
        pytest.skip("RUNPOD_TEST_REPO_URL is required for run_experiment_group e2e test.")

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

    command = (
        "cd /workspace/repos/scaling-llms && "
        "poetry run python scripts/run_experiment_group.py "
        "--config-module tests.integration.test_experiment_config "
        "--remote-project-root /workspace/remote_registry "
        "--local-project-root /workspace/local_registry "
        "--transfer-mode rclone"
    )

    result = run_ssh_command(
        conn,
        command,
        identity_file=setup_spec.expanded_identity_file,
        check=False,
    )

    assert result.returncode == 0, (
        "run_experiment_group script failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "[group] starting run:" in result.stdout
    assert "[group] finished run:" in result.stdout

    verify = run_ssh_command(
        conn,
        "test -f /workspace/remote_registry/run_registry/runs.db",
        identity_file=setup_spec.expanded_identity_file,
        check=False,
    )
    assert verify.returncode == 0, "Expected remote runs.db to exist after script execution"
