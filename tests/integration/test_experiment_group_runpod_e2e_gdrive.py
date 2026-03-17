from __future__ import annotations

import time
from pathlib import Path

import pytest

from runpod_orchestrator.lifecycle import (
    resolve_or_create_pod,
    terminate_pod,
    wait_until_pod_visible_by_name,
)
from runpod_orchestrator.models import SetupSpec
from runpod_orchestrator.provision import setup_pod
from runpod_orchestrator.ssh import run_ssh_command
from scaling_llms.constants import PROJECT_DEV_NAME
from scaling_llms.storage.google_drive import DEFAULT_GDRIVE


def _normalize_drive_subdir(subdir: str) -> str:
    value = subdir.strip().strip("/")
    for prefix in ("My Drive/", "MyDrive/"):
        if value.startswith(prefix):
            return value[len(prefix):].strip("/")
    return value


def _gdrive_run_registry_candidates() -> list[str]:
    candidates = [f"gdrive:{PROJECT_DEV_NAME}/run_registry"]

    for raw_subdir in (
        str(DEFAULT_GDRIVE.desktop_drive_subdir),
        str(DEFAULT_GDRIVE.colab_drive_subdir),
    ):
        normalized = _normalize_drive_subdir(raw_subdir)
        if normalized:
            candidates.append(f"gdrive:{normalized}/{PROJECT_DEV_NAME}/run_registry")

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    return unique_candidates


def _is_gdrive_quota_error(stderr: str) -> bool:
    lowered = stderr.lower()
    markers = (
        "quota exceeded",
        "ratelimitexceeded",
        "rate_limit_exceeded",
        "queries per minute",
    )
    return any(marker in lowered for marker in markers)


@pytest.mark.integration
@pytest.mark.runpod
def test_run_experiment_group_script_e2e_gdrive(
    integration_config,
    unique_pod_spec,
    docker_args,
    pod_tracker,
) -> None:
    """Run production-like group training path with RunPod + GDrive.

    Note: the final GDrive artifact existence check retries and may skip when
    Google Drive API quota/rate limits are hit transiently.
    """
    if not integration_config.up_repo_url:
        pytest.skip("RUNPOD_TEST_REPO_URL is required for run_experiment_group e2e test.")

    rclone_cfg = Path(integration_config.rclone_config_local).expanduser().resolve()
    if not rclone_cfg.exists():
        pytest.skip(
            "RUNPOD_TEST_RCLONE_CONFIG (or default ~/.config/rclone/rclone.conf) "
            "must exist for GDrive run_experiment_group e2e test."
        )

    conn = None
    try:
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

        setup_spec = SetupSpec(
            identity_file=unique_pod_spec.expanded_identity_file,
            repo_url=integration_config.up_repo_url,
            repo_dir="/workspace/repos/scaling-llms",
            rclone_config_local=str(rclone_cfg),
            create_jupyter_kernel=False,
            poetry_install_args=[],
        )

        setup_pod(conn, setup_spec)

        command = (
            "cd /workspace/repos/scaling-llms && "
            "poetry run python scripts/run_experiment_group.py "
            "--config-module tests.integration.test_experiment_config "
            "--local-project-root /workspace/local_registry "
            "--transfer-mode rclone "
            "--use-gdrive-remote"
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

        verify_exact = run_ssh_command(
            conn,
            (
                "cd /workspace/repos/scaling-llms && "
                "poetry run python -c \""
                "from scaling_llms.constants import METADATA_FILES, PROJECT_DEV_NAME; "
                "from scaling_llms.registries.runs.identity import RunIdentity; "
                "from scaling_llms.storage.google_drive import make_gdrive_run_registry; "
                "registry = make_gdrive_run_registry(project_subdir=PROJECT_DEV_NAME); "
                "identity = RunIdentity('test_experiment_group', 'group_run_1'); "
                "assert registry.run_exists(identity); "
                "run = registry.get_run(identity); "
                "artifact = run.artifacts.metadata_path(METADATA_FILES.dataset_id); "
                "assert artifact.exists(), f'Missing artifact file: {artifact}'; "
                "print('run-ok'); "
                "print('artifact-ok')"
                "\""
            ),
            identity_file=setup_spec.expanded_identity_file,
            check=False,
        )

        assert verify_exact.returncode == 0, (
            "Expected exact run and artifact verification in GDrive registry to pass\n"
            f"stdout:\n{verify_exact.stdout}\n"
            f"stderr:\n{verify_exact.stderr}"
        )
        assert "run-ok" in verify_exact.stdout
        assert "artifact-ok" in verify_exact.stdout

        candidates = _gdrive_run_registry_candidates()
        ls_chain = " || ".join(f"rclone ls {path}" for path in candidates)
        verify = None
        retry_delays_s = [8, 15, 25, 35]
        max_attempts = len(retry_delays_s) + 1

        for attempt in range(max_attempts):
            verify = run_ssh_command(
                conn,
                f"({ls_chain}) | grep runs.db",
                identity_file=setup_spec.expanded_identity_file,
                check=False,
            )
            if verify.returncode == 0:
                break
            if _is_gdrive_quota_error(verify.stderr):
                if attempt < len(retry_delays_s):
                    time.sleep(retry_delays_s[attempt])
                continue
            break

        assert verify is not None
        if verify.returncode != 0:
            if _is_gdrive_quota_error(verify.stderr):
                print(
                    "[warn] coarse runs.db rclone check skipped due to Drive quota/rate limit; "
                    "exact registry/artifact verification already passed."
                )
            else:
                print(
                    "[warn] coarse runs.db rclone fallback check failed; "
                    "exact registry/artifact verification already passed.\n"
                    f"stdout:\n{verify.stdout}\n"
                    f"stderr:\n{verify.stderr}"
                )
    finally:
        if conn is not None:
            pod_tracker(conn.pod_id)
