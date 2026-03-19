from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from scaling_llms.registries.datasets.identity import DatasetIdentity
from scaling_llms.registries.datasets.registry import DataRegistry
from scaling_llms.registries.runs.identity import RunIdentity
from scaling_llms.registries.runs.registry import RunRegistry
from scaling_llms.storage.base import DefaultRegistryStorage
from scaling_llms.storage.rclone import RCloneDiskConfigs, make_remote_storage, sync_local_to_remote, sync_remote_to_local


EXPERIMENT_NAME = "it_remote_runner_dev"
RUN_START = "run_start"
RUN_TRANSFER = "run_transfer"
REMOTE_SUBDIR = "ml-experiments/scaling-llms-dev"

TEST_DATASET_ID = DatasetIdentity(
    dataset_name="super_glue",
    dataset_config="cb",
    train_split="train[:1%]",
    eval_split="test[:1%]",
    tokenizer_name="gpt2_tiktoken",
    text_field="premise",
)


def _empty_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    return path


def _make_local_storage(root: Path):
    return DefaultRegistryStorage(project_root=root).to_registry_storage(create_dirs=True)


def _sync_db_snapshots_from_remote(
    *,
    remote_storage,
    local_storage,
    rclone_executable: str,
    rclone_extra_args: list[str] | None,
) -> tuple[RunRegistry, DataRegistry]:
    sync_remote_to_local(
        remote_path=str(remote_storage.runs_db_path),
        local_path=local_storage.run_registry_root,
        mode="copy",
        rclone_executable=rclone_executable,
        extra_args=rclone_extra_args,
    )
    sync_remote_to_local(
        remote_path=str(remote_storage.datasets_db_path),
        local_path=local_storage.data_registry_root,
        mode="copy",
        rclone_executable=rclone_executable,
        extra_args=rclone_extra_args,
    )
    return RunRegistry.from_storage(local_storage), DataRegistry.from_storage(local_storage)


def _push_db_snapshots_to_remote(
    *,
    remote_storage,
    local_storage,
    rclone_executable: str,
    rclone_extra_args: list[str] | None,
) -> None:
    sync_local_to_remote(
        local_path=local_storage.runs_db_path,
        remote_path=str(remote_storage.run_registry_root),
        mode="copy",
        rclone_executable=rclone_executable,
        extra_args=rclone_extra_args,
    )
    sync_local_to_remote(
        local_path=local_storage.datasets_db_path,
        remote_path=str(remote_storage.data_registry_root),
        mode="copy",
        rclone_executable=rclone_executable,
        extra_args=rclone_extra_args,
    )


def _run_remote_config(
    *,
    repo_root: Path,
    script_path: Path,
    configs_root: Path,
    yaml_name: str,
    local_project_root: Path,
) -> None:
    materialized_yaml = local_project_root.parent / f"materialized_{yaml_name}"
    _materialize_yaml(
        template_path=configs_root / yaml_name,
        local_project_root=local_project_root,
        target_path=materialized_yaml,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(materialized_yaml),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"run_remote_experiments failed for {yaml_name}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _get_remote_run_snapshot(
    *,
    remote_storage,
    tmp_root: Path,
    rclone_executable: str,
    rclone_extra_args: list[str] | None,
) -> RunRegistry:
    remote_snapshot_storage = _make_local_storage(tmp_root / "remote_snapshot_runs")
    sync_remote_to_local(
        remote_path=str(remote_storage.run_registry_root),
        local_path=remote_snapshot_storage.run_registry_root,
        mode="copy",
        rclone_executable=rclone_executable,
        extra_args=rclone_extra_args,
    )
    return RunRegistry.from_storage(remote_snapshot_storage)


def _delete_remote_run_state(
    *,
    remote_storage,
    tmp_root: Path,
    identity: RunIdentity,
    rclone_executable: str,
    rclone_extra_args: list[str] | None,
) -> None:
    local_admin_storage = _make_local_storage(tmp_root / f"admin_delete_{identity.run_name}")
    run_registry, _ = _sync_db_snapshots_from_remote(
        remote_storage=remote_storage,
        local_storage=local_admin_storage,
        rclone_executable=rclone_executable,
        rclone_extra_args=rclone_extra_args,
    )

    row = run_registry.fetchone(
        "SELECT artifacts_path FROM runs "
        "WHERE experiment_name=:experiment_name AND run_name=:run_name",
        {
            "experiment_name": identity.experiment_name,
            "run_name": identity.run_name,
        },
    )

    run_registry.execute(
        "DELETE FROM runs WHERE experiment_name=:experiment_name AND run_name=:run_name",
        {
            "experiment_name": identity.experiment_name,
            "run_name": identity.run_name,
        },
    )

    _push_db_snapshots_to_remote(
        remote_storage=remote_storage,
        local_storage=local_admin_storage,
        rclone_executable=rclone_executable,
        rclone_extra_args=rclone_extra_args,
    )

    if row is not None and row[0]:
        empty_dir = _empty_dir(tmp_root / f"empty_cleanup_{identity.run_name}")
        sync_local_to_remote(
            local_path=empty_dir,
            remote_path=str(Path(remote_storage.runs_artifacts_root) / row[0]),
            mode="sync",
            rclone_executable=rclone_executable,
            extra_args=rclone_extra_args,
        )


def _ensure_start_run_exists(
    *,
    remote_storage,
    repo_root: Path,
    script_path: Path,
    configs_root: Path,
    local_project_root: Path,
    tmp_root: Path,
    rclone_executable: str,
    rclone_extra_args: list[str] | None,
) -> None:
    remote_runs = _get_remote_run_snapshot(
        remote_storage=remote_storage,
        tmp_root=tmp_root,
        rclone_executable=rclone_executable,
        rclone_extra_args=rclone_extra_args,
    )
    try:
        remote_run_start = remote_runs.get_run(RunIdentity(EXPERIMENT_NAME, RUN_START))
        has_best = remote_run_start.artifacts.checkpoint_path("best.pt").exists()
    except Exception:
        has_best = False

    if not has_best:
        _delete_remote_run_state(
            remote_storage=remote_storage,
            tmp_root=tmp_root,
            identity=RunIdentity(EXPERIMENT_NAME, RUN_START),
            rclone_executable=rclone_executable,
            rclone_extra_args=rclone_extra_args,
        )
        _run_remote_config(
            repo_root=repo_root,
            script_path=script_path,
            configs_root=configs_root,
            yaml_name="remote_run_start.yaml",
            local_project_root=local_project_root,
        )


def _materialize_yaml(template_path: Path, local_project_root: Path, target_path: Path) -> None:
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["local_project_root"] = str(local_project_root)
    target_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@pytest.mark.integration
def test_run_remote_experiments_start(tmp_path: Path) -> None:
    if os.getenv("RUN_GDRIVE_E2E") != "1":
        pytest.skip("Set RUN_GDRIVE_E2E=1 to enable real DEV GDrive integration test.")
    if shutil.which("rclone") is None:
        pytest.skip("rclone executable is required for this integration test.")

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_remote_experiments.py"
    configs_root = Path(__file__).resolve().parent / "remote_run_configs"

    remote_storage = make_remote_storage(
        RCloneDiskConfigs(
            remote_name="gdrive",
            remote_project_subdir=REMOTE_SUBDIR,
        )
    )

    local_project_root = tmp_path / "local_start"

    _delete_remote_run_state(
        remote_storage=remote_storage,
        tmp_root=tmp_path,
        identity=RunIdentity(EXPERIMENT_NAME, RUN_START),
        rclone_executable="rclone",
        rclone_extra_args=None,
    )
    _run_remote_config(
        repo_root=repo_root,
        script_path=script_path,
        configs_root=configs_root,
        yaml_name="remote_run_start.yaml",
        local_project_root=local_project_root,
    )

    local_storage = _make_local_storage(local_project_root)
    local_runs = RunRegistry.from_storage(local_storage)
    local_run_start = local_runs.get_run(RunIdentity(EXPERIMENT_NAME, RUN_START))

    assert str(local_project_root).startswith(str(tmp_path))
    assert (local_project_root / "run_registry").exists()
    assert (local_project_root / "data_registry").exists()
    assert local_run_start.artifacts.checkpoint_path("best.pt").exists()

    remote_runs = _get_remote_run_snapshot(
        remote_storage=remote_storage,
        tmp_root=tmp_path,
        rclone_executable="rclone",
        rclone_extra_args=None,
    )
    remote_run_start = remote_runs.get_run(RunIdentity(EXPERIMENT_NAME, RUN_START))
    assert remote_run_start.artifacts.checkpoint_path("best.pt").exists()

    remote_snapshot_storage = _make_local_storage(tmp_path / "remote_snapshot_data_start")
    sync_remote_to_local(
        remote_path=str(remote_storage.datasets_db_path),
        local_path=remote_snapshot_storage.data_registry_root,
        mode="copy",
        rclone_executable="rclone",
        extra_args=None,
    )
    remote_snapshot_data = DataRegistry.from_storage(remote_snapshot_storage)
    assert remote_snapshot_data.dataset_exists(TEST_DATASET_ID)


@pytest.mark.integration
def test_run_remote_experiments_resume(tmp_path: Path) -> None:
    if os.getenv("RUN_GDRIVE_E2E") != "1":
        pytest.skip("Set RUN_GDRIVE_E2E=1 to enable real DEV GDrive integration test.")
    if shutil.which("rclone") is None:
        pytest.skip("rclone executable is required for this integration test.")

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_remote_experiments.py"
    configs_root = Path(__file__).resolve().parent / "remote_run_configs"
    local_project_root = tmp_path / "local_resume"

    remote_storage = make_remote_storage(
        RCloneDiskConfigs(
            remote_name="gdrive",
            remote_project_subdir=REMOTE_SUBDIR,
        )
    )

    _delete_remote_run_state(
        remote_storage=remote_storage,
        tmp_root=tmp_path,
        identity=RunIdentity(EXPERIMENT_NAME, RUN_START),
        rclone_executable="rclone",
        rclone_extra_args=None,
    )
    _run_remote_config(
        repo_root=repo_root,
        script_path=script_path,
        configs_root=configs_root,
        yaml_name="remote_run_start.yaml",
        local_project_root=local_project_root,
    )
    _run_remote_config(
        repo_root=repo_root,
        script_path=script_path,
        configs_root=configs_root,
        yaml_name="remote_run_resume.yaml",
        local_project_root=local_project_root,
    )

    local_storage = _make_local_storage(local_project_root)
    local_runs = RunRegistry.from_storage(local_storage)
    local_run_start = local_runs.get_run(RunIdentity(EXPERIMENT_NAME, RUN_START))

    assert local_run_start.artifacts.checkpoint_path("best.pt").exists()
    assert local_run_start.artifacts.checkpoint_path("latest.pt").exists()

    remote_runs = _get_remote_run_snapshot(
        remote_storage=remote_storage,
        tmp_root=tmp_path,
        rclone_executable="rclone",
        rclone_extra_args=None,
    )
    remote_run_start = remote_runs.get_run(RunIdentity(EXPERIMENT_NAME, RUN_START))
    assert remote_run_start.artifacts.checkpoint_path("best.pt").exists()
    assert remote_run_start.artifacts.checkpoint_path("latest.pt").exists()


@pytest.mark.integration
def test_run_remote_experiments_start_from_checkpoint(tmp_path: Path) -> None:
    if os.getenv("RUN_GDRIVE_E2E") != "1":
        pytest.skip("Set RUN_GDRIVE_E2E=1 to enable real DEV GDrive integration test.")
    if shutil.which("rclone") is None:
        pytest.skip("rclone executable is required for this integration test.")

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_remote_experiments.py"
    configs_root = Path(__file__).resolve().parent / "remote_run_configs"
    local_project_root = tmp_path / "local_transfer"

    remote_storage = make_remote_storage(
        RCloneDiskConfigs(
            remote_name="gdrive",
            remote_project_subdir=REMOTE_SUBDIR,
        )
    )

    _ensure_start_run_exists(
        remote_storage=remote_storage,
        repo_root=repo_root,
        script_path=script_path,
        configs_root=configs_root,
        local_project_root=local_project_root,
        tmp_root=tmp_path,
        rclone_executable="rclone",
        rclone_extra_args=None,
    )
    _delete_remote_run_state(
        remote_storage=remote_storage,
        tmp_root=tmp_path,
        identity=RunIdentity(EXPERIMENT_NAME, RUN_TRANSFER),
        rclone_executable="rclone",
        rclone_extra_args=None,
    )
    _run_remote_config(
        repo_root=repo_root,
        script_path=script_path,
        configs_root=configs_root,
        yaml_name="remote_run_start_from_checkpoint.yaml",
        local_project_root=local_project_root,
    )

    local_storage = _make_local_storage(local_project_root)
    local_runs = RunRegistry.from_storage(local_storage)
    local_run_transfer = local_runs.get_run(RunIdentity(EXPERIMENT_NAME, RUN_TRANSFER))

    assert local_run_transfer.artifacts.metadata_path("source_checkpoint.json").exists()

    remote_runs = _get_remote_run_snapshot(
        remote_storage=remote_storage,
        tmp_root=tmp_path,
        rclone_executable="rclone",
        rclone_extra_args=None,
    )
    remote_run_transfer = remote_runs.get_run(RunIdentity(EXPERIMENT_NAME, RUN_TRANSFER))
    remote_run_start = remote_runs.get_run(RunIdentity(EXPERIMENT_NAME, RUN_START))

    assert remote_run_transfer.artifacts.metadata_path("source_checkpoint.json").exists()
    assert remote_run_start.artifacts.checkpoint_path("best.pt").exists()
