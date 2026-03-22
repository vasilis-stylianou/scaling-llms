from __future__ import annotations

import os
from pathlib import Path
import shutil
from uuid import uuid4

import psycopg
import pytest

from scaling_llms.registries import (
    DatasetArtifacts,
    DatasetMetadata,
    DatasetRegistry,
    RunArtifacts,
    RunMetadata,
    RunRegistry,
)
from scaling_llms.registries.core.artifacts_sync import ArtifactsSyncHooks
from scaling_llms.registries.core.metadata_backend import PostgresBackend


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def database_url() -> str:
    value = os.getenv("DATABASE_URL")
    if not value:
        pytest.skip("DATABASE_URL is required for postgres-backed tests")
    return value


@pytest.fixture
def pg_conn(database_url: str):
    conn = psycopg.connect(database_url, autocommit=False)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.fixture
def pg_backend(pg_conn):
    return PostgresBackend(connection=pg_conn)


@pytest.fixture
def test_names():
    return {
        "runs_table": _unique_name("runs_test"),
        "datasets_table": _unique_name("datasets_test"),
    }

@pytest.fixture
def dataset_registry(pg_backend: PostgresBackend, test_names: dict[str, str], tmp_path: Path) -> DatasetRegistry:
    metadata = DatasetMetadata(
        backend=pg_backend,
        table_name=test_names["datasets_table"],
    )
    artifacts = DatasetArtifacts(root=tmp_path / "dataset_registry")
    return DatasetRegistry(metadata=metadata, artifacts=artifacts)

@pytest.fixture
def run_registry(pg_backend: PostgresBackend, test_names: dict[str, str], tmp_path: Path) -> RunRegistry:
    metadata = RunMetadata(
        backend=pg_backend,
        table_name=test_names["runs_table"],
    )
    artifacts = RunArtifacts(root=tmp_path / "run_registry")
    return RunRegistry(metadata=metadata, artifacts=artifacts)


@pytest.fixture
def dataset_registry_with_sync_hooks(
    tmp_path: Path,
    pg_backend: PostgresBackend,
    test_names: dict[str, str],
) -> dict[str, object]:
    local_datasets_artifacts_root = tmp_path / "dataset_registry"
    remote_datasets_artifacts_root = tmp_path / "remote_datasets"
    remote_datasets_artifacts_root.mkdir(parents=True, exist_ok=True)

    sync_calls: list[Path] = []
    prepare_calls: list[Path] = []

    class LocalMirrorSyncHooks(ArtifactsSyncHooks):
        def push_local_to_remote(self, relative_artifacts_path: str | Path) -> None:
            rel = Path(relative_artifacts_path)
            sync_calls.append(rel)
            local_path = local_datasets_artifacts_root / rel
            remote_path = remote_datasets_artifacts_root / rel

            if remote_path.exists():
                shutil.rmtree(remote_path)
            remote_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists():
                shutil.copytree(local_path, remote_path)

        def pull_remote_to_local(self, relative_artifacts_path: str | Path) -> None:
            rel = Path(relative_artifacts_path)
            prepare_calls.append(rel)
            local_path = local_datasets_artifacts_root / rel
            remote_path = remote_datasets_artifacts_root / rel

            if not remote_path.exists():
                raise FileNotFoundError(f"Remote dataset artifacts not found: {remote_path}")

            if local_path.exists():
                shutil.rmtree(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(remote_path, local_path)

    metadata = DatasetMetadata(
        backend=pg_backend,
        table_name=test_names["datasets_table"],
    )

    artifacts = DatasetArtifacts(
        root=local_datasets_artifacts_root,
        sync_hooks=LocalMirrorSyncHooks(),
    )

    registry = DatasetRegistry(metadata=metadata, artifacts=artifacts)

    return {
        "registry": registry,
        "remote_root": remote_datasets_artifacts_root,
        "sync_calls": sync_calls,
        "prepare_calls": prepare_calls,
    }