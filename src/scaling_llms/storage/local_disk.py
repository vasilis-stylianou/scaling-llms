from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scaling_llms.registries.datasets.registry import DataRegistry
from scaling_llms.registries.runs.artifacts import RunArtifacts
from scaling_llms.registries.runs.metadata import RunMetadata
from scaling_llms.registries.runs.registry import RunRegistry
from scaling_llms.storage.base import DefaultRegistryStorage, RegistryStorage


@dataclass
class LocalDiskConfigs:
    project_root: str | Path


def setup_local_storage(config: LocalDiskConfigs) -> RegistryStorage:
    return DefaultRegistryStorage(
        project_root=config.project_root,
    ).to_registry_storage(create_dirs=True)


def make_local_run_registry(configs: LocalDiskConfigs | None = None, **overrides) -> RunRegistry:
    if configs is None and "project_root" not in overrides:
        raise ValueError("project_root must be provided explicitly.")
    config = configs or LocalDiskConfigs(**overrides)
    storage = setup_local_storage(config)
    return RunRegistry(
        metadata=RunMetadata(),
        artifacts=RunArtifacts(storage.runs_artifacts_root),
    )


def make_local_data_registry(configs: LocalDiskConfigs | None = None, **overrides) -> DataRegistry:
    if configs is None and "project_root" not in overrides:
        raise ValueError("project_root must be provided explicitly.")
    config = configs or LocalDiskConfigs(**overrides)
    storage = setup_local_storage(config)
    return DataRegistry.from_storage(storage)