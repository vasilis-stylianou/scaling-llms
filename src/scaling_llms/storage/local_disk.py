from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scaling_llms.registries.datasets.registry import DataRegistry
from scaling_llms.registries.runs.registry import RunRegistry
from scaling_llms.storage.base import RegistryStorage


@dataclass(frozen=True)
class LocalDiskDefaults:
    run_registry_name: str = "run_registry"
    runs_db_name: str = "runs.db"
    runs_artifacts_subdir: str = "artifacts"
    data_registry_name: str = "data_registry"
    datasets_db_name: str = "datasets.db"
    datasets_artifacts_subdir: str = "tokenized_datasets"


DEFAULT_LOCAL_DISK = LocalDiskDefaults()


@dataclass
class LocalDiskConfigs:
    project_root: str | Path
    defaults: LocalDiskDefaults = DEFAULT_LOCAL_DISK
    run_registry_name: str | None = None
    runs_db_name: str | None = None
    runs_artifacts_subdir: str | None = None
    data_registry_name: str | None = None
    datasets_db_name: str | None = None
    datasets_artifacts_subdir: str | None = None


def setup_local_storage(config: LocalDiskConfigs) -> RegistryStorage:
    project_root = Path(config.project_root).expanduser()
    project_root.mkdir(parents=True, exist_ok=True)

    run_registry_name = config.run_registry_name or config.defaults.run_registry_name
    runs_db_name = config.runs_db_name or config.defaults.runs_db_name
    runs_artifacts_subdir = config.runs_artifacts_subdir or config.defaults.runs_artifacts_subdir

    data_registry_name = config.data_registry_name or config.defaults.data_registry_name
    datasets_db_name = config.datasets_db_name or config.defaults.datasets_db_name
    datasets_artifacts_subdir = (
        config.datasets_artifacts_subdir or config.defaults.datasets_artifacts_subdir
    )

    run_registry_root = project_root / run_registry_name
    data_registry_root = project_root / data_registry_name

    run_registry_root.mkdir(parents=True, exist_ok=True)
    data_registry_root.mkdir(parents=True, exist_ok=True)

    return RegistryStorage(
        project_root=project_root,
        run_registry_root=run_registry_root,
        runs_db_path=run_registry_root / runs_db_name,
        runs_artifacts_root=run_registry_root / runs_artifacts_subdir,
        data_registry_root=data_registry_root,
        datasets_db_path=data_registry_root / datasets_db_name,
        datasets_artifacts_root=data_registry_root / datasets_artifacts_subdir,
    )


def make_local_run_registry(configs: LocalDiskConfigs | None = None, **overrides) -> RunRegistry:
    if configs is None and "project_root" not in overrides:
        raise ValueError("project_root must be provided explicitly.")
    config = configs or LocalDiskConfigs(**overrides)
    storage = setup_local_storage(config)
    return RunRegistry.from_storage(storage)


def make_local_data_registry(configs: LocalDiskConfigs | None = None, **overrides) -> DataRegistry:
    if configs is None and "project_root" not in overrides:
        raise ValueError("project_root must be provided explicitly.")
    config = configs or LocalDiskConfigs(**overrides)
    storage = setup_local_storage(config)
    return DataRegistry.from_storage(storage)