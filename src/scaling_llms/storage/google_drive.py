from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from scaling_llms.constants import (
    PROJECT_NAME,
    DESKTOP_DRIVE_MOUNTPOINT,
    COLAB_DRIVE_MOUNTPOINT,
    DESKTOP_DRIVE_SUBDIR,
    COLAB_DRIVE_SUBDIR,
)

from scaling_llms.storage.base import RegistryStorage
from scaling_llms.registries.runs.registry import RunRegistry
from scaling_llms.registries.datasets.registry import DataRegistry


# ---------------------------
# GOOGLE DRIVE DEFAULTS
# ----------------------------------------------------------
@dataclass(frozen=True)
class GoogleDriveDefaults:
    """
    Example structure on Google Drive:

    {mountpoint}/{drive_subdir}/{project_subdir}/
    ├── data_registry/
    │   ├── datasets.db
    │   └── tokenized_datasets/
    │       ├── <dataset_name_1>/
    │       ├── <dataset_name_2>/
    │       └── ...
    ├── run_registry/
    │   ├── runs.db
    │   └── artifacts/
    │       ├── <experiment_name_1>/
    │       │   ├── <run_1>/
    │       │   └── <run_2>/
    │       ├── <experiment_name_2>/
    │       └── ...
    """

    # ENV-INDEPENDENT defaults (no os.environ access at import time)
    project_subdir: str = PROJECT_NAME
    run_registry_name: str = "run_registry"
    runs_db_name: str = "runs.db"
    runs_artifacts_subdir: str = "artifacts"
    data_registry_name: str = "data_registry"
    datasets_db_name: str = "datasets.db"
    datasets_artifacts_subdir: str = "tokenized_datasets"

    # Possible env-specific mount settings (selection happens in configs)
    desktop_mountpoint: Path = Path(DESKTOP_DRIVE_MOUNTPOINT)
    colab_mountpoint: Path = Path(COLAB_DRIVE_MOUNTPOINT)
    desktop_drive_subdir: str = DESKTOP_DRIVE_SUBDIR
    colab_drive_subdir: str = COLAB_DRIVE_SUBDIR


DEFAULT_GDRIVE = GoogleDriveDefaults()



@dataclass
class GoogleDriveConfigs:
    defaults: GoogleDriveDefaults = DEFAULT_GDRIVE

    # Explicit overrides
    mountpoint: str | Path | None = None
    drive_subdir: str | None = None
    project_subdir: str | None = None
    run_registry_name: str | None = None
    runs_db_name: str | None = None
    runs_artifacts_subdir: str | None = None
    data_registry_name: str | None = None
    datasets_db_name: str | None = None
    datasets_artifacts_subdir: str | None = None

    auto_mount: bool = True
    force_remount: bool = False


def setup_drive_storage(config: GoogleDriveConfigs = GoogleDriveConfigs()) -> RegistryStorage:
    """
    Factory function: resolves paths, mounts drive if needed, creates directories,
    and returns a pure Storage object.
    """
    env = os.environ.get("SCALING_LLMS_ENV", "local")

    # 1. Resolve Mountpoint & Subdir
    if config.mountpoint is None:
        mountpoint_str = config.defaults.colab_mountpoint if env == "colab" else config.defaults.desktop_mountpoint
    else:
        mountpoint_str = config.mountpoint

    drive_subdir = config.drive_subdir or (
        config.defaults.colab_drive_subdir if env == "colab" else config.defaults.desktop_drive_subdir
    )
    
    mountpoint = Path(mountpoint_str)
    drive_root = mountpoint / drive_subdir

    # 2. Perform Side Effects (Mounting & Mkdir)
    if not drive_root.exists():
        if env == "colab" and config.auto_mount:
            try:
                from google.colab import drive  # type: ignore
                drive.mount(str(mountpoint), force_remount=config.force_remount)
            except ImportError as exc:
                raise RuntimeError("Not in Colab or google.colab unavailable.") from exc
                
        drive_root.mkdir(parents=True, exist_ok=True)

    project_subdir = config.project_subdir or config.defaults.project_subdir
    project_root = drive_root / project_subdir
    project_root.mkdir(parents=True, exist_ok=True)

    # 3. Resolve the rest of the names (no side effects needed here)
    run_reg_name = config.run_registry_name or config.defaults.run_registry_name
    run_reg_root = project_root / run_reg_name

    data_reg_name = config.data_registry_name or config.defaults.data_registry_name
    data_reg_root = project_root / data_reg_name

    # 4. Return the clean, finalized Storage object
    return RegistryStorage(
        project_root=project_root,
        
        run_registry_root=run_reg_root,
        runs_artifacts_root=run_reg_root / (config.runs_artifacts_subdir or config.defaults.runs_artifacts_subdir),
        runs_db_path=run_reg_root / (config.runs_db_name or config.defaults.runs_db_name),
        
        data_registry_root=data_reg_root,
        datasets_db_path=data_reg_root / (config.datasets_db_name or config.defaults.datasets_db_name),
        datasets_artifacts_root=data_reg_root / (config.datasets_artifacts_subdir or config.defaults.datasets_artifacts_subdir),
    )


# ----------------------------------------------------------
# CONVENIENCE FACTORIES
# ----------------------------------------------------------
def make_gdrive_run_registry(configs: GoogleDriveConfigs | None = None, **overrides):
    # 1. Create the raw configuration
    config = configs or GoogleDriveConfigs(**overrides)
    
    # 2. Execute side-effects and resolve paths safely
    registry_storage = setup_drive_storage(config)
    
    # 3. Pass the resolved, frozen storage to your registry
    return RunRegistry.from_storage(registry_storage)


def make_gdrive_data_registry(configs: GoogleDriveConfigs | None = None, **overrides):
    # 1. Create the raw configuration
    config = configs or GoogleDriveConfigs(**overrides)
    
    # 2. Execute side-effects and resolve paths safely
    registry_storage = setup_drive_storage(config)
    
    # 3. Pass the resolved, frozen storage to your registry
    return DataRegistry.from_storage(registry_storage)

