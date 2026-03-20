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

from scaling_llms.storage.base import DefaultRegistryStorage, RegistryStorage
from scaling_llms.registries.runs.registry import RunRegistry
from scaling_llms.registries.runs.artifacts import RunArtifacts
from scaling_llms.registries.runs.metadata import RunMetadata
from scaling_llms.registries.datasets.registry import DataRegistry


# ---------------------------
# GOOGLE DRIVE DEFAULTS (LEGACY MOUNT MODE)
# ----------------------------------------------------------
@dataclass(frozen=True)
class GoogleDriveDefaults:
    """
    Legacy mount-backed layout (local-debug compatibility only).

    This module now treats Google Drive mounts as a compatibility path.
    Active training/registry operations should prefer local disk plus rclone sync.

    Expected structure when mounted:

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

    # Deprecated/legacy knobs kept only for backward-compat constructor calls.
    auto_mount: bool = False
    force_remount: bool = False


def setup_legacy_drive_storage(config: GoogleDriveConfigs = GoogleDriveConfigs()) -> RegistryStorage:
    """
    Resolve storage paths on an already-mounted Google Drive filesystem.

    Legacy-only behavior:
    - Does NOT auto-mount
    - Does NOT create missing mountpoint/drive-root directories
    - Raises RuntimeError when mount path is unavailable

    For cloud/RunPod workflows, use local registries + rclone sync utilities.
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

    if not mountpoint.exists():
        raise RuntimeError(
            "Google Drive mountpoint does not exist. Mount-backed storage is legacy-only. "
            "Use local disk registries with rclone sync for cloud workflows. "
            f"Missing mountpoint: {mountpoint}"
        )

    if not drive_root.exists():
        raise RuntimeError(
            "Google Drive root path does not exist under mountpoint. "
            "Mount-backed storage is legacy/local-debug only. "
            f"Missing path: {drive_root}"
        )

    project_subdir = config.project_subdir or config.defaults.project_subdir
    project_root = drive_root / project_subdir
    project_root.mkdir(parents=True, exist_ok=True)

    return DefaultRegistryStorage(
        project_root=project_root,
    ).to_registry_storage(create_dirs=True)


def setup_drive_storage(config: GoogleDriveConfigs = GoogleDriveConfigs()) -> RegistryStorage:
    """Compatibility wrapper for legacy mount-backed Google Drive storage."""
    return setup_legacy_drive_storage(config)


# ----------------------------------------------------------
# CONVENIENCE FACTORIES
# ----------------------------------------------------------
def make_gdrive_run_registry(configs: GoogleDriveConfigs | None = None, **overrides):
    # Legacy compatibility wrapper: mounted filesystem mode only.
    config = configs or GoogleDriveConfigs(**overrides)
    registry_storage = setup_legacy_drive_storage(config)
    return RunRegistry(
        metadata=RunMetadata(),
        artifacts=RunArtifacts(registry_storage.runs_artifacts_root),
    )


def make_gdrive_data_registry(configs: GoogleDriveConfigs | None = None, **overrides):
    # Legacy compatibility wrapper: mounted filesystem mode only.
    config = configs or GoogleDriveConfigs(**overrides)
    registry_storage = setup_legacy_drive_storage(config)
    return DataRegistry.from_storage(registry_storage)

