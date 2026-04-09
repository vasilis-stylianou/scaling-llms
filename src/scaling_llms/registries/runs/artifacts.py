from __future__ import annotations

import shutil
from pathlib import Path

from scaling_llms.registries.core.artifacts import Artifacts, ArtifactsDir
from scaling_llms.registries.core.artifacts_sync import ArtifactsSyncHooks
from scaling_llms.utils.loggers import BaseLogger


class RunArtifactsDir(ArtifactsDir):
    # ---- hardcoded subdirs -----
    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def tb(self) -> Path:
        return self.root / "tb"

    # ---- path operations ----
    def metadata_path(
        self,
        filename: str,
        subdir_name: str | None = None,
        ensure_dir: bool = False,
    ) -> Path:
        base = self.metadata
        if subdir_name is not None:
            base = base / subdir_name
            if ensure_dir:
                base.mkdir(parents=True, exist_ok=True)

        return base / filename

    def checkpoint_path(self, filename: str) -> Path:
        return self.checkpoints / filename

    def metric_path(self, category: str) -> Path:
        return self.metrics / f"{category}.jsonl"


class RunArtifacts(Artifacts):
    """
    Single source of truth for the run directory layout + path helpers.

    Owns:
      - root path
      - well-known subdirs
      - creation of directory structure
    """

    def __init__(
        self,
        root: str | Path,
        sync_hooks: ArtifactsSyncHooks | None = None,
    ):
        super().__init__(root)
        self.logger = BaseLogger(name="RunArtifacts")
        self.sync_hooks = sync_hooks

    def _ensure_exists(
        self,
        artifacts_dir: RunArtifactsDir,
        *,
        raise_if_not_found: bool,
    ) -> bool:
        if artifacts_dir.exists():
            return True
        if raise_if_not_found:
            raise FileNotFoundError(
                f"Run artifacts directory does not exist: {artifacts_dir}"
            )
        return False

    def make_new_dir(self, experiment_name: str) -> RunArtifactsDir:
        # Ensure experiment directory exists
        experiment_path = self.root / experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)

        # Allocate unique run directory under experiment directory
        run_path = self.make_unique_dir(parent_dir=experiment_path, create_dir=True)

        # Return RunArtifactsDir for the new run directory
        artifacts_dir = RunArtifactsDir(run_path)
        artifacts_dir.ensure_dirs(exist_ok=False)

        return artifacts_dir

    def push_dir(self, artifacts_dir: RunArtifactsDir) -> None:
        if self.sync_hooks is not None:
            relative_artifacts_path = self.get_relative_path(artifacts_dir.root)
            self.sync_hooks.push_local_to_remote(relative_artifacts_path)

    def pull_dir(self, artifacts_dir: RunArtifactsDir) -> None:
        if self.sync_hooks is not None:
            relative_artifacts_path = self.get_relative_path(artifacts_dir.root)
            self.sync_hooks.pull_remote_to_local(relative_artifacts_path)

    def get_dir(
        self, 
        relative_path: str | Path, 
        *,
        raise_if_not_found: bool = True,
        pull: bool = True,
    ) -> RunArtifactsDir | None:
        artifacts_dir = RunArtifactsDir(self.get_absolute_path(relative_path))
        if pull:
            self.pull_dir(artifacts_dir)

        if not self._ensure_exists(artifacts_dir, raise_if_not_found=raise_if_not_found):
            return None

        return artifacts_dir

    def delete_dir(
        self, 
        artifacts_dir: RunArtifactsDir, 
        *,
        raise_if_not_found: bool = True,
    ) -> None:
        if not self._ensure_exists(artifacts_dir, raise_if_not_found=raise_if_not_found):
            return None

        shutil.rmtree(artifacts_dir.root)
        self.push_dir(artifacts_dir)

    def move_dir(
        self,
        *,
        artifacts_dir: RunArtifactsDir,
        new_experiment_name: str,
        raise_if_not_found: bool = True,
    ) -> RunArtifactsDir | None:
        if not self._ensure_exists(artifacts_dir, raise_if_not_found=raise_if_not_found):
            return None

        # Move artifacts to new location under new experiment directory
        new_artifacts_dir = self.make_new_dir(experiment_name=new_experiment_name)
        shutil.move(str(artifacts_dir.root), str(new_artifacts_dir.root))

        # Sync
        self.push_dir(artifacts_dir)  # this will sync deletion of old dir to remote if needed
        self.push_dir(new_artifacts_dir)

        return new_artifacts_dir
