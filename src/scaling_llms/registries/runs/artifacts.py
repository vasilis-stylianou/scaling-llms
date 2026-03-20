from __future__ import annotations

import shutil
from pathlib import Path

from scaling_llms.registries.core.artifacts import Artifacts
from scaling_llms.registries.core.helpers import make_unique_dir
from scaling_llms.registries.core.artifacts_sync import ArtifactsSyncHooks
from scaling_llms.utils.loggers import BaseLogger


class RunArtifactsDir:
    root: Path

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
    
    # ---- dir operations ----
    def exists(self) -> bool:
        # TODO
        return (
            self.metadata.exists() or 
            self.metrics.exists() or 
            self.checkpoints.exists() or 
            self.tb.exists()
        )
    
    def ensure_dirs(self, *, exist_ok: bool = True) -> None:
        """
        If exist_ok=False, raises FileExistsError if any already exist.
        """
        # Create all standard subdirs defined as properties on this class
        for attr_name in dir(type(self)):
            attr = getattr(type(self), attr_name)
            if isinstance(attr, property):
                dir_path = getattr(self, attr_name)
                dir_path.mkdir(parents=True, exist_ok=exist_ok)

    def wipe(self) -> None:
        """Delete all contents of the run directory, but keep the directory itself."""
        # TODO:
        if self.root.exists() and self.root.is_dir():
            for item in self.root.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for subitem in item.rglob("*"):
                        if subitem.is_file():
                            subitem.unlink()
                        elif subitem.is_dir():
                            subitem.rmdir()
                    item.rmdir()
        else:
            self.logger.warning(
                "Run directory %s does not exist or is not a directory; cannot wipe.", 
                self.root
            )


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
    
    def get_dir(self, relative_path: str | Path) -> RunArtifactsDir:
        path = self.get_absolute_path(relative_path)

        if self.sync_hooks is not None:
            self.sync_hooks.pull_remote_to_local(relative_path)

        return RunArtifactsDir(path)
    
    def delete_dir(self, artifacts_path: str | Path) -> None:
        # TODO: artifacts_path -> artifacts_dir
        path = self.get_absolute_path(artifacts_path)
        if not path.exists():
            raise FileNotFoundError(f"Run artifacts directory does not exist: {path}")

        # BUG: Need to make sure that it returns a rel path <exp_name>/<run_dir> 
        # and not an absolute path, otherwise we might end up deleting the wrong directory 
        # on disk if the absolute path is not correct
        relative_artifacts_path = self.get_relative_path(path)
        shutil.rmtree(path)

        if self.sync_hooks is not None:
            self.sync_hooks.push_local_to_remote(relative_artifacts_path)

    def move_dir(self, *, artifacts_dir: RunArtifactsDir, new_experiment_name: str) -> RunArtifactsDir:
        src = artifacts_dir.root.expanduser().resolve()
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {src}")

        new_artifacts_dir = self.make_new_dir(experiment_name=new_experiment_name)
        dst = new_artifacts_dir.root.expanduser().resolve()

        shutil.move(str(src), str(dst))

        return new_artifacts_dir
    

    def push_artifacts_dir(self, artifacts_dir: RunArtifactsDir) -> None:
        # TODO
        if self.sync_hooks is not None:
            relative_artifacts_path = self.get_relative_path(artifacts_dir.root)
            self.sync_hooks.push_local_to_remote(relative_artifacts_path)

    def pull_artifacts_dir(self, artifacts_dir: RunArtifactsDir) -> None:
        # TODO
        if self.sync_hooks is not None:
            relative_artifacts_path = self.get_relative_path(artifacts_dir.root)
            self.sync_hooks.pull_remote_to_local(relative_artifacts_path)



    # def get_experiment_dir(self, experiment_name: str, *, create: bool = False) -> Path:
    #     experiment_dir = self.root / experiment_name
    #     if create:
    #         experiment_dir.mkdir(parents=True, exist_ok=True)
    #     elif not experiment_dir.exists():
    #         raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    #     return experiment_dir
    

    # def allocate_run_dir(self, experiment_name: str) -> Path:
    #     experiment_dir = self.get_experiment_dir(experiment_name, create=True)
    #     return make_unique_dir(parent_dir=experiment_dir)

    # def delete_run_dir(self, run_dir: str | Path) -> None:
    #     path = Path(run_dir).expanduser().resolve()
    #     relative_artifacts_path = self.get_relative_path(path)
    #     if path.exists():
    #         shutil.rmtree(path)
    #     if self.sync_hooks is not None:
    #         self.sync_hooks.push_local_to_remote(relative_artifacts_path)

    

    # def delete_experiment_dir(self, experiment_name: str) -> Path:
    #     experiment_dir = self.get_experiment_dir(experiment_name, create=False)
    #     relative_artifacts_path = self.get_relative_path(experiment_dir)
    #     if experiment_dir.exists():
    #         shutil.rmtree(experiment_dir)
    #     if self.sync_hooks is not None:
    #         self.sync_hooks.push_local_to_remote(relative_artifacts_path)
    #     return experiment_dir

    
    


    # ---- API ----
    

    

    

