# scaling_llms/registry/datasets/artifacts.py
from __future__ import annotations

import shutil
from pathlib import Path
from dataclasses import dataclass

from scaling_llms.constants import DATASET_FILES
from scaling_llms.registries.core.artifacts import Artifacts, ArtifactsDir
from scaling_llms.registries.core.artifacts_sync import ArtifactsSyncHooks
from scaling_llms.utils.config import BaseJsonConfig
from scaling_llms.utils.loggers import BaseLogger
from scaling_llms.utils.io import log_as_json

@dataclass(frozen=True)
class TokenizedDatasetInfo(BaseJsonConfig):
    vocab_size: int
    eos_id: int
    dtype: str
    total_train_tokens: int
    total_eval_tokens: int | None = None

class DatasetArtifactsDir(ArtifactsDir):

    @property
    def train_bin(self) -> Path:
        return self.root / DATASET_FILES.train_tokens

    @property
    def eval_bin(self) -> Path:
        return self.root / DATASET_FILES.eval_tokens
    
    @property
    def dataset_info(self) -> Path:
        return self.root / DATASET_FILES.dataset_info


class DatasetArtifacts(Artifacts):
    """
    TODO
    """
    def __init__(self, root: str | Path, sync_hooks: ArtifactsSyncHooks | None = None):
        super().__init__(root)
        self.logger = BaseLogger(name="DatasetArtifacts")
        self.sync_hooks = sync_hooks

    def make_new_dir(self) -> DatasetArtifactsDir:
        path = self.make_unique_dir(parent_dir=self.root, create_dir=True)
        return DatasetArtifactsDir(path)
    
    def push_dir(self, artifacts_dir: DatasetArtifactsDir) -> None:
        if self.sync_hooks is not None:
            relative_artifacts_path = self.get_relative_path(artifacts_dir.root)
            self.sync_hooks.push_local_to_remote(relative_artifacts_path)

    def pull_dir(self, artifacts_dir: DatasetArtifactsDir) -> None:
        if self.sync_hooks is not None:
            relative_artifacts_path = self.get_relative_path(artifacts_dir.root)
            self.sync_hooks.pull_remote_to_local(relative_artifacts_path)

    def get_dir(self, relative_path: str | Path, pull: bool = True) -> DatasetArtifactsDir:

        artifacts_dir = DatasetArtifactsDir(self.get_absolute_path(relative_path))
        if pull:
            self.pull_dir(artifacts_dir)

        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Dataset artifacts directory does not exist: {artifacts_dir}")

        return artifacts_dir

    def delete_dir(self, artifacts_dir: DatasetArtifactsDir) -> None:

        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Dataset artifacts directory does not exist: {artifacts_dir}")
        
        shutil.rmtree(artifacts_dir.root)
        self.push_dir(artifacts_dir)

    def write_dataset(
        self,
        src_train: Path,
        src_eval: Path | None = None,
        dataset_info: TokenizedDatasetInfo | None = None,
    ) -> DatasetArtifactsDir:
        # Validate source paths
        src_train = Path(src_train).expanduser().resolve()
        if not src_train.exists():
            raise FileNotFoundError(f"Train bin not found: {src_train}")
        if src_eval is not None:
            src_eval = Path(src_eval).expanduser().resolve()
            if not src_eval.exists():
                raise FileNotFoundError(f"Eval bin not found: {src_eval}")

        # Create new artifacts directory for the dataset
        artifacts_dir = self.make_new_dir()

        # Copy dataset files to local artifacts directory
        shutil.copy2(src_train, artifacts_dir.train_bin)
        if src_eval is not None:
            shutil.copy2(src_eval, artifacts_dir.eval_bin)

        if dataset_info is not None:
            log_as_json(dataset_info, artifacts_dir.dataset_info)

        self.push_dir(artifacts_dir)

        return artifacts_dir