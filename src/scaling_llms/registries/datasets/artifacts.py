# scaling_llms/registry/datasets/artifacts.py
from __future__ import annotations

import shutil
from pathlib import Path
from dataclasses import dataclass

from scaling_llms.constants import DATASET_FILES
from scaling_llms.registries.core.artifacts import Artifacts
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
    total_eval_tokens: int

class DatasetArtifactsDir:
    root: Path

    @property
    def train_bin(self) -> Path:
        return self.root / DATASET_FILES.train_tokens

    @property
    def eval_bin(self) -> Path:
        return self.root / DATASET_FILES.eval_tokens
    
    @property
    def dataset_info(self) -> Path:
        return self.root / DATASET_FILES.dataset_info
    
    def exists(self) -> bool:
        return self.train_bin.exists() and self.eval_bin.exists()


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

    def get_dir(self, relative_path: str | Path) -> DatasetArtifactsDir:
        path = self.get_absolute_path(relative_path)

        if self.sync_hooks is not None:
            self.sync_hooks.pull_remote_to_local(relative_path)

        return DatasetArtifactsDir(path)
    
    def write_dataset(
        self,
        src_train: Path,
        src_eval: Path,
        dataset_info: TokenizedDatasetInfo | None = None,
    ) -> Path:
        # Validate source paths
        src_train = Path(src_train).expanduser().resolve()
        src_eval = Path(src_eval).expanduser().resolve()
        if not src_train.exists():
            raise FileNotFoundError(f"Train bin not found: {src_train}")
        if not src_eval.exists():
            raise FileNotFoundError(f"Eval bin not found: {src_eval}")

        # Create new artifacts directory for the dataset
        artifacts_dir = self.make_new_dir()

        # Copy dataset files to local artifacts directory
        shutil.copy2(src_train, artifacts_dir.train_bin)
        shutil.copy2(src_eval, artifacts_dir.eval_bin)

        if dataset_info is not None:
            log_as_json(dataset_info, artifacts_dir.dataset_info)

        if self.sync_hooks is not None:
            relative_artifacts_path = self.get_relative_path(artifacts_dir.root)
            self.sync_hooks.push_local_to_remote(relative_artifacts_path)

        return artifacts_dir.root
    
    def delete_dir(self, artifacts_path: str | Path) -> None:
        # TODO: artifacts_path -> artifacts_dir
        path = self.get_absolute_path(artifacts_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset artifacts directory does not exist: {path}")

        relative_artifacts_path = self.get_relative_path(path)
        shutil.rmtree(path)

        if self.sync_hooks is not None:
            self.sync_hooks.push_local_to_remote(relative_artifacts_path)