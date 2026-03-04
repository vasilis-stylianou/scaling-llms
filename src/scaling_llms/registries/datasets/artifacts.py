# scaling_llms/registry/datasets/artifacts.py
from __future__ import annotations

from pathlib import Path

from scaling_llms.constants import DATA_FILES
from scaling_llms.utils.loggers import BaseLogger


class DatasetArtifacts:
    """
    Filesystem layout helper for a dataset directory (dataset_*/).
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.logger = BaseLogger(name="DatasetArtifacts")

    @property
    def train_bin(self) -> Path:
        return self.root / DATA_FILES.train_tokens

    @property
    def eval_bin(self) -> Path:
        return self.root / DATA_FILES.eval_tokens

    def ensure_dir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)