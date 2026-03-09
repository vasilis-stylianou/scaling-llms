# scaling_llms/registry/datasets/artifacts.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scaling_llms.constants import DATA_FILES
from scaling_llms.utils.config import BaseJsonConfig
from scaling_llms.utils.loggers import BaseLogger


@dataclass(frozen=True)
class TokenizedDatasetInfo(BaseJsonConfig):
    vocab_size: int
    eos_id: int
    dtype: str
    total_train_tokens: int
    total_eval_tokens: int


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
    
    @property
    def dataset_info(self) -> Path:
        return self.root / DATA_FILES.dataset_info

    def ensure_dir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)