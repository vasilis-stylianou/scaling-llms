from __future__ import annotations
from dataclasses import dataclass

from scaling_llms.utils.config import BaseJsonConfig

DATASET_IDENTITY_COLS = (
    "dataset_name",
    "dataset_config",
    "train_split",
    "eval_split",
    "tokenizer_name",
    "text_field",
)

@dataclass(frozen=True)
class DatasetIdentity(BaseJsonConfig):
    dataset_name: str
    dataset_config: str | None
    train_split: str
    eval_split: str
    tokenizer_name: str
    text_field: str

    def as_kwargs(self) -> dict[str, object]:
        return {k: getattr(self, k) for k in DATASET_IDENTITY_COLS}

    def slug(self) -> str:
        def norm(x: str) -> str:
            return x.replace("/", "_").replace(":", "_")

        cfg = self.dataset_config if self.dataset_config else "none"

        return (
            f"{norm(self.dataset_name)}"
            f"__cfg={norm(cfg)}"
            f"__train={norm(self.train_split)}"
            f"__eval={norm(self.eval_split)}"
            f"__tok={norm(self.tokenizer_name)}"
            f"__field={norm(self.text_field)}"
        )