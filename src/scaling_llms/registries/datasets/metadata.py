from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scaling_llms.registries.core.metadata_backend import MetadataBackend
from scaling_llms.registries.core.metadata import EntityIdentity, MetadataDB
from scaling_llms.registries.datasets.schema import (
    DATASET_IDENTITY_COLS,
    DEFAULT_DATASETS_TABLE_NAME,
    make_datasets_table_spec,
)
from scaling_llms.utils.config import BaseJsonConfig


@dataclass(frozen=True)
class DatasetIdentity(BaseJsonConfig, EntityIdentity):
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


class DatasetMetadata(MetadataDB):
    def __init__(
        self,
        *,
        database_url: str | None = None,
        datasets_table_name: str = DEFAULT_DATASETS_TABLE_NAME,
        backend: MetadataBackend | None = None,
    ):
        super().__init__(
            table_spec=make_datasets_table_spec(datasets_table_name),
            database_url=database_url,
            backend=backend,
        )

    def write_metadata(
        self, 
        identity: DatasetIdentity, 
        artifacts_path: str | Path, 
        **extra_params,
    ) -> None:
        # Collect all parameters for DB insertion
        params = {**identity.as_kwargs(), **(extra_params or {})}
        params["artifacts_path"] = str(artifacts_path)
        params["created_at"] = self._get_local_iso_timestamp()

        # Validate that all keys in params are valid columns in the table
        allowed_fields = {col.name for col in self.table_columns}
        for key in params:
            if key not in allowed_fields:
                raise ValueError(f"Invalid metadata field: {key}. Allowed fields: {sorted(allowed_fields)}")

        # Construct and execute the INSERT query
        columns = ", ".join(params.keys())
        placeholders = ", ".join(f":{k}" for k in params)
        qry = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
        self.execute(qry, params)