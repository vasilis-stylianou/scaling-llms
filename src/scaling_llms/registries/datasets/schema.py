# scaling_llms/registry/datasets/schema.py
from __future__ import annotations

from scaling_llms.registries.core.schema import ColumnSpec, IndexSpec, TableSpec
from scaling_llms.registries.datasets.identity import DATASET_IDENTITY_COLS


DATASETS_TABLE = TableSpec(
    name="datasets",
    columns=[
        # Identity fields (Used for dataset lookup and must be immutable after registration)
        ColumnSpec("dataset_name", "TEXT", nullable=False),
        ColumnSpec("dataset_config", "TEXT", nullable=True, default_sql="NULL"),
        ColumnSpec("train_split", "TEXT", nullable=True, default_sql="NULL"),
        ColumnSpec("eval_split", "TEXT", nullable=True, default_sql="NULL"),
        ColumnSpec("tokenizer_name", "TEXT", nullable=True, default_sql="NULL"),
        ColumnSpec("text_field", "TEXT", nullable=True, default_sql="NULL"),
        
        # Artifact fields (Created at dataset registration time, not nullable)
        ColumnSpec("artifacts_path", "TEXT", nullable=False, ),
        ColumnSpec("dataset_absolute_path", "TEXT", nullable=False),
        ColumnSpec("created_at", "TEXT", nullable=False),

        # Metadata fields (Optional)
        ColumnSpec("vocab_size", "INTEGER", nullable=True, default_sql="NULL"),
        ColumnSpec("total_tokens", "INTEGER", nullable=True, default_sql="NULL"),
        ColumnSpec("total_train_tokens", "INTEGER", nullable=True, default_sql="NULL"),
        ColumnSpec("total_eval_tokens", "INTEGER", nullable=True, default_sql="NULL"),
    ],
    primary_key_sql=None,
    indexes=[
        IndexSpec(
            name="idx_datasets_name_created",
            columns=("dataset_name", "created_at"),
        ),
        IndexSpec(
            name="idx_datasets_artifacts_path",
            columns=("artifacts_path",),
            unique=True,
        ),
        IndexSpec(
            name="idx_datasets_identity",
            columns=DATASET_IDENTITY_COLS,
            unique=True,
        ),
    ],
)

TABLE_SPECS = [DATASETS_TABLE]