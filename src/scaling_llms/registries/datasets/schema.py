# scaling_llms/registry/datasets/schema.py
from __future__ import annotations

from scaling_llms.registries.core.schema import (
    validate_table_name,
    ColumnSpec, 
    IndexSpec, 
    TableSpec
)


DATASET_IDENTITY_COLS = (
    "dataset_name",
    "dataset_config",
    "train_split",
    "eval_split",
    "tokenizer_name",
    "text_field",
)


DEFAULT_DATASETS_TABLE_NAME = "datasets"

DATASETS_COLUMNS = [
    # Identity fields (Used for dataset lookup and must be immutable after registration)
    ColumnSpec("dataset_name", "TEXT", nullable=False),
    ColumnSpec("dataset_config", "TEXT", nullable=True, default_sql="NULL"),
    ColumnSpec("train_split", "TEXT", nullable=True, default_sql="NULL"),
    ColumnSpec("eval_split", "TEXT", nullable=True, default_sql="NULL"),
    ColumnSpec("tokenizer_name", "TEXT", nullable=True, default_sql="NULL"),
    ColumnSpec("text_field", "TEXT", nullable=True, default_sql="NULL"),

    # Artifact fields (Created at dataset registration time, not nullable)
    ColumnSpec("artifacts_path", "TEXT", nullable=False),
    ColumnSpec("created_at", "TEXT", nullable=False),

    # Metadata fields (Optional)
    ColumnSpec("vocab_size", "INTEGER", nullable=True, default_sql="NULL"),
    ColumnSpec("total_tokens", "INTEGER", nullable=True, default_sql="NULL"),
    ColumnSpec("total_train_tokens", "INTEGER", nullable=True, default_sql="NULL"),
    ColumnSpec("total_eval_tokens", "INTEGER", nullable=True, default_sql="NULL"),
]

DATASETS_PRIMARY_KEY_SQL = None

DATASETS_INDEX_DEFS = [
    dict(name="idx_datasets_name_created", columns=("dataset_name", "created_at"), unique=False),
    dict(name="idx_datasets_artifacts_path", columns=("artifacts_path",), unique=True),
    dict(name="idx_datasets_identity", columns=DATASET_IDENTITY_COLS, unique=True),
]


def _index_name(base_name: str, table_name: str) -> str:
    if table_name == DEFAULT_DATASETS_TABLE_NAME:
        return base_name
    return f"{base_name}_{table_name}"


def make_datasets_table_spec(table_name: str = DEFAULT_DATASETS_TABLE_NAME) -> TableSpec:
    table_name = validate_table_name(table_name)
    return TableSpec(
        name=table_name,
        columns=DATASETS_COLUMNS,
        primary_key_sql=DATASETS_PRIMARY_KEY_SQL,
        indexes=[
            IndexSpec(
                name=_index_name(idx["name"], table_name),
                columns=idx["columns"],
                unique=idx["unique"],
            )
            for idx in DATASETS_INDEX_DEFS
        ],
    )