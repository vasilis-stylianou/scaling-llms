from __future__ import annotations

from scaling_llms.registries.core.schema import (
    validate_table_name,
    ColumnSpec, 
    IndexSpec, 
    TableSpec
)


DEFAULT_RUNS_TABLE_NAME = "runs"

RUN_IDENTITY_COLS = ("experiment_name", "run_name")

RUNS_COLUMNS = [
    # Identity fields
    ColumnSpec("experiment_name", "TEXT", nullable=False),
    ColumnSpec("run_name", "TEXT", nullable=False),

    # Artifact fields (Created at run creation time, not nullable)
    ColumnSpec("artifacts_path", "TEXT", nullable=False),
    ColumnSpec("created_at", "TEXT", nullable=False),

    # Status fields (Updated throughout the run lifecycle)
    ColumnSpec("status", "TEXT", nullable=True, default_sql="NULL"),
    ColumnSpec("updated_at", "TEXT", nullable=True, default_sql="NULL"),
    ColumnSpec("status_msg", "TEXT", nullable=True, default_sql="NULL"),

    # Metadata fields (Optional)
    ColumnSpec("git_commit", "TEXT", nullable=True, default_sql="NULL"),
    ColumnSpec("device_name", "TEXT", nullable=True, default_sql="NULL"),
    ColumnSpec("other_data", "TEXT", nullable=True, default_sql="NULL"),
]

RUNS_PRIMARY_KEY_SQL = "PRIMARY KEY (experiment_name, run_name)"

RUNS_INDEX_DEFS = [
    dict(name="idx_runs_experiment_created", columns=("experiment_name", "created_at"), unique=False),
    dict(name="idx_runs_artifacts_path", columns=("artifacts_path",), unique=True),
]


def _index_name(base_name: str, table_name: str) -> str:
    if table_name == DEFAULT_RUNS_TABLE_NAME:
        return base_name
    return f"{base_name}_{table_name}"


def make_runs_table_spec(table_name: str = DEFAULT_RUNS_TABLE_NAME) -> TableSpec:
    table_name = validate_table_name(table_name)
    return TableSpec(
        name=table_name,
        columns=RUNS_COLUMNS,
        primary_key_sql=RUNS_PRIMARY_KEY_SQL,
        indexes=[
            IndexSpec(
                name=_index_name(idx["name"], table_name),
                columns=idx["columns"],
                unique=idx["unique"],
            )
            for idx in RUNS_INDEX_DEFS
        ],
    )


RUNS_TABLE = make_runs_table_spec(DEFAULT_RUNS_TABLE_NAME)