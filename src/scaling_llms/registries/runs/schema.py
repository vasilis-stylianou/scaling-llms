from __future__ import annotations

from scaling_llms.registries.core.schema import ColumnSpec, IndexSpec, TableSpec


RUNS_TABLE = TableSpec(
    name="runs",
    columns=[
        # Identity fields
        ColumnSpec("experiment_name", "TEXT", nullable=False),
        ColumnSpec("run_name", "TEXT", nullable=False),

        # Artifact fields (Created at run creation time, not nullable)
        ColumnSpec("artifacts_path", "TEXT", nullable=False),
        ColumnSpec("run_absolute_path", "TEXT", nullable=False),
        ColumnSpec("created_at", "TEXT", nullable=False),

        # Status fields (Updated throughout the run lifecycle)
        ColumnSpec("status", "TEXT", nullable=True, default_sql="NULL"),
        ColumnSpec("updated_at", "TEXT", nullable=True, default_sql="NULL"),
        ColumnSpec("status_msg", "TEXT", nullable=True, default_sql="NULL"),

        # Metadata fields (Optional)
        ColumnSpec("other_data", "TEXT", nullable=True, default_sql="NULL"),

    ],
    primary_key_sql="PRIMARY KEY (experiment_name, run_name)",
    indexes=[
        IndexSpec(
            name="idx_runs_experiment_created",
            columns=("experiment_name", "created_at"),
        ),
        IndexSpec(
            name="idx_runs_artifacts_path",
            columns=("artifacts_path",),
            unique=True,
        ),
    ],
)

TABLE_SPECS = [RUNS_TABLE]