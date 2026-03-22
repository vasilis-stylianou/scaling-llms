import re
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Iterator

import psycopg

from scaling_llms.registries.core.metadata_backend import PostgresBackend
from scaling_llms.registries.core.schema import TableSpec


def _has_column(cur: psycopg.Cursor, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def _is_nullish_default(default_sql) -> bool:
    if default_sql is None:
        return True
    s = str(default_sql).strip().upper()
    return s == "" or s == "NULL"


def _ddl_fragment_force_nullable(col) -> str:
    frag = col.ddl_fragment()
    frag = frag.replace(" NOT NULL", "").replace(" not null", "")
    return frag


def _primary_key_columns(primary_key_sql: str | None) -> tuple[str, ...]:
    if not primary_key_sql:
        return ()
    m = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", primary_key_sql, flags=re.IGNORECASE)
    if not m:
        return ()
    cols = [c.strip() for c in m.group(1).split(",") if c.strip()]
    return tuple(cols)


def _normalize_table_specs(table_spec: TableSpec | Sequence[TableSpec]) -> tuple[TableSpec, ...]:
    if isinstance(table_spec, TableSpec):
        return (table_spec,)
    return tuple(table_spec)


@contextmanager
def _migration_connection(backend: PostgresBackend) -> Iterator[tuple[psycopg.Connection, bool]]:
    """
    Yield (connection, should_commit).

    - If backend has an injected connection, reuse it and do not commit here.
    - Otherwise open a managed connection and commit on success.
    """
    if backend.connection is not None:
        yield backend.connection, False
        return

    if not backend.database_url:
        raise ValueError("DATABASE_URL must be set to run registry schema migration.")

    with psycopg.connect(backend.database_url, autocommit=False) as con:
        yield con, True


def migrate(backend: PostgresBackend, table_spec: TableSpec | Sequence[TableSpec]) -> None:
    table_specs = _normalize_table_specs(table_spec)

    with _migration_connection(backend) as (con, should_commit):
        with con.cursor() as cur:
            for spec in table_specs:
                cur.execute(spec.create_table_sql())

                for col in spec.columns:
                    if _has_column(cur, spec.name, col.name):
                        continue

                    ddl = col.ddl_fragment()
                    if (col.nullable is False) and _is_nullish_default(col.default_sql):
                        ddl = _ddl_fragment_force_nullable(col)

                    cur.execute(
                        f"ALTER TABLE {spec.name} ADD COLUMN IF NOT EXISTS {ddl};"
                    )

                    if not _is_nullish_default(col.default_sql):
                        cur.execute(
                            f"UPDATE {spec.name} "
                            f"SET {col.name} = {col.default_sql} "
                            f"WHERE {col.name} IS NULL;"
                        )

                for idx in spec.indexes:
                    cur.execute(idx.ddl(spec.name))

                pk_cols = _primary_key_columns(spec.primary_key_sql)
                if pk_cols:
                    pk_cols_sql = ", ".join(pk_cols)
                    pk_idx_name = f"idx_{spec.name}_pk"
                    cur.execute(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS {pk_idx_name} "
                        f"ON {spec.name} ({pk_cols_sql});"
                    )

        if should_commit:
            con.commit()