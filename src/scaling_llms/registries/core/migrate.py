import sqlite3
from pathlib import Path

from scaling_llms.registries.core.schema import TableSpec


def _has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def _is_nullish_default(default_sql) -> bool:
    if default_sql is None:
        return True
    s = str(default_sql).strip().upper()
    return s == "" or s == "NULL"


def _ddl_fragment_force_nullable(col) -> str:
    # assumes col.ddl_fragment() includes "NOT NULL" when nullable=False
    frag = col.ddl_fragment()
    frag = frag.replace(" NOT NULL", "").replace(" not null", "")
    return frag


def migrate(db_path: str | Path, table_specs: list[TableSpec]) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as con:
        con.execute("BEGIN;")

        for spec in table_specs:
            con.execute(spec.create_table_sql())

            for col in spec.columns:
                if _has_column(con, spec.name, col.name):
                    continue

                # SQLite cannot ADD COLUMN ... NOT NULL without a non-NULL DEFAULT
                ddl = col.ddl_fragment()
                if (col.nullable is False) and _is_nullish_default(col.default_sql):
                    ddl = _ddl_fragment_force_nullable(col)

                con.execute(f"ALTER TABLE {spec.name} ADD COLUMN {ddl};")

                # Backfill defaults for existing rows
                if not _is_nullish_default(col.default_sql):
                    con.execute(
                        f"UPDATE {spec.name} "
                        f"SET {col.name} = {col.default_sql} "
                        f"WHERE {col.name} IS NULL;"
                    )

            for idx in spec.indexes:
                con.execute(idx.ddl(spec.name))

        con.execute("COMMIT;")