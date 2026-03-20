from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


def validate_table_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid SQL table name: {name}")
    return name


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type_sql: str
    nullable: bool = True
    default_sql: str | None = None  # SQL literal: "'CREATED'", "0", "NULL"

    def ddl_fragment(self) -> str:
        parts = [self.name, self.type_sql]
        if self.default_sql is not None:
            parts.append(f"DEFAULT {self.default_sql}")
        if not self.nullable:
            parts.append("NOT NULL")
        return " ".join(parts)


@dataclass(frozen=True)
class IndexSpec:
    name: str
    columns: Sequence[str]
    unique: bool = False

    def ddl(self, table: str) -> str:
        uniq = "UNIQUE " if self.unique else ""
        cols = ", ".join(self.columns)
        return f"CREATE {uniq}INDEX IF NOT EXISTS {self.name} ON {table} ({cols});"


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: Sequence[ColumnSpec]
    primary_key_sql: str | None = None  # e.g. "PRIMARY KEY (a,b)"
    indexes: Sequence[IndexSpec] = ()

    def create_table_sql(self) -> str:
        cols_sql = ",\n    ".join([c.ddl_fragment() for c in self.columns])
        pk_sql = f",\n    {self.primary_key_sql}" if self.primary_key_sql else ""
        return f"""
        CREATE TABLE IF NOT EXISTS {self.name} (
            {cols_sql}{pk_sql}
        );
        """.strip()