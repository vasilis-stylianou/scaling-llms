from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from scaling_llms.constants import LOCAL_TIMEZONE
from scaling_llms.registries.core.migrate import migrate
from scaling_llms.registries.core.schema import TableSpec


class RegistryDB:
    """
    Generic DB wrapper:
      - holds db_path
      - migrates to a provided schema (TableSpec list)
      - provides execute/fetch/read_df helpers

    No run- or dataset-specific logic belongs here.
    """

    def __init__(self, db_path: str | Path, *, table_specs: list[TableSpec]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        migrate(self.db_path, table_specs)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def execute(self, qry: str, params: dict[str, Any] | None = None) -> None:
        with self._connect() as con:
            con.execute(qry, params or {})
            con.commit()

    def fetchone(self, qry: str, params: dict[str, Any] | None = None) -> tuple[Any, ...] | None:
        with self._connect() as con:
            return con.execute(qry, params or {}).fetchone()

    def fetchall(self, qry: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        with self._connect() as con:
            return con.execute(qry, params or {}).fetchall()

    def read_sql_df(self, qry: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self._connect() as con:
            df = pd.read_sql_query(qry, con, params=params)

        if not df.empty and "created_at" in df.columns:
            created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
            df["created_at"] = created_at.dt.tz_convert(LOCAL_TIMEZONE)

        return df