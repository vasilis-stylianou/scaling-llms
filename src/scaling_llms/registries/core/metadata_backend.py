from __future__ import annotations

import re
from typing import Any, Protocol

import pandas as pd
import psycopg

from scaling_llms.constants import LOCAL_TIMEZONE


_NAMED_PARAM_RE = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")


def _to_psycopg_named_params(query: str) -> str:
    return _NAMED_PARAM_RE.sub(r"%(\1)s", query)


class MetadataBackend(Protocol):
    def execute(self, qry: str, params: dict[str, Any] | None = None) -> None: ...

    def fetchone(self, qry: str, params: dict[str, Any] | None = None) -> tuple[Any, ...] | None: ...

    def fetchall(self, qry: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]: ...

    def read_sql_df(self, qry: str, params: dict[str, Any] | None = None) -> pd.DataFrame: ...


class PostgresBackend:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, autocommit=False)

    def execute(self, qry: str, params: dict[str, Any] | None = None) -> None:
        with self._connect() as con:
            with con.cursor() as cur:
                cur.execute(_to_psycopg_named_params(qry), params or {})
            con.commit()

    def fetchone(self, qry: str, params: dict[str, Any] | None = None) -> tuple[Any, ...] | None:
        with self._connect() as con:
            with con.cursor() as cur:
                cur.execute(_to_psycopg_named_params(qry), params or {})
                return cur.fetchone()

    def fetchall(self, qry: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        with self._connect() as con:
            with con.cursor() as cur:
                cur.execute(_to_psycopg_named_params(qry), params or {})
                return cur.fetchall()

    def read_sql_df(self, qry: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self._connect() as con:
            with con.cursor() as cur:
                cur.execute(_to_psycopg_named_params(qry), params or {})
                rows = cur.fetchall()
                columns = [desc.name for desc in cur.description] if cur.description else []

        df = pd.DataFrame.from_records(rows, columns=columns)

        if not df.empty:
            for col in ["created_at", "updated_at"]:
                if col in df.columns:
                    dt_col = pd.to_datetime(df[col], errors="coerce", utc=True)
                    df[col] = dt_col.dt.tz_convert(LOCAL_TIMEZONE)

        return df