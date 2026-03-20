from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from scaling_llms.constants import LOCAL_TIMEZONE
from scaling_llms.registries.core.metadata_backend import MetadataBackend, PostgresBackend
from scaling_llms.registries.core.migrate import migrate
from scaling_llms.registries.core.schema import TableSpec


class EntityIdentity(ABC):
    @abstractmethod
    def as_kwargs(self) -> dict[str, Any]:
        ...


class MetadataDB:
    """
    Generic metadata DB wrapper:
        - migrates to a provided schema (TableSpec)
      - provides execute/fetch/read_df helpers

    No run- or dataset-specific logic belongs here.

    TODO: Methods are allowed to use Identity objects but not path-like objects
    """

    def __init__(
        self,
        *,
        table_spec: TableSpec,
        database_url: str,
        backend: MetadataBackend,
    ):
        self.table_name = table_spec.name
        self.table_columns = tuple(table_spec.columns)
        self.database_url = database_url
        self._backend: MetadataBackend = backend or self._make_default_backend()
        migrate(self.database_url, table_spec)

    @staticmethod
    def _build_identity_placeholders(identity: EntityIdentity) -> str:
        return " AND ".join(f"{col}=:{col}" for col in identity.as_kwargs())

    def _make_default_backend(self) -> MetadataBackend:
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL must be set to use Postgres-backed registries. "
                "Provide database_url explicitly or export DATABASE_URL."
            )
        return PostgresBackend(self.database_url)

    def _get_local_iso_timestamp(self) -> str:
        return datetime.now(ZoneInfo(LOCAL_TIMEZONE)).isoformat()
    
    def _get_current_git_commit_sha(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            commit = result.stdout.strip()
            return commit or None
        except Exception:
            return None

    # --- Low-level DB helpers ---
    def execute(self, qry: str, params: dict[str, Any] | None = None) -> None:
        self._backend.execute(qry, params)

    def fetchone(self, qry: str, params: dict[str, Any] | None = None) -> tuple[Any, ...] | None:
        return self._backend.fetchone(qry, params)

    def fetchall(self, qry: str, params: dict[str, Any] | None = None) -> list[tuple[Any, ...]]:
        return self._backend.fetchall(qry, params)

    def read_sql_df(self, qry: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        return self._backend.read_sql_df(qry, params)

    # --- Entity-level helpers ---
    def get_entity_state(self, identity: EntityIdentity) -> dict[str, Any] | None:
        row = self.fetchone(
            f"SELECT * FROM {self.table_name} WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        if row is None:
            return None
        return {
            col.name: value
            for col, value in zip(self.table_columns, row)
        }

    def entity_exists(self, identity: EntityIdentity) -> bool:
        row = self.get_entity_state(identity)
        return row is not None

    def delete_entity(self, identity: EntityIdentity) -> None:
        self.execute(
            f"DELETE FROM {self.table_name} WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )