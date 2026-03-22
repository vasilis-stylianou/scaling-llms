from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scaling_llms.registries.core.metadata_backend import MetadataBackend
from scaling_llms.registries.core.metadata import EntityIdentity, MetadataDB
from scaling_llms.registries.runs.schema import (
    DEFAULT_RUNS_TABLE_NAME,
    RUN_IDENTITY_COLS,
    make_runs_table_spec,
)


@dataclass(frozen=True)
class RunIdentity(EntityIdentity):
    experiment_name: str
    run_name: str

    def as_kwargs(self) -> dict[str, str]:
        return {col: getattr(self, col) for col in RUN_IDENTITY_COLS}

    def __str__(self) -> str:
        return f"({self.experiment_name}, {self.run_name})"


class RunMetadata(MetadataDB):
    def __init__(
        self,
        *,
        database_url: str | None = None,
        runs_table_name: str = DEFAULT_RUNS_TABLE_NAME,
        backend: MetadataBackend | None = None,
    ):
        super().__init__(
            table_spec=make_runs_table_spec(runs_table_name),
            database_url=database_url,
            backend=backend,
        )

    def get_git_commit(self, identity: RunIdentity) -> str | None:
        row = self.get_entity_state(identity)
        if row is None:
            raise FileNotFoundError(f"Run not found: {identity}")
        commit = row.get("git_commit")
        if commit is None:
            return None
        commit_str = str(commit).strip()
        return commit_str or None

    def set_status_value(self, identity: RunIdentity, status_value: str) -> None:
        if not self.entity_exists(identity):
            raise FileNotFoundError(f"Run not found: {identity}")
        self.execute(
            f"UPDATE {self.table_name} SET status=:status, updated_at=:updated_at WHERE {self._build_identity_placeholders(identity)}",
            {
                "status": status_value,
                "updated_at": self._get_local_iso_timestamp(),
                **identity.as_kwargs(),
            },
        )

    def set_device_name(self, identity: RunIdentity, device_name: str | None) -> None:
        if not self.entity_exists(identity):
            raise FileNotFoundError(f"Run not found: {identity}")
        self.execute(
            f"UPDATE {self.table_name} SET device_name=:device_name WHERE {self._build_identity_placeholders(identity)}",
            {"device_name": device_name, **identity.as_kwargs()},
        )

    def upsert_run(
        self,
        identity: RunIdentity,
        *,
        artifacts_path: str,
        status: str = "CREATED",
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        timestamp = self._get_local_iso_timestamp()
        params: dict[str, Any] = {
            **identity.as_kwargs(),
            "artifacts_path": artifacts_path,
            "created_at": timestamp,
            "updated_at": timestamp,
            "status": status,
            "git_commit": self._get_current_git_commit_sha(),
        }
        if extra_params:
            params.update(extra_params)

        conflict_cols = ", ".join(identity.as_kwargs())
        columns = ", ".join(params)
        placeholders = ", ".join(f":{k}" for k in params)
        update_set = ", ".join(
            f"{k}=excluded.{k}" for k in params if k not in identity.as_kwargs()
        )

        self.execute(
            f"""
            INSERT INTO {self.table_name} ({columns})
            VALUES ({placeholders})
            ON CONFLICT({conflict_cols})
            DO UPDATE SET {update_set}
            """,
            params,
        )

    def rename_run(
        self,
        *,
        identity: RunIdentity,
        new_identity: RunIdentity,
        artifacts_path: str,
    ) -> None:
        if not self.entity_exists(identity):
            raise FileNotFoundError(f"Run not found: {identity}")

        new_id_params = {f"new_{k}": v for k, v in new_identity.as_kwargs().items()}
        set_id_clause = ", ".join(f"{k}=:new_{k}" for k in new_identity.as_kwargs())
        self.execute(
            f"""
            UPDATE {self.table_name}
            SET {set_id_clause}, artifacts_path=:artifacts_path, updated_at=:updated_at
            WHERE {self._build_identity_placeholders(identity)}
            """,
            {
                **new_id_params,
                "artifacts_path": artifacts_path,
                "updated_at": self._get_local_iso_timestamp(),
                **identity.as_kwargs(),
            },
        )