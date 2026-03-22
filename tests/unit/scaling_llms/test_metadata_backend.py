from __future__ import annotations

import pytest

from scaling_llms.registries.core import metadata as metadata_module
from scaling_llms.registries.core.metadata import MetadataDB
from scaling_llms.registries.core.metadata_backend import (
    PostgresBackend,
    _to_psycopg_named_params,
)
from scaling_llms.registries.core.migrate import _primary_key_columns, migrate
from scaling_llms.registries.core.schema import ColumnSpec, TableSpec


def test_to_psycopg_named_params_converts_colon_style_placeholders() -> None:
    query = "SELECT * FROM runs WHERE experiment_name = :experiment_name AND run_name = :run_name"
    assert _to_psycopg_named_params(query) == (
        "SELECT * FROM runs WHERE experiment_name = %(experiment_name)s AND run_name = %(run_name)s"
    )


def test_registry_db_requires_explicit_database_url_and_backend() -> None:
    table_spec = TableSpec(name="dev_runs", columns=[])

    with pytest.raises(TypeError):
        MetadataDB(table_spec=table_spec)


def test_registry_db_uses_explicit_database_url_and_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    table_spec = TableSpec(name="dev_runs", columns=[])

    class FakeBackend:
        def __init__(self, database_url: str | None = None, connection=None):
            captured["backend_url"] = database_url
            captured["backend_connection"] = connection

        def execute(self, qry, params=None):
            captured["execute"] = (qry, params)

        def fetchone(self, qry, params=None):
            return None

        def fetchall(self, qry, params=None):
            return []

        def read_sql_df(self, qry, params=None):
            raise AssertionError("read_sql_df should not be called in this test")

    def fake_migrate(backend, table_spec_arg):
        captured["migrate_backend"] = backend
        captured["migrate_table_spec"] = table_spec_arg

    monkeypatch.setattr(metadata_module, "migrate", fake_migrate)

    backend = FakeBackend(database_url="postgresql://example")
    db = MetadataDB(
        database_url="postgresql://example",
        backend=backend,
        table_spec=table_spec,
    )
    db.execute("UPDATE runs SET status=:status", {"status": "RUNNING"})

    assert captured["backend_url"] == "postgresql://example"
    assert captured["backend_connection"] is None
    assert captured["migrate_backend"] is backend
    assert captured["migrate_table_spec"] == table_spec
    assert captured["execute"] == ("UPDATE runs SET status=:status", {"status": "RUNNING"})


class _FakeCursor:
    def __init__(self):
        self.executed: list[tuple[str, object]] = []
        self._fetchone_value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, sql, params=None):
        normalized_sql = " ".join(str(sql).split())
        self.executed.append((normalized_sql, params))
        if "FROM information_schema.columns" in normalized_sql:
            self._fetchone_value = None

    def fetchone(self):
        return self._fetchone_value


class _FakeConnection:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def test_migrate_uses_backend_managed_connection_and_add_column_if_not_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_con = _FakeConnection()

    class _FakePsycopg:
        @staticmethod
        def connect(_database_url, autocommit=False):
            assert autocommit is False
            return fake_con

    backend = PostgresBackend(database_url="postgresql://example")
    table_specs = [
        TableSpec(
            name="demo",
            columns=[ColumnSpec("new_col", "TEXT", nullable=False, default_sql=None)],
            primary_key_sql=None,
            indexes=(),
        )
    ]

    monkeypatch.setattr("scaling_llms.registries.core.migrate.psycopg", _FakePsycopg)

    migrate(backend, table_specs)

    executed_sql = [sql for sql, _ in fake_con.cursor_obj.executed]
    alter_sql = [sql for sql in executed_sql if "ALTER TABLE demo ADD COLUMN IF NOT EXISTS" in sql]

    assert alter_sql
    assert "NOT NULL" not in alter_sql[0]
    assert fake_con.committed is True


def test_migrate_uses_injected_connection_without_commit() -> None:
    fake_con = _FakeConnection()
    backend = PostgresBackend(connection=fake_con)

    table_specs = [
        TableSpec(
            name="demo",
            columns=[ColumnSpec("new_col", "TEXT", nullable=False, default_sql=None)],
            primary_key_sql=None,
            indexes=(),
        )
    ]

    migrate(backend, table_specs)

    executed_sql = [sql for sql, _ in fake_con.cursor_obj.executed]
    assert any("CREATE TABLE IF NOT EXISTS demo" in sql for sql in executed_sql)
    assert fake_con.committed is False


def test_primary_key_columns_parsing() -> None:
    assert _primary_key_columns("PRIMARY KEY (experiment_name, run_name)") == (
        "experiment_name",
        "run_name",
    )
    assert _primary_key_columns(None) == ()
    assert _primary_key_columns("UNRELATED") == ()