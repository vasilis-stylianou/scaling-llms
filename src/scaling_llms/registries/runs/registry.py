from __future__ import annotations

from contextlib import contextmanager
from enum import StrEnum
import shutil
from pathlib import Path
import pandas as pd

from scaling_llms.registries.core.db import RegistryDB
from scaling_llms.registries.core.helpers import (
    get_current_git_commit_sha,
    get_local_iso_timestamp,
    make_unique_dir,
)
from scaling_llms.registries.runs.artifacts import RunArtifacts
from scaling_llms.registries.runs.identity import RunIdentity
from scaling_llms.registries.runs.schema import TABLE_SPECS
from scaling_llms.storage.base import RegistryStorage
from scaling_llms.tracking.run import Run


class RunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunRegistry(RegistryDB):
    """
    Run registry backed by sqlite DB and an artifacts root directory.
    """

    def __init__(
        self,
        root: str | Path,
        db_path: str | Path,
        artifacts_root: str | Path,
        storage: RegistryStorage | None = None,
    ):
        super().__init__(db_path=db_path, table_specs=TABLE_SPECS)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.storage = storage

    # ---- Factory methods ----
    @classmethod
    def from_storage(cls, storage: RegistryStorage) -> "RunRegistry":
        return cls(
            root=storage.run_registry_root,
            db_path=storage.runs_db_path,
            artifacts_root=storage.runs_artifacts_root,
            storage=storage,
        )

    # ---- Context manager for run lifecycle management ----
    @contextmanager
    def managed_run(
        self,
        identity: RunIdentity,
        resume: bool = False,
        overwrite: bool = False,
    ):
        """
        Context manager that guarantees correct run status transitions.

        Lifecycle:
            CREATED  -> RUNNING
            RUNNING  -> SUCCEEDED | FAILED | CANCELLED
        """

        def _safe_set_status(identity: RunIdentity, status: RunStatus) -> None:
            try:
                self.set_status(identity, status)
            except Exception:
                pass

        run = self.create_run(identity=identity, resume=resume, overwrite=overwrite)

        # Force RUNNING on resume as well
        _safe_set_status(identity, RunStatus.RUNNING)

        try:
            run.start(resume=resume)
            yield run

        except KeyboardInterrupt:
            _safe_set_status(identity, RunStatus.CANCELLED)
            raise

        except Exception:
            _safe_set_status(identity, RunStatus.FAILED)
            raise

        else:
            _safe_set_status(identity, RunStatus.SUCCEEDED)

        finally:
            try:
                run.close()
            except Exception:
                # Do not override previous FAILED/CANCELLED
                pass
    
    # ---- API ----
    def get_run(self, identity: RunIdentity) -> Run:
        run_dir = self.get_run_dir(identity)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        artifacts = RunArtifacts(run_dir)
        return Run(artifacts)

    def get_experiment_dir(self, experiment_name: str) -> Path:
        exp_dir = self.artifacts_root / experiment_name
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
        return exp_dir

    def get_runs_as_df(self, experiment_name: str | None = None) -> pd.DataFrame:
        if experiment_name is not None:
            return self.read_sql_df(
                "SELECT * FROM runs WHERE experiment_name = :experiment_name ORDER BY experiment_name, created_at",
                {"experiment_name": experiment_name},
            )
        return self.read_sql_df("SELECT * FROM runs ORDER BY experiment_name, created_at")

    def get_run_dir(self, identity: RunIdentity) -> Path:
        row = self.fetchone(
            f"SELECT artifacts_path FROM runs WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        if row is None:
            raise FileNotFoundError(f"Run not found: {identity}")
        return self.artifacts_root / row[0]

    def get_run_state(self, identity: RunIdentity) -> dict[str, object]:
        row = self.fetchone(
            f"SELECT * FROM runs WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        if row is None:
            raise FileNotFoundError(f"Run not found: {identity}")

        runs_table = next(spec for spec in TABLE_SPECS if spec.name == "runs")
        cols = [col.name for col in runs_table.columns]
        return dict(zip(cols, row))

    def run_exists(self, identity: RunIdentity) -> bool:
        row = self.fetchone(
            f"SELECT 1 FROM runs WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        return row is not None

    def get_git_commit(self, identity: RunIdentity) -> str | None:
        row = self.fetchone(
            f"SELECT git_commit FROM runs WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        if row is None:
            raise FileNotFoundError(f"Run not found: {identity}")
        commit = row[0]
        if commit is None:
            return None
        commit_str = str(commit).strip()
        return commit_str or None

    def set_status(self, identity: RunIdentity, status: RunStatus) -> None:
        if not self.run_exists(identity):
            raise FileNotFoundError(f"Run not found: {identity}")
        ts = get_local_iso_timestamp()
        self.execute(
            f"UPDATE runs SET status=:status, updated_at=:updated_at WHERE {self._build_identity_placeholders(identity)}",
            {"status": status.value, "updated_at": ts, **identity.as_kwargs()},
        )

    def set_device_name(self, identity: RunIdentity, device_name: str | None) -> None:
        if not self.run_exists(identity):
            raise FileNotFoundError(f"Run not found: {identity}")
        self.execute(
            f"UPDATE runs SET device_name=:device_name WHERE {self._build_identity_placeholders(identity)}",
            {"device_name": device_name, **identity.as_kwargs()},
        )

    def create_run(
        self,
        identity: RunIdentity,
        resume: bool = False,
        overwrite: bool = False,
    ) -> Run:
        exists = self.run_exists(identity)

        if exists and resume:
            return self.get_run(identity)
        elif exists and not overwrite:
            raise ValueError(
                f"Run already exists: {identity}. "
                "Set resume=True to reuse it or overwrite=True to replace it."
            )

        # Prepare Run Artifacts
        if not exists:
            run_dir = self._allocate_run_dir(identity.experiment_name)
            artifacts = RunArtifacts(run_dir)
        else:
            run_dir = self.get_run_dir(identity)
            artifacts = RunArtifacts(run_dir)
            artifacts.wipe()

        # Ensure run directory layout
        artifacts.ensure_dirs(exist_ok=exists)

        # Create Run instance
        run = Run(artifacts)

        # Register in DB
        created_at = updated_at = get_local_iso_timestamp()
        params = {
            **identity.as_kwargs(),
            "artifacts_path": str(run_dir.relative_to(self.artifacts_root).as_posix()),
            "run_absolute_path": str(run_dir.resolve()),
            "created_at": created_at,
            "updated_at": updated_at,
            "status": RunStatus.CREATED.value,
            "git_commit": get_current_git_commit_sha(),
        }
        conflict_cols = ", ".join(identity.as_kwargs())
        cols = ", ".join(params)
        placeholders = ", ".join(f":{k}" for k in params)
        update_set = ", ".join(
            f"{k} = excluded.{k}" for k in params if k not in identity.as_kwargs()
        )
        self.execute(
            f"""
            INSERT INTO runs ({cols})
            VALUES ({placeholders})
            ON CONFLICT({conflict_cols})
            DO UPDATE SET {update_set}
            """,
            params,
        )

        return run

    def delete_run(self, identity: RunIdentity, confirm: bool = True) -> None:
        run_dir = self.get_run_dir(identity).resolve()

        if confirm:
            response = input(
                f"Are you sure you want to delete run '{identity.run_name}' "
                f"from experiment '{identity.experiment_name}'? Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        if run_dir.exists():
            shutil.rmtree(run_dir)

        self.execute(
            f"DELETE FROM runs WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )

    def rename_run(
        self,
        identity: RunIdentity,
        new_experiment: str | None = None,
        new_run_name: str | None = None,
        move_artifacts: bool = False,
        confirm: bool = True,
    ) -> None:
        new_experiment = new_experiment or identity.experiment_name
        new_run_name = new_run_name or identity.run_name

        if (new_experiment == identity.experiment_name) and (new_run_name == identity.run_name):
            return

        new_identity = RunIdentity(new_experiment, new_run_name)
        if self.run_exists(new_identity):
            raise FileExistsError(f"Target run already exists: {new_identity}")

        if confirm:
            resp = input(
                f"Rename run {identity} -> {new_identity}? Type 'y' to confirm: "
            )
            if resp.strip().lower() not in ("y", "yes"):
                print("Rename cancelled.")
                return

        row = self.fetchone(
            f"SELECT artifacts_path, run_absolute_path FROM runs WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        if row is None:
            raise FileNotFoundError(f"Run not found: {identity}")

        current_artifacts_path, current_abs_path = row[0], Path(row[1])

        if (new_experiment != identity.experiment_name) and not move_artifacts:
            raise ValueError(
                "Changing the experiment requires move_artifacts=True to relocate artifacts on disk."
            )

        new_artifacts_path = current_artifacts_path
        new_abs_path = current_abs_path

        if move_artifacts:
            src = current_abs_path
            dst_exp_dir = self.artifacts_root / new_experiment
            dst_exp_dir.mkdir(parents=True, exist_ok=True)

            dst = dst_exp_dir / src.name
            if dst.exists():
                raise FileExistsError(f"Destination path already exists: {dst}")

            shutil.move(str(src), str(dst))

            new_abs_path = dst
            new_artifacts_path = str(dst.relative_to(self.artifacts_root).as_posix())

        new_id_params = {f"new_{k}": v for k, v in new_identity.as_kwargs().items()}
        set_id_clause = ", ".join(f"{k}=:new_{k}" for k in new_identity.as_kwargs())
        self.execute(
            f"""
            UPDATE runs
            SET {set_id_clause}, artifacts_path=:artifacts_path, run_absolute_path=:run_absolute_path, updated_at=:updated_at
            WHERE {self._build_identity_placeholders(identity)}
            """,
            {
                **new_id_params,
                "artifacts_path": new_artifacts_path,
                "run_absolute_path": str(new_abs_path),
                "updated_at": get_local_iso_timestamp(),
                **identity.as_kwargs(),
            },
        )

    def delete_experiment(self, experiment_name: str, confirm: bool = True) -> None:
        exp_dir = self.get_experiment_dir(experiment_name).resolve()
        try:
            exp_dir.relative_to(self.artifacts_root.resolve())
        except ValueError:
            raise ValueError(
                f"Experiment directory {exp_dir} must be under artifacts_root {self.artifacts_root}"
            )

        if confirm:
            response = input(
                f"Are you sure you want to delete experiment '{experiment_name}'? "
                f"Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        if exp_dir.exists():
            shutil.rmtree(exp_dir)

        self.execute("DELETE FROM runs WHERE experiment_name=:experiment_name", {"experiment_name": experiment_name})

    # ---- private methods ----
    def _allocate_run_dir(self, experiment_name: str) -> Path:
        exp_dir = self.artifacts_root / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return make_unique_dir(
            parent_dir=exp_dir,
        )
    
    @staticmethod
    def _build_identity_placeholders(identity: RunIdentity) -> str:
        """Build a WHERE clause from identity fields: 'col=:col AND ...'"""
        return " AND ".join(f"{col}=:{col}" for col in identity.as_kwargs())
