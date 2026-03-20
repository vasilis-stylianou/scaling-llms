from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Callable

from scaling_llms.registries.core.helpers import get_current_git_commit_sha, get_local_iso_timestamp
from scaling_llms.registries.runs.artifacts import RunArtifacts
from scaling_llms.registries.runs.metadata import RunIdentity
from scaling_llms.tracking.run import Run


class RunManager:
    """
    New run lifecycle API used by RunRegistry as a compatibility façade.

    This manager keeps the create/resume/overwrite semantics and run status
    transitions compatible with the existing registry behavior.
    """

    def __init__(
        self,
        *,
        runs_table_name: str,
        artifacts_root: Path,
        fetchone: Callable[[str, dict | None], tuple | None],
        execute: Callable[[str, dict | None], None],
        get_run: Callable[[RunIdentity], Run],
        allocate_run_dir: Callable[[str], Path],
        get_git_commit_sha: Callable[[], str | None] = get_current_git_commit_sha,
        prepare_run_artifacts: Callable[[RunIdentity, Run, bool], None] | None = None,
        sync_run_artifacts: Callable[[RunIdentity, Run], None] | None = None,
        default_sync_on_success: bool = False,
        default_fail_on_sync_error: bool = True,
    ):
        self.runs_table_name = runs_table_name
        self.artifacts_root = artifacts_root
        self._fetchone = fetchone
        self._execute = execute
        self._get_run = get_run
        self._allocate_run_dir = allocate_run_dir
        self._get_git_commit_sha = get_git_commit_sha
        self._prepare_run_artifacts = prepare_run_artifacts
        self._sync_run_artifacts = sync_run_artifacts
        self._default_sync_on_success = default_sync_on_success
        self._default_fail_on_sync_error = default_fail_on_sync_error

    @staticmethod
    def _build_identity_placeholders(identity: RunIdentity) -> str:
        return " AND ".join(f"{col}=:{col}" for col in identity.as_kwargs())

    def run_exists(self, identity: RunIdentity) -> bool:
        row = self._fetchone(
            f"SELECT 1 FROM {self.runs_table_name} WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        return row is not None

    def get_run_dir(self, identity: RunIdentity) -> Path:
        row = self._fetchone(
            f"SELECT artifacts_path FROM {self.runs_table_name} WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        if row is None:
            raise FileNotFoundError(f"Run not found: {identity}")
        return self.artifacts_root / str(row[0])

    def get_run_state_row(self, identity: RunIdentity) -> tuple:
        row = self._fetchone(
            f"SELECT * FROM {self.runs_table_name} WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        if row is None:
            raise FileNotFoundError(f"Run not found: {identity}")
        return row

    def get_git_commit(self, identity: RunIdentity) -> str | None:
        row = self._fetchone(
            f"SELECT git_commit FROM {self.runs_table_name} WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )
        if row is None:
            raise FileNotFoundError(f"Run not found: {identity}")
        commit = row[0]
        if commit is None:
            return None
        commit_str = str(commit).strip()
        return commit_str or None

    def set_status_value(self, identity: RunIdentity, status_value: str) -> None:
        if not self.run_exists(identity):
            raise FileNotFoundError(f"Run not found: {identity}")
        ts = get_local_iso_timestamp()
        self._execute(
            f"UPDATE {self.runs_table_name} SET status=:status, updated_at=:updated_at WHERE {self._build_identity_placeholders(identity)}",
            {"status": status_value, "updated_at": ts, **identity.as_kwargs()},
        )

    def set_device_name(self, identity: RunIdentity, device_name: str | None) -> None:
        if not self.run_exists(identity):
            raise FileNotFoundError(f"Run not found: {identity}")
        self._execute(
            f"UPDATE {self.runs_table_name} SET device_name=:device_name WHERE {self._build_identity_placeholders(identity)}",
            {"device_name": device_name, **identity.as_kwargs()},
        )

    def get_experiment_dir(self, experiment_name: str) -> Path:
        exp_dir = self.artifacts_root / experiment_name
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
        return exp_dir

    def delete_run(self, identity: RunIdentity, *, confirm: bool = True) -> None:
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

        self._execute(
            f"DELETE FROM {self.runs_table_name} WHERE {self._build_identity_placeholders(identity)}",
            identity.as_kwargs(),
        )

    def rename_run(
        self,
        identity: RunIdentity,
        *,
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
            resp = input(f"Rename run {identity} -> {new_identity}? Type 'y' to confirm: ")
            if resp.strip().lower() not in ("y", "yes"):
                print("Rename cancelled.")
                return

        row = self._fetchone(
            f"SELECT artifacts_path, run_absolute_path FROM {self.runs_table_name} WHERE {self._build_identity_placeholders(identity)}",
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
        self._execute(
            f"""
            UPDATE {self.runs_table_name}
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

    def delete_experiment(self, experiment_name: str, *, confirm: bool = True) -> None:
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
                "Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        if exp_dir.exists():
            shutil.rmtree(exp_dir)

        self._execute(
            f"DELETE FROM {self.runs_table_name} WHERE experiment_name=:experiment_name",
            {"experiment_name": experiment_name},
        )

    def create_run(
        self,
        identity: RunIdentity,
        *,
        resume: bool = False,
        overwrite: bool = False,
    ) -> Run:
        exists = self.run_exists(identity)

        if exists and resume:
            return self._get_run(identity)
        if exists and not overwrite:
            raise ValueError(
                f"Run already exists: {identity}. "
                "Set resume=True to reuse it or overwrite=True to replace it."
            )

        if not exists:
            run_dir = self._allocate_run_dir(identity.experiment_name)
            artifacts = RunArtifacts(run_dir)
        else:
            run_dir = self.get_run_dir(identity)
            artifacts = RunArtifacts(run_dir)
            artifacts.wipe()

        artifacts.ensure_dirs(exist_ok=exists)
        run = Run(artifacts)

        created_at = updated_at = get_local_iso_timestamp()
        params = {
            **identity.as_kwargs(),
            "artifacts_path": str(run_dir.relative_to(self.artifacts_root).as_posix()),
            "run_absolute_path": str(run_dir.resolve()),
            "created_at": created_at,
            "updated_at": updated_at,
            "status": "CREATED",
            "git_commit": self._get_git_commit_sha(),
        }
        conflict_cols = ", ".join(identity.as_kwargs())
        cols = ", ".join(params)
        placeholders = ", ".join(f":{k}" for k in params)
        update_set = ", ".join(
            f"{k} = excluded.{k}" for k in params if k not in identity.as_kwargs()
        )
        self._execute(
            f"""
            INSERT INTO {self.runs_table_name} ({cols})
            VALUES ({placeholders})
            ON CONFLICT({conflict_cols})
            DO UPDATE SET {update_set}
            """,
            params,
        )

        return run

    @contextmanager
    def managed_run(
        self,
        identity: RunIdentity,
        *,
        resume: bool = False,
        overwrite: bool = False,
        sync_on_success: bool | None = None,
        fail_on_sync_error: bool | None = None,
    ):
        should_sync_on_success = (
            self._default_sync_on_success if sync_on_success is None else sync_on_success
        )
        should_fail_on_sync_error = (
            self._default_fail_on_sync_error
            if fail_on_sync_error is None
            else fail_on_sync_error
        )

        def _safe_set(status_value: str) -> None:
            try:
                self.set_status_value(identity, status_value)
            except Exception:
                pass

        run = self.create_run(identity=identity, resume=resume, overwrite=overwrite)
        if self._prepare_run_artifacts is not None:
            self._prepare_run_artifacts(identity, run, resume=resume)
        _safe_set("RUNNING")
        completed_successfully = False

        try:
            run.start(resume=resume)
            yield run
        except KeyboardInterrupt:
            _safe_set("CANCELLED")
            raise
        except Exception:
            _safe_set("FAILED")
            raise
        else:
            completed_successfully = True
        finally:
            try:
                run.close()
            except Exception:
                pass

            if completed_successfully:
                if should_sync_on_success and self._sync_run_artifacts is not None:
                    try:
                        self._sync_run_artifacts(identity, run)
                    except Exception:
                        _safe_set("FAILED")
                        if should_fail_on_sync_error:
                            raise
                _safe_set("SUCCEEDED")
