from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from scaling_llms.registries.core.artifacts_sync import make_sync_hooks
from scaling_llms.registries.core.registry import MakeRegistryConfig
from scaling_llms.registries.runs.artifacts import RunArtifacts
from scaling_llms.registries.runs.metadata import RunIdentity, RunMetadata

# NOTE: for type hints only, avoid circular import at runtime
from typing import TYPE_CHECKING, Iterator
if TYPE_CHECKING:
    from scaling_llms.tracking import Run


class RunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunRegistry:
    def __init__(
        self,
        *,
        metadata: RunMetadata,
        artifacts: RunArtifacts,
    ):
        self.metadata = metadata
        self.artifacts = artifacts
        self.artifacts.ensure_root_dir()

    def get_runs_as_df(self, experiment_name: str | None = None) -> pd.DataFrame:
        qry = f"SELECT * FROM {self.metadata.table_name}"
        params = None

        # Optional: if experiment_name is provided, add a WHERE clause
        if experiment_name is not None:
            qry += " WHERE experiment_name = :experiment_name"
            params = {"experiment_name": experiment_name}

        qry += " ORDER BY experiment_name, created_at"

        return self.metadata.read_sql_df(qry, params)

    def run_exists(self, identity: RunIdentity) -> bool:
        return self.metadata.entity_exists(identity)
    
    def set_device_name(self, identity: RunIdentity, device_name: str | None) -> None:
        self.metadata.set_device_name(identity, device_name)

    def set_other_data(
        self, 
        identity: RunIdentity, 
        data: dict[str, str | int | float | bool],
    ) -> None:
        self.metadata.set_other_data(identity, data)
    
    def get_run_metadata(
        self, 
        identity: RunIdentity, 
        raise_if_not_found: bool = True
    ) -> dict[str, Any] | None:
        run_metadata = self.metadata.get_entity_state(identity)
        if run_metadata is None:
            if raise_if_not_found:
                raise FileNotFoundError(f"Run metadata not found for identity: {identity}")
            return None
        return run_metadata

    def get_run_artifacts(
        self, 
        identity: RunIdentity, 
        raise_if_not_found: bool = True
    ) -> RunArtifacts | None:
        # Get Metadata for the run
        run_metadata = self.get_run_metadata(identity, raise_if_not_found=raise_if_not_found)
        if run_metadata is None:
            return None
        
        artifacts_path = run_metadata.get("artifacts_path")
        if artifacts_path is None:
            if raise_if_not_found:
                raise FileNotFoundError(f"Run metadata for identity {identity} does not contain artifacts_path")
            return None
        
        artifacts_dir = self.artifacts.get_dir(artifacts_path, raise_if_not_found=False) # this will also pull/sync artifacts if needed
        if not artifacts_dir.exists():
            if raise_if_not_found:
                raise FileNotFoundError(f"Run artifacts not found at path: {artifacts_path} for identity: {identity}")
            return None

        return artifacts_dir
    
    def get_run( 
        self, 
        identity: RunIdentity, 
        raise_if_not_found: bool = True
    ) -> Run:
        from scaling_llms.tracking import Run

        artifacts_dir = self.get_run_artifacts(identity, raise_if_not_found=raise_if_not_found)
        
        return Run(artifacts_dir)
    
    def get_artifacts_path(
        self,
        identity: RunIdentity,
        raise_if_not_found: bool = True,
    ) -> Path | None:
        # Get Artifacts dir for the run (this will also validate that metadata and artifacts exist and are linked correctly)
        artifacts_dir = self.get_run_artifacts(identity, raise_if_not_found=raise_if_not_found)
        if artifacts_dir is None:
            return None
    
        return artifacts_dir.root # absolute path to the (local) artifacts directory

    def create_run(
        self,
        identity: RunIdentity,
        resume: bool = False,
        overwrite: bool = False,
    ) -> Run:
        from scaling_llms.tracking import Run

        # Validate run existence and handle according to resume/overwrite flags
        exists = self.run_exists(identity)
        if exists and resume:
            # if sync_hooks are configured, this will also attempt to pull/sync artifacts from remote if needed
            return self.get_run(identity)
        if exists and not overwrite:
            raise ValueError(
                f"Run already exists: {identity}. "
                "Set resume=True to reuse it or overwrite=True to replace it."
            )

        # Create run artifacts directory (either new or by wiping existing)
        if exists:
            # Overwriting existing run: wipe artifacts and reuse directory
            artifacts_dir = self.get_run_artifacts(identity)
            artifacts_dir.wipe()
            artifacts_dir.ensure_dirs(exist_ok=True)
        else:
            artifacts_dir = self.artifacts.make_new_dir(identity.experiment_name)

        # Register run metadata
        artifacts_path = self.artifacts.get_relative_path(artifacts_dir.root)
        self.metadata.upsert_run(
            identity,
            artifacts_path=str(artifacts_path.as_posix()),
            status=RunStatus.CREATED.value,
        )

        return Run(artifacts_dir)
    
    def delete_run(
        self,
        identity: RunIdentity,
        confirm: bool = True,
    ) -> None:
        if identity is None:
            raise ValueError("Identity must be provided.")

        if confirm:
            response = input(
                f"Are you sure you want to delete run '{identity.run_name}' "
                f"from experiment '{identity.experiment_name}'? Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        # Get run artifacts path and delete artifacts and metadata
        artifacts_dir = self.get_run_artifacts(identity, raise_if_not_found=True) 
        self.artifacts.delete_dir(artifacts_dir)
        self.metadata.delete_entity(identity)

    def rename_run(
        self,
        identity: RunIdentity,
        new_experiment: str | None = None,
        new_run_name: str | None = None,
        confirm: bool = True,
    ) -> None:
        # Validate inputs
        new_experiment = new_experiment or identity.experiment_name
        new_run_name = new_run_name or identity.run_name
        new_identity = RunIdentity(new_experiment, new_run_name)

        same_experiment = new_experiment == identity.experiment_name
        same_run_name = new_run_name == identity.run_name

        if same_experiment and same_run_name:
            return

        if not self.run_exists(identity):
            raise FileNotFoundError(f"Run not found: {identity}")
        
        if self.run_exists(new_identity):
            raise FileExistsError(f"Target run already exists: {new_identity}")
        
        # Confirm action with the user
        if confirm:
            resp = input(f"Rename run {identity} -> {new_identity}? Type 'y' to confirm: ")
            if resp.strip().lower() not in ("y", "yes"):
                print("Rename cancelled.")
                return

        artifacts_dir = self.get_run_artifacts(identity, raise_if_not_found=True)
        if not same_experiment:
            # Move artifacts to new location
            new_artifacts_dir = self.artifacts.move_dir(
                artifacts_dir=artifacts_dir,
                new_experiment_name=new_experiment,
            )
        else:
            # Keep artifacts in the same place on disk
            # Will simply change the run_name in the DB 
            new_artifacts_dir = artifacts_dir

        # Update metadata with new identity and (possibly) artifacts path
        self.metadata.rename_run(
            identity,
            new_identity=new_identity,
            artifacts_path=str(self.artifacts.get_relative_path(new_artifacts_dir.root).as_posix()),
        )

    @contextmanager
    def managed_run(
        self,
        identity: RunIdentity,
        resume: bool = False,
        overwrite: bool = False,
        fail_on_sync_error: bool = True,
    ) -> Iterator["Run"]:
        from scaling_llms.tracking import Run  # local import, runs at runtime

        def _safe_set(status_value: str) -> None:
            try:
                self.metadata.set_status_value(identity, status_value)
            except Exception:
                pass
        
        # If sync hooks are configured and resume is True, 
        # it'll attempt to pull/sync artifacts from remote if needed
        run = self.create_run(identity, resume=resume, overwrite=overwrite)

        _safe_set(RunStatus.RUNNING.value)
        completed_successfully = False
        try:
            run.start(resume=resume)
            yield run
        except KeyboardInterrupt:
            _safe_set(RunStatus.CANCELLED.value)
            raise
        except Exception:
            _safe_set(RunStatus.FAILED.value)
            raise
        else:
            completed_successfully = True
        finally:
            try:
                run.close()
            except Exception:
                pass

            if completed_successfully:
                try:
                    self.artifacts.push_dir(run.artifacts_dir)
                except Exception:
                    _safe_set(RunStatus.FAILED.value)
                    if fail_on_sync_error:
                        raise
                _safe_set(RunStatus.SUCCEEDED.value)


# -------------------------------
# FACTORY METHOD
# -------------------------------
@dataclass(slots=True)
class MakeRunRegistryConfig(MakeRegistryConfig):
    pass


def make_run_registry(
    config: MakeRunRegistryConfig,
) -> RunRegistry:
    metadata = RunMetadata(
        table_name=config.table_name,
        database_url=config.database_url,
        backend=config.backend,
    )

    sync_hooks = make_sync_hooks(
        local_artifacts_root=config.artifacts_root,
        sync_hooks_type=config.sync_hooks_type,
        sync_hooks_args=config.sync_hooks_args,
    )

    artifacts = RunArtifacts(
        root=config.artifacts_root,
        sync_hooks=sync_hooks,
    )

    return RunRegistry(
        metadata=metadata,
        artifacts=artifacts,
    )