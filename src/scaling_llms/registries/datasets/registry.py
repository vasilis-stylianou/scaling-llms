from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from scaling_llms.registries.core.artifacts_sync import RCloneArtifactsSyncHooks
from scaling_llms.registries.datasets.artifacts import DatasetArtifacts, DatasetArtifactsDir, TokenizedDatasetInfo
from scaling_llms.registries.datasets.metadata import DatasetIdentity, DatasetMetadata


class DatasetRegistry:
    def __init__(
        self,
        metadata: DatasetMetadata,
        artifacts: DatasetArtifacts,
    ):
        self.metadata = metadata
        self.artifacts = artifacts

        # self.metadata.ensure_table_exists()
        self.artifacts.ensure_root_dir()

    def get_datasets_as_df(self) -> pd.DataFrame:
        query = f"SELECT * FROM {self.metadata.table_name} ORDER BY dataset_name, created_at"
        return self.metadata.read_sql_df(query)

    def dataset_exists(self, identity: DatasetIdentity) -> bool:
        return self.metadata.entity_exists(identity)

    def get_dataset_metadata(
        self, 
        identity: DatasetIdentity, 
        raise_if_not_found: bool = True
    ) -> dict[str, Any] | None:
        dataset_metadata = self.metadata.get_entity_state(identity)
        if dataset_metadata is None:
            if raise_if_not_found:
                raise FileNotFoundError(f"Dataset metadata not found for identity: {identity}")
            return None
        return dataset_metadata
    
    def get_dataset_artifacts(
        self, 
        identity: DatasetIdentity, 
        raise_if_not_found: bool = True
    ) -> DatasetArtifactsDir | None:
        # Get Metadata for the dataset
        dataset_metadata = self.get_dataset_metadata(identity, raise_if_not_found=raise_if_not_found)
        if dataset_metadata is None:
            return None
        
        artifacts_path = dataset_metadata.get("artifacts_path")
        if artifacts_path is None:
            if raise_if_not_found:
                raise FileNotFoundError(f"Dataset metadata for identity {identity} does not contain artifacts_path")
            return None
        
        artifacts_dir = self.artifacts.get_dir(artifacts_path) # this will also pull/sync artifacts if needed
        if not artifacts_dir.exists():
            if raise_if_not_found:
                raise FileNotFoundError(f"Dataset artifacts not found at path: {artifacts_path} for identity: {identity}")
            return None

        return artifacts_dir
    
    def get_dataset_info(
        self,
        identity: DatasetIdentity,
        raise_if_not_found: bool = True,
    ) -> TokenizedDatasetInfo | None:

        artifacts_dir = self.get_dataset_artifacts(identity, raise_if_not_found=raise_if_not_found)
        if artifacts_dir is None:
            return None
        
        dataset_info_path = artifacts_dir.dataset_info
        if not dataset_info_path.exists():
            if raise_if_not_found:
                raise FileNotFoundError(f"Dataset info not found at path: {dataset_info_path} for identity: {identity}")
            return None
        
        return TokenizedDatasetInfo.from_json(dataset_info_path)

    def register_dataset(
        self,
        src_path_train_bin: str | Path,
        src_path_eval_bin: str | Path,
        identity: DatasetIdentity,
        dataset_info: TokenizedDatasetInfo | None = None,
        **extra_params,
    ) -> DatasetArtifactsDir:
        # Validate that dataset with the same identity doesn't already exist
        if self.dataset_exists(identity):
            raise ValueError("Dataset with the same metadata already exists.")

        # Write dataset artifacts
        artifacts_dir = self.artifacts.write_dataset(
            src_train=src_path_train_bin,
            src_eval=src_path_eval_bin,
            dataset_info=dataset_info,
        )

        # Write metadata
        artifacts_path = self.artifacts.get_relative_path(artifacts_dir.root)
        self.metadata.write_metadata(
            identity=identity,
            artifacts_path=artifacts_path,
            **extra_params,
        )

        return artifacts_dir

    def delete_dataset(
        self,
        identity: DatasetIdentity,
        confirm: bool = True,
    ) -> None:
        if identity is None:
            raise ValueError("Identity must be provided.")

        if confirm:
            response = input(
                "Are you sure you want to delete this dataset? "
                "Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        # Find dataset artifacts path and delete artifacts and metadata
        artifacts_dir = self.get_dataset_artifacts(
            identity, 
            raise_if_not_found=True # raise if metadata or artifacts not found
        )
        self.artifacts.delete_dir(artifacts_dir) # this will also sync deletion to remote if configured
        self.metadata.delete_entity(identity)


def make_dataset_registry(
    *,
    artifacts_root: str | Path, # TODO 
    database_url: str,
    datasets_table_name: str,
    sync_hooks_type: str | None = None,
    sync_hooks_args: dict | None = None,
) -> DatasetRegistry:
    
    # TODO: when not using sync hooks, we shouldn't be writing to the ground truth DB

    # Setup Metadata
    metadata = DatasetMetadata(
        database_url=database_url, # TODO: will use os.getenv("DATABASE_URL")
        datasets_table_name=datasets_table_name,
    )

    # Setup sync hooks
    if sync_hooks_type is None:
        sync_hooks = None
    elif sync_hooks_type == "rclone":
        if sync_hooks_args is None:
            raise ValueError("sync_hooks_args must be provided if sync_hooks_type is specified")
        sync_hooks = RCloneArtifactsSyncHooks(**sync_hooks_args)
    else:        
        raise ValueError(f"Unsupported sync_hooks_type: {sync_hooks_type}")

    # Setup Artifacts
    artifacts = DatasetArtifacts(
        root=artifacts_root,
        sync_hooks=sync_hooks,
    )
    
    return DatasetRegistry(metadata=metadata, artifacts=artifacts)