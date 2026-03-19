from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from scaling_llms.constants import DATASET_FILES
from scaling_llms.registries.core.db import RegistryDB
from scaling_llms.registries.core.helpers import get_local_iso_timestamp, make_unique_dir
from scaling_llms.registries.datasets.identity import DatasetIdentity
from scaling_llms.registries.datasets.schema import TABLE_SPECS
from scaling_llms.registries.datasets.artifacts import DatasetArtifacts, TokenizedDatasetInfo
from scaling_llms.storage.base import RegistryStorage
from scaling_llms.utils.io import log_as_json


class DataRegistry(RegistryDB):
    """
    Registry for dataset artifacts stored under a datasets/ directory.
    Tracks dataset metadata in an SQLite DB.
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
    def from_storage(cls, storage: RegistryStorage) -> "DataRegistry":
        return cls(
            root=storage.data_registry_root,
            db_path=storage.datasets_db_path,
            artifacts_root=storage.datasets_artifacts_root,
            storage=storage,
        )

    # --- API ---
    def get_datasets_as_df(self) -> pd.DataFrame:
        query = "SELECT * FROM datasets ORDER BY dataset_name, created_at"
        return self.read_sql_df(query)

    def copy_dataset_to_local(
        self,
        dataset_path: str | Path,
        local_path: str | Path,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        """
        Copy train.bin and eval.bin from a dataset directory to a local directory.
        """
        src_dir = Path(dataset_path)
        if not src_dir.is_absolute():
            src_dir = (self.artifacts_root / src_dir).resolve()
        else:
            src_dir = src_dir.resolve()

        src = DatasetArtifacts(src_dir)
        if not src.train_bin.exists():
            raise FileNotFoundError(f"Train bin not found: {src.train_bin}")
        if not src.eval_bin.exists():
            raise FileNotFoundError(f"Eval bin not found: {src.eval_bin}")

        dst_dir = Path(local_path).expanduser().resolve()
        dst_dir.mkdir(parents=True, exist_ok=True)

        train_dst = dst_dir / DATASET_FILES.train_tokens
        eval_dst = dst_dir / DATASET_FILES.eval_tokens

        if train_dst.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {train_dst}")
        if eval_dst.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {eval_dst}")

        if train_dst.exists():
            train_dst.unlink()
        if eval_dst.exists():
            eval_dst.unlink()

        shutil.copy2(src.train_bin, train_dst)
        shutil.copy2(src.eval_bin, eval_dst)

        return train_dst, eval_dst

    def find_dataset_path(
        self,
        identity: DatasetIdentity,
        raise_if_not_found: bool = True,
    ) -> Path | None:
        kwargs = identity.as_kwargs()
        where = " AND ".join(f"{col} IS :{col}" for col in kwargs)
        row = self.fetchone(
            f"SELECT artifacts_path FROM datasets WHERE {where}",
            kwargs,
        )

        if row is None and raise_if_not_found:
            raise FileNotFoundError(f"Dataset not found for identity: {identity}")

        return (self.artifacts_root / Path(row[0])) if row is not None else None

    def dataset_exists(self, identity: DatasetIdentity) -> bool:
        path = self.find_dataset_path(identity, raise_if_not_found=False)
        return (path is not None) and path.exists()
    
    def get_dataset_info(self, identity: DatasetIdentity) -> dict[str, Any] | None:
        path = self.find_dataset_path(identity, raise_if_not_found=False)
        if path is None:
            return None

        artifacts = DatasetArtifacts(path) 
        dataset_info_path = artifacts.dataset_info
        if not dataset_info_path.exists():
            return None
 
        return TokenizedDatasetInfo.from_json(dataset_info_path)

    def register_dataset(
        self,
        src_path_train_bin: str | Path,
        src_path_eval_bin: str | Path,
        identity: DatasetIdentity,
        dataset_info: TokenizedDatasetInfo | None = None,
        **kwargs,
    ) -> Path:
        if self.dataset_exists(identity):
            raise FileExistsError("Dataset with the same metadata already exists.")

        # Validate source paths exist
        src_train = Path(src_path_train_bin).expanduser().resolve()
        src_eval = Path(src_path_eval_bin).expanduser().resolve()
        if not src_train.exists():
            raise FileNotFoundError(f"Train bin not found: {src_train}")
        if not src_eval.exists():
            raise FileNotFoundError(f"Eval bin not found: {src_eval}")

        # Create new dataset directory
        dataset_dir = self.make_dataset_dir()
        dst = DatasetArtifacts(dataset_dir)
        dst.ensure_dir()

        shutil.copy2(src_train, dst.train_bin)
        shutil.copy2(src_eval, dst.eval_bin)

        if dataset_info is not None:
            log_as_json(dataset_info, dst.dataset_info)

        params = {
            **identity.as_kwargs(),
            **kwargs,
            "artifacts_path": str(dataset_dir.relative_to(self.artifacts_root).as_posix()),
            "dataset_absolute_path": str(dataset_dir.resolve().as_posix()),
            "created_at": get_local_iso_timestamp(),
        }

        cols = ", ".join(params)
        placeholders = ", ".join(f":{col}" for col in params)
        self.execute(f"INSERT INTO datasets ({cols}) VALUES ({placeholders})", params)

        return dataset_dir

    def delete_dataset(
        self,
        dataset_path: str | Path | None = None,
        identity: DatasetIdentity | None = None,
        confirm: bool = True,
    ) -> None:
        if dataset_path is None:
            assert identity is not None, "Must provide either dataset_path or identity"
            dataset_path = self.find_dataset_path(identity, raise_if_not_found=False)

        path = Path(dataset_path)
        rel_path = path
        if path.is_absolute():
            rel_path = path.relative_to(self.artifacts_root)

        if confirm:
            response = input(
                f"Are you sure you want to delete dataset at '{rel_path}'? "
                "Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        abs_path = (self.artifacts_root / rel_path).resolve()
        if abs_path.exists():
            shutil.rmtree(abs_path)

        self.execute(
            "DELETE FROM datasets WHERE artifacts_path=:artifacts_path",
            {"artifacts_path": str(rel_path.as_posix())},
        )

    def make_dataset_dir(self) -> Path:
        return make_unique_dir(
            parent_dir=self.artifacts_root,
        )
