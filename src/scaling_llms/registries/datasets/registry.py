from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from scaling_llms.constants import DATA_FILES
from scaling_llms.registries.core.db import RegistryDB
from scaling_llms.registries.core.helpers import get_next_id, get_local_iso_timestamp
from scaling_llms.registries.datasets.schema import TABLE_SPECS
from scaling_llms.registries.datasets.artifacts import DatasetArtifacts
from scaling_llms.storage.base import RegistryStorage


class DataRegistry(RegistryDB):
    """
    Registry for dataset artifacts stored under a datasets/ directory.
    Tracks dataset metadata in an SQLite DB.
    """

    _DATASET_PREFIX = "dataset_"

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

        train_dst = dst_dir / DATA_FILES.train_tokens
        eval_dst = dst_dir / DATA_FILES.eval_tokens

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
        dataset_name: str,
        tokenizer_name: str,
        dataset_config: str | None = None,
        train_split: str | None = None,
        eval_split: str | None = None,
        text_field: str | None = None,
        raise_if_not_found: bool = True,
    ) -> Path | None:
        row = self.fetchone(
            """
            SELECT artifacts_path FROM datasets
            WHERE 
                dataset_name=? 
                AND dataset_config IS ?
                AND train_split IS ? 
                AND eval_split IS ? 
                AND tokenizer_name=?
                AND text_field IS ?
            """,
            (
                dataset_name, 
                dataset_config, 
                train_split, 
                eval_split, 
                tokenizer_name, 
                text_field
            ),
        )

        if (row is None) and raise_if_not_found:
            raise FileNotFoundError(
                "Dataset not found for metadata: "
                f"({dataset_name}, {dataset_config}, {train_split}, {eval_split}, {tokenizer_name}, {text_field})"
            )

        return (self.artifacts_root / Path(row[0])) if row is not None else None

    def dataset_exists(
        self,
        dataset_name: str,
        tokenizer_name: str,
        dataset_config: str | None = None,
        train_split: str | None = None,
        eval_split: str | None = None,
        text_field: str | None = None,
    ) -> bool:
        path = self.find_dataset_path(
            dataset_name=dataset_name,             
            tokenizer_name=tokenizer_name,         
            dataset_config=dataset_config,
            train_split=train_split,
            eval_split=eval_split,
            text_field=text_field,
            raise_if_not_found=False,
        )
        return (path is not None) and path.exists()
    
    def register_dataset(
        self,
        src_path_train_bin: str | Path,
        src_path_eval_bin: str | Path,
        dataset_name: str,
        tokenizer_name: str,
        dataset_config: str | None = None,
        train_split: str | None = None,
        eval_split: str | None = None,
        text_field: str | None = None,
        vocab_size: int | None = None,
        total_train_tokens: int | None = None,
        total_eval_tokens: int | None = None,
    ) -> Path:
        # Validate dataset with same metadata doesn't already exist
        if self.dataset_exists(
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            dataset_config=dataset_config,
            train_split=train_split,
            eval_split=eval_split,
            text_field=text_field
        ):
            raise FileExistsError("Dataset with the same metadata already exists.")

        # Validate source paths exist
        src_train = Path(src_path_train_bin).expanduser().resolve()
        src_eval = Path(src_path_eval_bin).expanduser().resolve()
        if not src_train.exists():
            raise FileNotFoundError(f"Train bin not found: {src_train}")
        if not src_eval.exists():
            raise FileNotFoundError(f"Eval bin not found: {src_eval}")

        # Create new dataset dir (dataset_<N>)
        dataset_dir = self._make_dataset_dir()
        dst = DatasetArtifacts(dataset_dir)
        dst.ensure_dir()

        shutil.copy2(src_train, dst.train_bin)
        shutil.copy2(src_eval, dst.eval_bin)

        created_at = get_local_iso_timestamp()
        artifacts_path = str(dataset_dir.relative_to(self.artifacts_root).as_posix())

        params = (
            dataset_name, dataset_config, train_split, eval_split, tokenizer_name, text_field,
            vocab_size, total_train_tokens, total_eval_tokens, artifacts_path,
            str(dataset_dir.resolve().as_posix()), created_at
        )
        for i, p in enumerate(params, start=1):
            if isinstance(p, Path):
                raise TypeError(f"Param {i} is Path: {p}")
            
        self.execute(
            """
            INSERT INTO datasets (
                dataset_name,
                dataset_config,
                train_split,
                eval_split,
                tokenizer_name,
                text_field,
                vocab_size,
                total_train_tokens,
                total_eval_tokens,
                artifacts_path,
                dataset_absolute_path,
                created_at
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                dataset_name, 
                dataset_config, 
                train_split, 
                eval_split, 
                tokenizer_name, 
                text_field,
                vocab_size, 
                total_train_tokens, 
                total_eval_tokens, 
                artifacts_path, 
                str(dataset_dir.resolve().as_posix()), 
                created_at
            )
        )

        return dataset_dir

    def delete_dataset(
        self,
        dataset_path: str | Path | None = None,
        dataset_name: str | None = None,
        dataset_config: str | None = None,
        train_split: str | None = None,
        eval_split: str | None = None,
        tokenizer_name: str | None = None,
        text_field: str | None = None,
        confirm: bool = True,
    ) -> None:
        if dataset_path is None:
            min_metadata_provided = (dataset_name is not None) and (tokenizer_name is not None)
            assert min_metadata_provided, "Must provide either dataset_path or dataset_name and tokenizer_name"
            dataset_path = self.find_dataset_path(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                train_split=train_split,
                eval_split=eval_split,
                tokenizer_name=tokenizer_name,
                text_field=text_field,
                raise_if_not_found=False,
            )

        if dataset_path is None:
            raise FileNotFoundError("Dataset not found (no matching metadata and no dataset_path provided).")

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
            "DELETE FROM datasets WHERE artifacts_path=?",
            (str(rel_path.as_posix()),),
        )

    # --- internal ---
    def _make_dataset_dir(self) -> Path:
        next_id = get_next_id(self._DATASET_PREFIX, self.artifacts_root)
        dataset_dir = self.artifacts_root / f"{self._DATASET_PREFIX}{next_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir