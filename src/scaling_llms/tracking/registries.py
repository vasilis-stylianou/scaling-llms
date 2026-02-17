from dataclasses import is_dataclass, asdict, dataclass
from datetime import datetime
import json
import os
import pandas as pd
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from scaling_llms.constants import (
    DATA_FILES,
    GOOGLE_DRIVE_DEFAULTS,
    LOCAL_TIMEZONE,
    METRIC_CATS,
    RUN_DIRS, 
)
from scaling_llms.tracking.trackers import (
    TrackerConfig,
    TrackerDict,
    JsonlTracker,
    TensorBoardTracker,
)
from scaling_llms.utils.config import BaseJsonConfig
from scaling_llms.utils.loggers import BaseLogger


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def _json_default(o: Any):
    if isinstance(o, BaseJsonConfig):
        return o.to_json()
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def log_as_json(obj, path) -> Path:
    path = Path(path)
    if path.exists():
        return path

    if isinstance(obj, BaseJsonConfig):
        obj = obj.to_json()

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        f.write("\n")

    os.replace(tmp, path)
    return path

def _get_next_id(prefix: str, parent_dir: Path) -> int:
    existing_ids: list[int] = []
    for p in parent_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suffix = p.name[len(prefix) :]
            if suffix.isdigit():
                existing_ids.append(int(suffix))

    return max(existing_ids, default=0) + 1


# -----------------------------
# RUN MANAGER
# -----------------------------
class RunManager:
    """
    Creates and owns a run directory with a standard folder layout.
    Exposes:
      - log_metrics(cat2metrics, step=None)
      - log_tb(cat2metrics, step=None)
      - log_metadata(obj, filename)
    """

    _RUN_DIR_PREFIX = "run_"

    logger = BaseLogger(name="RunManager")

    def __init__(self, root: Path):
        # Run's root path
        self.root = Path(root)

        # Internal mapping
        self._subdir_name2path: dict[str, Path] = {}

        # Run's sub dir paths
        for subdir_name in RUN_DIRS.as_list():
            path = self.root / subdir_name
            self._subdir_name2path[subdir_name] = path
            setattr(self, subdir_name, path)

        # Lazy tracker dicts
        self._jsonl_tracker_dict: TrackerDict | None = None
        self._tb_tracker_dict: TrackerDict | None = None

    def __bool__(self) -> bool:
        return self.root is not None

    def __getitem__(self, name: str) -> Path:
        return self._subdir_name2path[name]

    def __contains__(self, name: str) -> bool:
        return name in self._subdir_name2path

    # -----------------------------
    # FACTORIES
    # -----------------------------
    @classmethod
    def create_new_run_dir(cls, exp_dir: str | Path) -> "RunManager":
        exp_dir = Path(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        cls.logger.info("[init] Creating new run directory under %s", exp_dir)

        # Create path for next available run (e.g. run_<N>)
        next_id = _get_next_id(cls._RUN_DIR_PREFIX, exp_dir)
        run_root_dir = exp_dir / f"{cls._RUN_DIR_PREFIX}{next_id}"
        run_root_dir.mkdir(exist_ok=False)

        cls.logger.info("[init] Created run directory %s", run_root_dir)

        # Create directory structure
        for subdir_name in RUN_DIRS.as_list():
            subdir = run_root_dir / subdir_name
            subdir.mkdir()
            cls.logger.info("[init] Created run sub-directory '%s/'", subdir.name)

        return cls(run_root_dir)

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def get_run_dir(self) -> Path:
        if not self.root.exists():
            raise FileNotFoundError(f"Run directory does not exist: {self.root}")
        return self.root

    def get_metrics_dir(self) -> Path:
        metrics_dir = self.root / RUN_DIRS.metrics 
        if not metrics_dir.exists(): 
            raise FileNotFoundError(f"Metrics directory does not exist: {metrics_dir}") 
        return metrics_dir
    
    def get_metadata_dir(self) -> Path:
        metadata_dir = self.root / RUN_DIRS.metadata
        if not metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory does not exist: {metadata_dir}")
        
        return metadata_dir 
    
    def get_checkpoint_dir(self) -> Path:
        ckpt_dir = self.root / RUN_DIRS.checkpoints
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
        
        return ckpt_dir 
    
    def get_tb_dir(self) -> Path:
        tb_dir = self.root / RUN_DIRS.tensorboard
        if not tb_dir.exists():
            raise FileNotFoundError(f"TensorBoard directory does not exist: {tb_dir}")
        
        return tb_dir

    def get_metric_path(self, category: str) -> Path: 
        jsonl_tracker = self._jsonl_tracker_dict.get(category)
        if jsonl_tracker is None:
            raise ValueError(f"No JsonlTracker found for category '{category}'")
        
        metric_path = jsonl_tracker.get_file_path()
        if (metric_path is None) or not metric_path.exists(): 
            raise FileNotFoundError(f"Metrics file does not exist: {metric_path}") 
        
        return metric_path
    
    def get_metadata_path(self, filename: str) -> Path: 
        
        metadata_path = self.get_metadata_dir() / filename
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

        return metadata_path
    
    def get_checkpoint_path(self, filename: str) -> Path: 
        
        ckpt_path = self.get_checkpoint_dir() / filename
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

        return ckpt_path
    
    def log_metrics(self, cat2metrics: dict[str, dict[str, float]], step: int) -> None:
        tracker_dict = self._get_jsonl_trackers(METRIC_CATS.as_list())
        for cat, metrics in cat2metrics.items():
            self.logger.debug("[trackers] Logging metrics for category '%s' at step %d", cat, step)
            tracker_dict[cat].log_metrics(step, metrics)

    def log_tb(self, cat2metrics: dict[str, dict[str, float]], step: int) -> None:
        tracker_dict = self._get_tb_trackers(METRIC_CATS.as_list())
        for cat, metrics in cat2metrics.items():
            self.logger.debug("[trackers] Logging tensorboard metrics for category '%s' at step %d", cat, step)
            tracker_dict[cat].log_metrics(step=step, metrics=metrics)

    def log_metadata(self, obj: Any, filename: str, format: str = "json") -> Path:
        if format != "json":
            raise NotImplementedError(
                f"log_metadata format '{format}' is not implemented yet. "
                "Only 'json' is currently supported."
            )

        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        path = self.get_metadata_dir() / filename

        self.logger.debug("[metadata] Logging metadata to %s", path)

        return log_as_json(obj, path)

    def start(self, resume=False) -> None:
        """
        Ensure run directories exist and initialize trackers.
        """
        # If resume is True, directories should already exist; if False, create them
        try:
            self.root.mkdir(parents=True, exist_ok=resume)
            for path in self._subdir_name2path.values():
                path.mkdir(parents=True, exist_ok=resume)
        except FileExistsError:
            raise ValueError(f"Run directory already exists: {self.root}. Set resume=True to resume run.")

        # Initialize JSONL and TensorBoard trackers for all metric categories
        self._get_jsonl_trackers(METRIC_CATS.as_list())
        self._get_tb_trackers(METRIC_CATS.as_list())

        self.logger.info("[init] %s run at %s", ("Resuming" if resume else "Started"), self.root)

    def close(self) -> None:
        for tracker_dict in (self._jsonl_tracker_dict, self._tb_tracker_dict):
            if tracker_dict is not None:
                tracker_dict.close()
        self._jsonl_tracker_dict = None
        self._tb_tracker_dict = None

        self.logger.info("[trackers] Closed all trackers.")

    # -----------------------------
    # Tracker creation (internal)
    # -----------------------------
    def _make_tracker_dict(
        self, subdir_name: str, categories: Iterable[str]
    ) -> TrackerDict:
        SUBDIR_NAME2TRACKER_CLS = {
            RUN_DIRS.metrics: JsonlTracker,
            RUN_DIRS.tensorboard: TensorBoardTracker,
        }
        if subdir_name not in SUBDIR_NAME2TRACKER_CLS:
            raise ValueError(f"Invalid subdir_name; got {subdir_name}")

        log_dir = self.root / subdir_name
        if not log_dir.exists():
            raise FileNotFoundError(f"Expected directory does not exist: {log_dir}")

        tracker_configs = [
            TrackerConfig(
                cls=SUBDIR_NAME2TRACKER_CLS[subdir_name],
                enabled=True,
                log_dir=log_dir,
                name=cat,
            )
            for cat in categories
        ]
        return TrackerDict(tracker_configs)

    def _get_jsonl_trackers(self, categories: Iterable[str]) -> TrackerDict:
        if self._jsonl_tracker_dict is None:
            self._jsonl_tracker_dict = self._make_tracker_dict(RUN_DIRS.metrics, categories)

        return self._jsonl_tracker_dict

    def _get_tb_trackers(self, categories: Iterable[str]) -> TrackerDict:
        if self._tb_tracker_dict is None:
            self._tb_tracker_dict = self._make_tracker_dict(
                RUN_DIRS.tensorboard, categories
            )
        return self._tb_tracker_dict


# -----------------------------
# BASE RUN REGISTRY
# -----------------------------
class BaseRunRegistry:
    """
    Need a registry to keep track of runs and their directories. This is the base class that defines the interface.
    The registry should be able to:
        - Register a new run with its experiment name, run name, and run directory.
        - Retrieve the run directory given the experiment name and run name.
        - List all runs and their metadata (e.g., creation time).
    """

    def __init__(self, db_path: str | Path, artifacts_root: str | Path):
        self.db_path = Path(db_path)
        self.artifacts_root = Path(artifacts_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # --- API ---
    def connect_run(
        self,
        experiment_name: str,
        run_name: str,
    ) -> RunManager:
        run_dir = self.get_run_dir(experiment_name, run_name)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return RunManager(run_dir)
    
    def get_experiment_dir(self, experiment_name: str) -> Path:
        exp_dir = self.artifacts_root / experiment_name
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
        return exp_dir

    def get_runs_as_df(self) -> pd.DataFrame:
        query = (
            "SELECT * "
            "FROM runs ORDER BY experiment_name, created_at"
        )
        with self._connect() as con:
            df = pd.read_sql_query(query, con)

        if not df.empty:
            created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
            df["created_at"] = created_at.dt.tz_convert(LOCAL_TIMEZONE)

        return df
    
    def get_run_dir(
        self,
        experiment_name: str,
        run_name: str,
    ) -> Path:
        # Fetch artifacts_path from DB and resolve full path
        with self._connect() as con:
            row = con.execute(
                "SELECT artifacts_path FROM runs WHERE experiment_name=? AND run_name=?",
                (experiment_name, run_name),
            ).fetchone()

        if row is None:
            raise FileNotFoundError(f"Run not found: ({experiment_name}, {run_name})")

        return self.artifacts_root / row[0]

    def start_run(
        self,
        experiment_name: str,
        run_name: str,
        resume: bool = False,
    ) -> RunManager:
        # Resume existing run if requested
        if self._run_exists(experiment_name, run_name):
            if resume:
                return self.connect_run(experiment_name, run_name)
            raise ValueError(
                f"Run already exists: ({experiment_name}, {run_name}). "
                "Set resume=True to reuse it."
            )

        # Create a new run directory 
        exp_dir = self.artifacts_root / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        run_manager = RunManager.create_new_run_dir(exp_dir)
        
        # Validate the run directory is under the artifact root
        run_absolute_path = run_manager.root.resolve()
        try:
            artifacts_path = run_absolute_path.relative_to(self.artifacts_root)
        except ValueError:
            run_manager.root.rmdir()
            raise ValueError(
                f"Run directory {run_absolute_path} must be under artifacts_root {self.artifacts_root}"
            )

        # Register the new run in the DB
        self._register(experiment_name, run_name, artifacts_path, run_absolute_path)

        return run_manager

    def delete_run(
        self,
        experiment_name: str,
        run_name: str,
        confirm: bool = True,
    ) -> None:
        
        run_dir = self.get_run_dir(experiment_name, run_name).resolve()
        
        # Confirmation prompt
        if confirm:
            response = input(
                f"Are you sure you want to delete run '{run_name}' "
                f"from experiment '{experiment_name}'? Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        # Delete Run's artifacts
        if run_dir.exists():
            shutil.rmtree(run_dir)

        # Delete Run from DB
        with self._connect() as con:
            con.execute(
                "DELETE FROM runs WHERE experiment_name=? AND run_name=?",
                (experiment_name, run_name),
            )
            con.commit()

    def delete_experiment(
        self,
        experiment_name: str,
        confirm: bool = True,
    ) -> None:
        # Validate experiment dir
        exp_dir = self.get_experiment_dir(experiment_name).resolve()
        try:
            exp_dir.relative_to(self.artifacts_root.resolve())
        except ValueError:
            raise ValueError(
                f"Experiment directory {exp_dir} must be under artifacts_root {self.artifacts_root}"
            )

        # Confirmation prompt
        if confirm:
            response = input(
                f"Are you sure you want to delete experiment '{experiment_name}'? "
                f"Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        # Delete all run artifacts for the experiment
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

        # Remove all runs for this experiment from the DB
        with self._connect() as con:
            con.execute(
                "DELETE FROM runs WHERE experiment_name=?",
                (experiment_name,),
            )
            con.commit()

    # --- Internal DB methods ---
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                experiment_name    TEXT NOT NULL,
                run_name           TEXT NOT NULL,
                artifacts_path     TEXT NOT NULL,
                run_absolute_path  TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                other_data         TEXT,
                PRIMARY KEY (experiment_name, run_name)
            );
            """)
            con.commit()

    def _run_exists(self, experiment_name: str, run_name: str) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM runs WHERE experiment_name=? AND run_name=?",
                (experiment_name, run_name),
            ).fetchone()
        return row is not None

    def _artifacts_path_exists(self, artifacts_path: str | Path) -> bool:
        artifacts_path = str(Path(artifacts_path).as_posix())
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM runs WHERE artifacts_path=?",
                (artifacts_path,),
            ).fetchone()
        return row is not None

    def _register(
        self,
        experiment_name: str,
        run_name: str,
        artifacts_path: str | Path,
        run_absolute_path: str | Path,
    ) -> None:
        # Validate run does not already exist (by name or path)
        if self._run_exists(experiment_name, run_name):
            raise ValueError(f"Run already exists: ({experiment_name}, {run_name}).")
        
        # Validate artifacts_path is unique (no other run has the same relative path)
        artifacts_path = str(Path(artifacts_path).as_posix())
        run_absolute_path = str(Path(run_absolute_path))
        if self._artifacts_path_exists(artifacts_path):
            raise ValueError(f"Run path already registered: {artifacts_path}")
        
        # Register the run in the DB
        created_at = datetime.now(ZoneInfo(LOCAL_TIMEZONE)).isoformat()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO runs (
                    experiment_name, 
                    run_name, 
                    artifacts_path, 
                    run_absolute_path, 
                    created_at
                ) 
                VALUES (?,?,?,?,?)
                """,
                (experiment_name, run_name, artifacts_path, run_absolute_path, created_at),
            )
            con.commit()


# -----------------------------
# BASE DATA REGISTRY
# -----------------------------
class BaseDataRegistry:
    """
    Registry for dataset artifacts stored under a datasets/ directory.

    Tracks dataset metadata in an SQLite DB.
    """

    _DATASET_PREFIX = "dataset_"

    def __init__(self, db_path: str | Path, datasets_root: str | Path):
        self.db_path = Path(db_path)
        self.datasets_root = Path(datasets_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # --- API ---
    def get_datasets_as_df(self) -> pd.DataFrame:
        query = (
            "SELECT * "
            "FROM datasets ORDER BY dataset_name, created_at"
        )
        
        with self._connect() as con:
            df = pd.read_sql_query(query, con)

        if not df.empty:
            created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
            df["created_at"] = created_at.dt.tz_convert(LOCAL_TIMEZONE)

        return df
    
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
            src_dir = (self.datasets_root / src_dir).resolve()
        else:
            src_dir = src_dir.resolve()

        train_src = src_dir / DATA_FILES.train_tokens
        eval_src = src_dir / DATA_FILES.eval_tokens

        if not train_src.exists():
            raise FileNotFoundError(f"Train bin not found: {train_src}")
        if not eval_src.exists():
            raise FileNotFoundError(f"Eval bin not found: {eval_src}")

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

        shutil.copy2(train_src, train_dst)
        shutil.copy2(eval_src, eval_dst)

        return train_dst, eval_dst

    def find_dataset_path(
        self,
        dataset_name: str,
        dataset_config: str | None = None,
        train_split: str | None = None,
        eval_split: str | None = None,
        raise_if_not_found: bool = True,
    ) -> Path | None:
        """
        Find dataset_path from metadata and copy train/eval bins to local_path.
        """
        with self._connect() as con:
            row = con.execute(
                """
                SELECT dataset_path FROM datasets
                WHERE dataset_name=? AND dataset_config IS ?
                  AND train_split IS ? AND eval_split IS ?
                """,
                (dataset_name, dataset_config, train_split, eval_split),
            ).fetchone()

        if (row is None) and raise_if_not_found:
            raise FileNotFoundError(
                "Dataset not found for metadata: "
                f"({dataset_name}, {dataset_config}, {train_split}, {eval_split})"
            )
        
        return self.datasets_root / Path(row[0]) if row is not None else None
    
    def register_dataset(
        self,
        src_path_train_bin: str | Path,
        src_path_eval_bin: str | Path,
        dataset_name: str,
        dataset_config: str | None = None,
        train_split: str | None = None,
        eval_split: str | None = None,
    ) -> Path:
        """
        Copy train/eval memmaps into datasets/ and record metadata in the DB.
        """
        # Validate dataset with same metadata doesn't already exist
        existing_path = self.find_dataset_path(
            dataset_name,
            dataset_config,
            train_split,
            eval_split,
            raise_if_not_found=False,
        )     
        if existing_path is not None:
            raise FileExistsError(f"Dataset with the same metadata already exists at: {existing_path}")

        # Validate source paths exist
        src_train = Path(src_path_train_bin).expanduser().resolve()
        src_eval = Path(src_path_eval_bin).expanduser().resolve()
        if not src_train.exists():
            raise FileNotFoundError(f"Train bin not found: {src_train}")
        if not src_eval.exists():
            raise FileNotFoundError(f"Eval bin not found: {src_eval}")

        # Create a new dataset directory with an auto-incremented name like dataset_1, dataset_2, etc.
        dataset_path = self._make_dataset_path()

        # Copy memmaps into dataset directory 
        # (note: these can be large files, so we use copy2 to preserve metadata and be efficient when possible)
        shutil.copy2(src_train, dataset_path / DATA_FILES.train_tokens)
        shutil.copy2(src_eval, dataset_path / DATA_FILES.eval_tokens)

        # Register the dataset in the DB with its metadata and relative path
        created_at = datetime.now(ZoneInfo(LOCAL_TIMEZONE)).isoformat()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO datasets (
                    dataset_name,
                    dataset_config,
                    train_split,
                    eval_split,
                    dataset_path,
                    created_at
                )
                VALUES (?,?,?,?,?,?)
                """,
                (
                    dataset_name,
                    dataset_config,
                    train_split,
                    eval_split,
                    str(dataset_path.relative_to(self.datasets_root)),
                    created_at,
                ),
            )
            con.commit()

        return dataset_path

    def delete_dataset(
        self,
        dataset_path: str | Path | None = None, 
        dataset_name: str | None = None,
        dataset_config: str | None = None,
        train_split: str | None = None,
        eval_split: str | None = None,
        confirm: bool = True,
    ) -> None:
        """
        Delete a dataset directory and its DB row.
        """
        if dataset_path is None:
            assert dataset_name is not None, "Must provide either dataset_path or dataset_name"
            dataset_path = self.find_dataset_path(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                train_split=train_split,
                eval_split=eval_split,
                raise_if_not_found=False,
            )
        
        path = Path(dataset_path) # type: ignore
        rel_path = path
        if path.is_absolute():
            rel_path = path.relative_to(self.datasets_root)

        if confirm:
            response = input(
                f"Are you sure you want to delete dataset at '{rel_path}'? "
                "Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        abs_path = self.datasets_root / rel_path
        if abs_path.exists():
            shutil.rmtree(abs_path)

        with self._connect() as con:
            con.execute(
                "DELETE FROM datasets WHERE dataset_path=?",
                (str(rel_path),),
            )
            con.commit()

    # --- Internal DB methods ---
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_name   TEXT NOT NULL,
                    dataset_config TEXT,
                    train_split    TEXT,
                    eval_split     TEXT,
                    dataset_path   TEXT NOT NULL,
                    created_at     TEXT NOT NULL
                );
                """
            )
            con.commit()

    def _make_dataset_path(self) -> Path:

        # Find next available id (e.g. dataset_<N>)
        next_id = _get_next_id(self._DATASET_PREFIX, self.datasets_root)

        # Create dataset directory
        dataset_path = self.datasets_root / f"{self._DATASET_PREFIX}{next_id}"
        dataset_path.mkdir(parents=True, exist_ok=True)

        return dataset_path


# -----------------------------
# GOOGLE DRIVE REGISTRIES
# -----------------------------
@dataclass
class GoogleDriveConfigs:
    
    mountpoint: str | Path = GOOGLE_DRIVE_DEFAULTS.mountpoint
    drive_subdir: str = GOOGLE_DRIVE_DEFAULTS.drive_subdir
    project_subdir: str = GOOGLE_DRIVE_DEFAULTS.project_subdir
    run_registry_name: str = GOOGLE_DRIVE_DEFAULTS.run_registry_name
    runs_db_name: str = GOOGLE_DRIVE_DEFAULTS.runs_db_name
    runs_artifacts_subdir: str = GOOGLE_DRIVE_DEFAULTS.runs_artifacts_subdir
    data_registry_name: str = GOOGLE_DRIVE_DEFAULTS.data_registry_name
    datasets_db_name: str = GOOGLE_DRIVE_DEFAULTS.datasets_db_name
    tokenized_datasets_subdir: str = GOOGLE_DRIVE_DEFAULTS.tokenized_datasets_subdir
    auto_mount: bool = True
    force_remount: bool = False

    def __post_init__(self) -> None:
        
        # Resolve mountpoint and drive root paths
        self.mountpoint = Path(self.mountpoint)
        self.drive_root = Path(self.mountpoint) / self.drive_subdir

        # Validate drive root exists or can be mounted
        if not self.drive_root.exists():
            
            # Mount to Drive if we're in Colab and auto_mount is enabled; otherwise, expect the drive to already be mounted
            if (os.environ["SCALING_LLMS_ENV"] == "colab") and self.auto_mount:
                try:
                    from google.colab import drive  # type: ignore
                except Exception as exc:
                    raise RuntimeError(
                        "Google Drive is not mounted and google.colab is not available. "
                        "Mount Drive manually or set auto_mount=False with a valid mountpoint."
                    ) from exc

                drive.mount(str(self.mountpoint), force_remount=self.force_remount)

            # Create the drive root directory if it doesn't exist
            self.drive_root.mkdir(parents=True, exist_ok=True)

        # Create the project root directory if it doesn't exist
        self.project_root = self.drive_root / self.project_subdir
        self.project_root.mkdir(parents=True, exist_ok=True)

        # Set up all registry paths based on the drive root and project subdir
        self.run_registry = self.project_root / self.run_registry_name
        self.runs_artifacts_root = self.run_registry / self.runs_artifacts_subdir
        self.runs_db_path = self.run_registry / self.runs_db_name
        self.data_registry = self.project_root / self.data_registry_name
        self.datasets_db_path = self.data_registry / self.datasets_db_name
        self.tokenized_datasets_root = self.data_registry / self.tokenized_datasets_subdir


class GoogleDriveRunRegistry(BaseRunRegistry):
    """
    Wrapper around BaseRunRegistry that stores the runs + artifacts in Google Drive.

    In Colab, it can mount Drive via OAuth using google.colab.drive.mount.
    Locally, set auto_mount=False and ensure the Drive folder is already mounted/synced.

    Stores runs under:
        - Metadata: <drive_root>/<project_subdir>/run_registry/<db_name>.db
        - Artifacts: <drive_root>/<project_subdir>/run_registry/<artifacts_subdir>
    """

    def __init__(
        self,
        configs: GoogleDriveConfigs | None = None,
        **overrides: Any,
    ) -> None:
        configs = configs or GoogleDriveConfigs(**overrides)
        self.project_root = configs.project_root
        self.registry_root = configs.run_registry
        super().__init__(db_path=configs.runs_db_path, artifacts_root=configs.runs_artifacts_root)


class GoogleDriveDataRegistry(BaseDataRegistry):
    """
    Data registry backed by Google Drive.

    Stores datasets under:
        - Metadata: <drive_root>/<project_subdir>/data_registry/<db_name>
        - Tokenized Datasets: <drive_root>/<project_subdir>/data_registry/<datasets_subdir>
    """

    def __init__(
        self,
        configs: GoogleDriveConfigs | None = None,
        **overrides: Any,
    ) -> None:
        configs = configs or GoogleDriveConfigs(**overrides)
        self.project_root = configs.project_root
        self.registry_root = configs.data_registry
        super().__init__(db_path=configs.datasets_db_path, datasets_root=configs.tokenized_datasets_root)

    
