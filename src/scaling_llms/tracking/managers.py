from dataclasses import is_dataclass, asdict, dataclass
from datetime import datetime
import json
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Any, Iterable
from scaling_llms.tracking.trackers import (
    TrackerConfig,
    TrackerDict,
    JsonlTracker,
    TensorBoardTracker,
)
from scaling_llms.constants import (
    RUN_DIRS, 
    METRIC_CATS,
    GOOGLE_DRIVE_DEFAULTS,
    LOCAL_TIMEZONE
)
import sqlite3
from zoneinfo import ZoneInfo


# -----------------------------
# CONSTANTS & DEFAULTS
# -----------------------------



# -----------------------------
# JSON helpers
# -----------------------------
def _json_default(o: Any):
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
    def create_new_run_dir(cls, log_dir: str | Path) -> "RunManager":
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Find next available run_<N>
        existing_ids: list[int] = []
        for p in log_dir.iterdir():
            if p.is_dir() and p.name.startswith(cls._RUN_DIR_PREFIX):
                suffix = p.name[len(cls._RUN_DIR_PREFIX) :]
                if suffix.isdigit():
                    existing_ids.append(int(suffix))

        next_id = max(existing_ids, default=0) + 1
        run_root_dir = log_dir / f"{cls._RUN_DIR_PREFIX}{next_id}"

        # Create directory structure
        run_root_dir.mkdir(exist_ok=False)
        for subdir_name in RUN_DIRS.as_list():
            (run_root_dir / subdir_name).mkdir()

        return cls(run_root_dir)

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def log_metrics(self, cat2metrics: dict[str, dict[str, float]], step: int) -> None:
        tracker_dict = self._get_jsonl_trackers(METRIC_CATS.as_list())
        for cat, metrics in cat2metrics.items():
            tracker_dict[cat].log_metrics(step, metrics)

    def log_tb(self, cat2metrics: dict[str, dict[str, float]], step: int) -> None:
        tracker_dict = self._get_tb_trackers(METRIC_CATS.as_list())
        for cat, metrics in cat2metrics.items():
            tracker_dict[cat].log_metrics(step=step, metrics=metrics)

    def log_metadata(self, obj: Any, filename: str, format: str = "json") -> Path:
        if format != "json":
            raise NotImplementedError(
                f"log_metadata format '{format}' is not implemented yet. "
                "Only 'json' is currently supported."
            )

        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        path = self[RUN_DIRS.metadata] / filename

        return log_as_json(obj, path)

    def close(self) -> None:
        for tracker_dict in (self._jsonl_tracker_dict, self._tb_tracker_dict):
            if tracker_dict is not None:
                tracker_dict.close()
        self._jsonl_tracker_dict = None
        self._tb_tracker_dict = None

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

        log_dir = Path(self[subdir_name])
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

    def __init__(self, db_path: str | Path, artifact_root: str | Path):
        self.db_path = Path(db_path)
        self.artifact_root = Path(artifact_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
        return self.artifact_root / experiment_name

    def get_runs_as_df(self) -> pd.DataFrame:
        query = (
            "SELECT experiment_name, run_name, run_relpath, run_absolute_path, created_at "
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
        # Fetch run_relpath from DB and resolve full path
        with self._connect() as con:
            row = con.execute(
                "SELECT run_relpath FROM runs WHERE experiment_name=? AND run_name=?",
                (experiment_name, run_name),
            ).fetchone()

        if row is None:
            raise KeyError(f"Run not found: ({experiment_name}, {run_name})")

        return self.artifact_root / row[0]

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
        exp_dir = self.get_experiment_dir(experiment_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        run_manager = RunManager.create_new_run_dir(exp_dir)
        
        # Validate the run directory is under the artifact root
        run_absolute_path = run_manager.root.resolve()
        try:
            run_relpath = run_absolute_path.relative_to(self.artifact_root)
        except ValueError:
            run_manager.root.rmdir()
            raise ValueError(
                f"Run directory {run_absolute_path} must be under artifact_root {self.artifact_root}"
            )

        # Register the new run in the DB
        self._register(experiment_name, run_name, run_relpath, run_absolute_path)

        return run_manager

    def delete_run(
        self,
        experiment_name: str,
        run_name: str,
        delete_artifacts: bool = True,
        confirm: bool = True,
    ) -> None:
        # Validate run_dir
        run_dir = self.get_run_dir(experiment_name, run_name).resolve()
        try:
            run_dir.relative_to(self.artifact_root.resolve())
        except ValueError:
            raise ValueError(
                f"Run directory {run_dir} must be under artifact_root {self.artifact_root}"
            )

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
        if delete_artifacts and run_dir.exists():
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
        delete_artifacts: bool = True,
        confirm: bool = True,
    ) -> None:
        # Validate experiment dir
        exp_dir = self.get_experiment_dir(experiment_name).resolve()
        try:
            exp_dir.relative_to(self.artifact_root.resolve())
        except ValueError:
            raise ValueError(
                f"Experiment directory {exp_dir} must be under artifact_root {self.artifact_root}"
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
        if delete_artifacts and exp_dir.exists():
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
                run_relpath        TEXT NOT NULL,
                run_absolute_path  TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                metadata           TEXT,
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

    def _run_relpath_exists(self, run_relpath: str | Path) -> bool:
        run_relpath = str(Path(run_relpath).as_posix())
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM runs WHERE run_relpath=?",
                (run_relpath,),
            ).fetchone()
        return row is not None

    def _register(
        self,
        experiment_name: str,
        run_name: str,
        run_relpath: str | Path,
        run_absolute_path: str | Path,
    ) -> None:
        # Validate run does not already exist (by name or path)
        if self._run_exists(experiment_name, run_name):
            raise ValueError(f"Run already exists: ({experiment_name}, {run_name}).")
        
        # Validate run_relpath is unique (no other run has the same relative path)
        run_relpath = str(Path(run_relpath).as_posix())
        run_absolute_path = str(Path(run_absolute_path))
        if self._run_relpath_exists(run_relpath):
            raise ValueError(f"Run path already registered: {run_relpath}")
        
        # Register the run in the DB
        created_at = datetime.now(ZoneInfo(LOCAL_TIMEZONE)).isoformat()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO runs (
                    experiment_name, 
                    run_name, 
                    run_relpath, 
                    run_absolute_path, 
                    created_at
                ) 
                VALUES (?,?,?,?,?)
                """,
                (experiment_name, run_name, run_relpath, run_absolute_path, created_at),
            )
            con.commit()


# -----------------------------
# GOOGLE DRIVE CONFIGS
# -----------------------------
@dataclass
class GoogleDriveConfigs:
    drive_subdir: str = GOOGLE_DRIVE_DEFAULTS.drive_subdir
    data_subdir: str = GOOGLE_DRIVE_DEFAULTS.data_subdir
    mountpoint: str | Path | None = None
    drive_root_name: str | None = GOOGLE_DRIVE_DEFAULTS.drive_root_name
    db_name: str = GOOGLE_DRIVE_DEFAULTS.db_name
    artifact_subdir: str = GOOGLE_DRIVE_DEFAULTS.artifact_subdir
    auto_mount: bool = True
    force_remount: bool = False

    def __post_init__(self) -> None:
        if self.mountpoint is None:
            self.mountpoint = (
                GOOGLE_DRIVE_DEFAULTS.colab_mountpoint
                if os.environ.get("SCALING_LLMS_ENV") == "colab"
                else GOOGLE_DRIVE_DEFAULTS.desktop_mountpoint
            )

        self._mountpoint = Path(self.mountpoint)

        if self.auto_mount:
            self._mount_if_needed(force_remount=self.force_remount)

        self.drive_root = self._resolve_drive_root()
        self.project_root = self.drive_root / self.drive_subdir
        self.artifact_root = self.project_root / self.artifact_subdir
        self.data_root = self.project_root / self.data_subdir
        self.db_path = self.project_root / self.db_name

    def _mount_if_needed(self, force_remount: bool = False) -> None:
        if self._drive_root_exists():
            return

        try:
            from google.colab import drive  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Google Drive is not mounted and google.colab is not available. "
                "Mount Drive manually or set auto_mount=False with a valid mountpoint."
            ) from exc

        drive.mount(str(self._mountpoint), force_remount=force_remount)

    def _drive_root_exists(self) -> bool:
        if self.drive_root_name:
            return (self._mountpoint / self.drive_root_name).exists()

        return (self._mountpoint / "MyDrive").exists() or (
            self._mountpoint / "My Drive"
        ).exists()

    def _resolve_drive_root(self) -> Path:
        if self.drive_root_name:
            drive_root = self._mountpoint / self.drive_root_name
            if not drive_root.exists():
                raise FileNotFoundError(f"Drive root not found: {drive_root}")
            return drive_root

        mydrive = self._mountpoint / "MyDrive"
        if mydrive.exists():
            return mydrive

        my_drive = self._mountpoint / "My Drive"
        if my_drive.exists():
            return my_drive

        raise FileNotFoundError(
            f"Neither 'MyDrive' nor 'My Drive' found under {self._mountpoint}"
        )


# -----------------------------
# GOOGLE DRIVE RUN REGISTRY
# -----------------------------
class GoogleDriveRunRegistry(BaseRunRegistry):
    """
    Wrapper around BaseRunRegistry that stores the registry + artifacts in Google Drive.

    In Colab, it can mount Drive via OAuth using google.colab.drive.mount.
    Locally, set auto_mount=False and ensure the Drive folder is already mounted/synced.
    """

    def __init__(
        self,
        configs: GoogleDriveConfigs | None = None,
        **overrides: Any,
    ) -> None:
        configs = configs or GoogleDriveConfigs(**overrides)
        super().__init__(db_path=configs.db_path, artifact_root=configs.artifact_root)


# -----------------------------
# GOOGLE DRIVE DATA REGISTRY
# -----------------------------
class GoogleDriveDataRegistry:
    """
    Manages a project data root in Google Drive at:
        <drive_root>/<drive_subdir>/<data_subdir>

    Useful for organizing datasets and artifacts alongside run registries.
    """

    def __init__(
        self,
        configs: GoogleDriveConfigs | None = None,
        **overrides: Any,
    ) -> None:
        self.configs = configs or GoogleDriveConfigs(**overrides)
        self.data_root = self.configs.data_root
        self.data_root.mkdir(parents=True, exist_ok=True)

    def get_data_root(self) -> Path:
        return self.data_root

    def get_dataset_dir(self, name: str) -> Path:
        path = self.data_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def copy_local_to_data_root(
        self,
        local_path: str | Path,
        relative_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Copy a local file or directory into data_root at the given relative path.

        Works for both local and Colab since Drive is mounted to the filesystem.
        """
        src = Path(local_path).expanduser().resolve()
        dst = (self.data_root / relative_path).expanduser().resolve()

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        if dst.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {dst}")
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        return dst

    def copy_data_root_to_local(
        self,
        relative_path: str | Path,
        local_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Copy a file or directory from data_root to a local destination path.

        Works for both local and Colab since Drive is mounted to the filesystem.
        """
        src = (self.data_root / relative_path).expanduser().resolve()
        dst = Path(local_path).expanduser().resolve()

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        if dst.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {dst}")
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        return dst
