import os
from pathlib import Path
from dataclasses import dataclass

os.environ["SCALING_LLMS_ENV"] = os.getenv("SCALING_LLMS_ENV", "local")

LOCAL_TIMEZONE: str = "Europe/Athens"
DESKTOP_DRIVE_MOUNTPOINT: str = "/Users/vasilis/Library/CloudStorage/GoogleDrive-stylianouvasilis@gmail.com/My Drive"
COLAB_DRIVE_MOUNTPOINT: str = "/content/drive/MyDrive"
DRIVE_SUBDIR_NAME = "ml-experiments"  # subdir within Google Drive where runs and data will be stored
PROJECT_NAME: str = "scaling-llms"
PROJECT_DEV_NAME: str = "scaling-llms-dev"
LOCAL_DATA_DIR = Path.home() / ".local" / "share" / PROJECT_NAME
LOCAL_DEV_DATA_DIR = Path.home() / ".local" / "share" / PROJECT_DEV_NAME
HF_CACHE_DIR_NAME = "huggingface_cache"
TOKENIZED_CACHE_DIR_NAME = "tokenized_cache"
MAX_CACHE_GB = 5 if os.environ["SCALING_LLMS_ENV"] == "local" else 30  # per cache dir (to prevent OOM issues on limited local storage)


# -------------------------
# METRIC TRACKING SCHEMA
# -------------------------
@dataclass(frozen=True)
class SchemaColumns:
    step: str = "step"
    metric: str = "metric"
    value: str = "value"


# -------------------------
# METRIC CATEGORIES
# -------------------------
@dataclass(frozen=True)
class MetricCategories:
    network: str = "network"
    system: str = "system"
    train: str = "train"
    eval: str = "eval"

    def as_list(self) -> list[str]:
        """Return all categories as a list."""
        return [self.network, self.system, self.train, self.eval]


# --------------------------
# RUN FILE & DIRECTORY NAMES
# --------------------------
@dataclass(frozen=True)
class RunFileNames:
    trainer_config: str = "trainer_configs.json"
    data_config: str = "data_configs.json"
    model_config: str = "model_configs.json"
    model_class: str = "model_class.json"
    train_log: str = "train.log"
    data_log: str = "data_log.log"
    best_ckpt: str = "best.pt"
    last_ckpt: str = "latest.pt"

    def as_list(self) -> list[str]:
        """Return all run-related file names as a list.

        The order is deterministic and mirrors the attributes defined on the
        dataclass.
        """
        return [
            self.trainer_config,
            self.data_config,
            self.model_config,
            self.model_class,
            self.train_log,
            self.data_log,
            self.best_ckpt,
            self.last_ckpt,
        ]

    

@dataclass(frozen=True)
class RunDirNames:
    metadata: str = "metadata"
    metrics: str = "metrics"
    checkpoints: str = "checkpoints"
    tensorboard: str = "tb"

    def as_list(self) -> list[str]:
        """Return all directory names as a list."""
        return [self.metadata, self.metrics, self.checkpoints, self.tensorboard]


# --------------------------
# DATA FILE NAMES
# --------------------------
# TODO
@dataclass(frozen=True)
class DataFileNames:
    train_tokens: str = "train.bin"
    eval_tokens: str = "eval.bin"




# --------------------------
# GOOGLE DRIVE DEFAULTS
# --------------------------
@dataclass(frozen=True)
class GoogleDriveDefaults:
    """    
    Example structure on Google Drive:

    {mountpoint}/{drive_subdir}/{project_subdir}/
    ├── data_registry/
    │   ├── datasets.db
    │   └── tokenized_datasets/
    │       ├── <dataset_name_1>/
    │       ├── <dataset_name_2>/
    │       └── ...
    ├── run_registry/
    │   ├── runs.db
    │   └── artifacts/
    │       ├── <experiment_name_1>/
    │       │   ├── <run_1>/
    │       │   └── <run_2>/
    │       ├── <experiment_name_2>/
    │       └── ...
    """
    mountpoint: Path = Path(DESKTOP_DRIVE_MOUNTPOINT if os.environ["SCALING_LLMS_ENV"] == "local" else COLAB_DRIVE_MOUNTPOINT)
    drive_subdir: str = DRIVE_SUBDIR_NAME 
    project_subdir: str = PROJECT_NAME
    run_registry_name: str = "run_registry"
    runs_db_name: str = "runs.db"
    runs_artifacts_subdir: str = "artifacts"
    data_registry_name: str = "data_registry"
    datasets_db_name: str = "datasets.db"
    tokenized_datasets_subdir: str = "tokenized_datasets"



# -------------------------
# INSTANTIATE SINGLETONS
# -------------------------
METRIC_SCHEMA = SchemaColumns()
METRIC_CATS = MetricCategories()
RUN_FILES = RunFileNames()
RUN_DIRS = RunDirNames()
DATA_FILES = DataFileNames()
GOOGLE_DRIVE_DEFAULTS = GoogleDriveDefaults()
