import os
from pathlib import Path
from dataclasses import dataclass

os.environ["SCALING_LLMS_ENV"] = os.getenv("SCALING_LLMS_ENV", "local")

LOCAL_TIMEZONE: str = "Europe/Athens"


# -------------------------
# GOOGLE DRIVE SETTINGS
# -------------------------
DESKTOP_DRIVE_MOUNTPOINT: str = "/Users/vasilis/Library/CloudStorage/GoogleDrive-stylianouvasilis@gmail.com"
COLAB_DRIVE_MOUNTPOINT: str = "/content/drive"
DESKTOP_DRIVE_SUBDIR = "My Drive/ml-experiments"  # subdir within Google Drive where runs and data will be stored
COLAB_DRIVE_SUBDIR = "MyDrive/ml-experiments" 


# -------------------------
# PROJECT SETTINGS
# -------------------------
PROJECT_NAME: str = "scaling-llms"
PROJECT_DEV_NAME: str = "scaling-llms-dev"
LOCAL_DATA_DIR = Path.home() / ".local" / "share" / PROJECT_NAME
LOCAL_DEV_DATA_DIR = Path.home() / ".local" / "share" / PROJECT_DEV_NAME
HF_CACHE_DIR_NAME = "huggingface_cache"
TOKENIZED_CACHE_DIR_NAME = "tokenized_cache"
MAX_CACHE_GB = 5 if os.environ["SCALING_LLMS_ENV"] == "local" else 30  # per cache dir (to prevent OOM issues on limited local storage)


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
# RUN ARTIFACTS FILE NAMES (CKPT & METADATA)
# --------------------------
@dataclass(frozen=True)
class CheckpointFileNames:
    best_ckpt: str = "best.pt"
    last_ckpt: str = "latest.pt"

    def as_list(self) -> list[str]:
        return [
            self.best_ckpt,
            self.last_ckpt,
        ]


@dataclass(frozen=True)
class MetadataFileNames:
    trainer_config: str = "trainer_configs.json"
    dataset_id: str = "dataset_id.json"
    dataloader_config: str = "dataloader_configs.json"
    model_config: str = "model_configs.json"
    model_class: str = "model_class.json"
    train_log: str = "train.log"
    data_log: str = "data.log"

    def as_list(self) -> list[str]:
        return [
            self.trainer_config,
            self.dataset_id,
            self.dataloader_config,
            self.model_config,
            self.model_class,
            self.train_log,
            self.data_log,
        ]


# --------------------------
# DATA ARTIFACTS FILE NAMES
# --------------------------
@dataclass(frozen=True)
class DatasetFileNames:
    train_tokens: str = "train.bin"
    eval_tokens: str = "eval.bin"
    dataset_info: str = "dataset_info.json"


# -------------------------
# INSTANTIATE SINGLETONS
# -------------------------
METRIC_CATS = MetricCategories() 
DATASET_FILES = DatasetFileNames()
CKPT_FILES = CheckpointFileNames()
METADATA_FILES = MetadataFileNames()