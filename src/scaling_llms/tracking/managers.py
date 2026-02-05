from dataclasses import is_dataclass, asdict
import json
import os
from pathlib import Path
from typing import Any, Iterable
from scaling_llms.tracking.trackers import TrackerConfig, TrackerDict, JsonlTracker, TensorBoardTracker
from scaling_llms.tracking.constants import DIRS, METRICS


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
# RunManager
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
        for subdir_name in DIRS.as_list():
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
        for subdir_name in DIRS.as_list():
            (run_root_dir / subdir_name).mkdir()

        return cls(run_root_dir)


    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def log_metrics(self, cat2metrics: dict[str, dict[str, float]], step: int) -> None:
        tracker_dict = self._get_jsonl_trackers(METRICS.as_list())
        for cat, metrics in cat2metrics.items():
            tracker_dict[cat].log_metrics(step, metrics)

    def log_tb(self, cat2metrics: dict[str, dict[str, float]], step: int) -> None:
        tracker_dict = self._get_tb_trackers(METRICS.as_list())
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

        path = self[DIRS.metadata] / filename

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
    def _make_tracker_dict(self, subdir_name: str, categories: Iterable[str]) -> TrackerDict:
        SUBDIR_NAME2TRACKER_CLS = {
            DIRS.metrics: JsonlTracker,
            DIRS.tensorboard: TensorBoardTracker,
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
            self._jsonl_tracker_dict = self._make_tracker_dict(DIRS.metrics, categories)
            
        return self._jsonl_tracker_dict

    def _get_tb_trackers(self, categories: Iterable[str]) -> TrackerDict:
        if self._tb_tracker_dict is None:
            self._tb_tracker_dict = self._make_tracker_dict(DIRS.tensorboard, categories)
        return self._tb_tracker_dict
