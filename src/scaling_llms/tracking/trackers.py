import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Any

from scaling_llms.constants import METRIC_SCHEMA


# -------------------------
# BASE + STEP TRACKER
# -------------------------
class BaseTracker(ABC):
    enabled: bool
    log_dir: Path | None
    name: str

    @abstractmethod
    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None: ...

    @abstractmethod
    def write(self, step: int, metric_name: str, value: Any) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class StepTracker(BaseTracker):
    def __init__(self, log_dir: Path | None, name: str):
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.name = name
        self.enabled = (self.log_dir is not None)

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return

        for metric_name, value in metrics.items():
            self.write(step, metric_name, value)

# -------------------------
# JSONL TRACKER
# -------------------------
class JsonlTracker(StepTracker):

    def __init__(
        self, 
        log_dir: Path | None, 
        name: str, 
        filename: str | None = None
    ):
        super().__init__(log_dir=log_dir, name=name)
        self.fp = None
        if self.enabled:
            fname = filename if filename is not None else f"{name}.jsonl"
            self.fp = (self.log_dir / fname).open("a", encoding="utf-8")

    def write(self, step: int, metric_name: str, value: Any) -> None:
        if not self.enabled:
            return
        payload = {
            METRIC_SCHEMA.step: int(step),
            METRIC_SCHEMA.metric: metric_name,
            METRIC_SCHEMA.value: value,
        }
        self.fp.write(json.dumps(payload) + "\n")
        self.fp.flush()

    def close(self) -> None:
        if self.enabled and self.fp is not None:
            self.fp.close()
            self.fp = None

# -------------------------
# TENSORBOARD TRACKER
# -------------------------
class TensorBoardTracker(StepTracker):
    def __init__(
        self,
        log_dir: Path | None,
        name: str,
    ):
        super().__init__(log_dir=log_dir, name=name)
        self.w = None
        if self.enabled:
            if SummaryWriter is None:
                raise RuntimeError(
                    "TensorBoard not available (torch.utils.tensorboard.SummaryWriter import failed)."
                )
            log_dir.mkdir(parents=True, exist_ok=True)
            self.w = SummaryWriter(str(log_dir))

    def write(self, step: int, metric_name: str, value: Any) -> None:
        if not self.enabled:
            return
        if isinstance(value, (int, float)) and (value == value):  # skip NaN
            self.w.add_scalar(f"{self.name}/{metric_name}", value, int(step))

    def close(self) -> None:
        if self.enabled and self.w is not None:
            self.w.close()
            self.w = None



# -------------------------
# TRACKER DICT
# -------------------------
@dataclass(frozen=True)
class TrackerConfig:
    """
    Specifies how to build a tracker instance.
    - cls: which tracker class to instantiate (JsonlTracker, TensorBoardTracker, etc.)
    - enabled: toggles tracker on/off
    - log_dir: base directory for logs (None disables)
    - name: prefix/name for this tracker category (e.g., "step", "debug")
    - kwargs: extra constructor kwargs (e.g., filename=..., subdir=...)
    """
    cls: type[StepTracker] = JsonlTracker
    enabled: bool = True
    log_dir: Path | None = None
    name: str = "metrics"
    kwargs: dict[str, Any] | None = None


class TrackerDict(Mapping[str, BaseTracker]):
    def __init__(self, configs: list[TrackerConfig]):
        self._name2tracker: dict[str, BaseTracker] = {}

        for cfg in configs:
            kwargs = {} if cfg.kwargs is None else dict(cfg.kwargs)
            log_dir = cfg.log_dir if cfg.enabled else None

            tracker_obj = cfg.cls(
                log_dir=log_dir,
                name=cfg.name,
                **kwargs,
            )

            self._name2tracker[cfg.name] = tracker_obj
            setattr(self, cfg.name, tracker_obj)

    # --- REQUIRED BY Mapping ---

    def __getitem__(self, key: str) -> BaseTracker:
        return self._name2tracker[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._name2tracker)

    def __len__(self) -> int:
        return len(self._name2tracker)

    # --- API ---

    def get(self, key: str, default=None) -> BaseTracker | None:
        return self._name2tracker.get(key, default)

    def close(self) -> None:
        for lg in self._name2tracker.values():
            lg.close()


# -------------------------
# JSONL TRACKER READER
# -------------------------
class JsonlTrackerReader:
    """
    Reads JSONL logs produced by TrackerDict and exposes them as pandas DataFrames.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.log_dir}")

        # Cache raw records per category (list[dict])
        self._cache_records: dict[str, list[dict[str, Any]]] = {}

        # Cache DataFrames per category
        self._cache_df: dict[str, pd.DataFrame] = {}

    def __getitem__(self, category: str) -> pd.DataFrame:
        df = self._load_df(category)
        if df.empty:
            raise KeyError(
                f"category '{category}' not found or empty. "
                f"Available: {list(self.keys())}"
            )
        return df

    # --- PRIVATE METHODS ---
    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _load_records(self, name: str) -> list[dict[str, Any]]:
        if name not in self._cache_records:
            path = self.log_dir / f"{name}.jsonl"
            self._cache_records[name] = self._read_jsonl(path)
        return self._cache_records[name]

    def _load_df(self, name: str) -> pd.DataFrame:
        if name not in self._cache_df:
            records = self._load_records(name)
            df = pd.DataFrame.from_records(records)

            # Helpful default ordering if present
            for col in (METRIC_SCHEMA.step, METRIC_SCHEMA.metric):
                if col in df.columns:
                    # stable sort by these if they exist
                    pass
            if METRIC_SCHEMA.step in df.columns:
                sort_cols = [METRIC_SCHEMA.step] + ([METRIC_SCHEMA.metric] if METRIC_SCHEMA.metric in df.columns else [])
                df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

            self._cache_df[name] = df
        return self._cache_df[name]

    # --- API ---
    def keys(self) -> Iterable[str]:
        """Return available category names based on files present."""
        return sorted(p.stem for p in self.log_dir.glob("*.jsonl"))

    def get(self, category: str) -> pd.DataFrame:
        """Get DataFrame for a category (empty DataFrame if missing)."""
        path = self.log_dir / f"{category}.jsonl"
        if not path.exists():
            return pd.DataFrame()
        return self._load_df(category)

    def reload(self, category: str | None = None) -> None:
        """Clear cache for one category or all categories."""
        if category is None:
            self._cache_records.clear()
            self._cache_df.clear()
        else:
            self._cache_records.pop(category, None)
            self._cache_df.pop(category, None)