from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from scaling_llms.constants import METRIC_CATS
from scaling_llms.registries.runs.artifacts import RunArtifacts
from scaling_llms.utils.loggers import BaseLogger
from scaling_llms.tracking.helpers import (
    log_as_json, 
    log_as_html, 
    log_as_png
)
from scaling_llms.tracking.trackers import (
    TrackerConfig,
    TrackerDict,
    JsonlTracker,
    TensorBoardTracker,
)


class Run:
    """
    Runtime session object.
    Owns trackers; uses RunArtifacts for paths.
    """

    logger = BaseLogger(name="Run")

    def __init__(
        self,
        run_artifacts: RunArtifacts,
        metric_categories: Iterable[str] | None = None,
    ):
        self.artifacts = run_artifacts
        self.metric_categories = (
            list(metric_categories)
            if metric_categories is not None
            else METRIC_CATS.as_list()
        )

        # Expose some convenient properties for direct access
        self.root = run_artifacts.root
        self.metrics_dir = run_artifacts.metrics
        self.metadata_dir = run_artifacts.metadata
        self.checkpoints_dir = run_artifacts.checkpoints
        self.tb_dir = run_artifacts.tb

        # Trackers are lazily initialized on first use, so that we don't create any files until we have to.
        self._metrics_tracker_dict: TrackerDict | None = None
        self._tb_tracker_dict: TrackerDict | None = None

    # ---- lifecycle ----
    def start(self, resume: bool = False) -> None:
        self._get_metrics_trackers()
        self._get_tb_trackers()

        self.logger.info(
            "[init] %s run at %s",
            ("Resuming" if resume else "Started"),
            self.root,
        )

    def close(self) -> None:
        for tracker_dict in (self._metrics_tracker_dict, self._tb_tracker_dict):
            if tracker_dict is not None:
                tracker_dict.close()
        self._metrics_tracker_dict = None
        self._tb_tracker_dict = None
        self.logger.info("[trackers] Closed all trackers.")

    # ---- logging ----
    def log_metrics(self, cat2metrics: dict[str, dict[str, float]], step: int) -> None:
        tracker_dict = self._get_metrics_trackers()
        for cat, metrics in cat2metrics.items():
            tracker_dict[cat].log_metrics(step, metrics)

    def log_tb(self, cat2metrics: dict[str, dict[str, float]], step: int) -> None:
        tracker_dict = self._get_tb_trackers()
        for cat, metrics in cat2metrics.items():
            tracker_dict[cat].log_metrics(step=step, metrics=metrics)

    def log_metadata(
        self,
        obj: Any,
        filename: str,
        format: str = "json",
        subdir_name: str | None = None,
    ) -> Path:
        if format == "json":
            log_fn = log_as_json
        elif format == "png":
            log_fn = log_as_png
        elif format == "html":
            log_fn = log_as_html
        else:
            raise NotImplementedError(
                f"log_metadata format '{format}' is not implemented yet. "
                "Only 'json', 'png', and 'html' are currently supported."
            )

        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        path = self.artifacts.metadata_path(
            filename,
            subdir_name=subdir_name,
            ensure_dir=True,
        )
        self.logger.debug("[metadata] Logging metadata to %s", path)
        return log_fn(obj, path)

    
    def get_metric_path(self, category: str) -> Path: 
        jsonl_tracker = self._metrics_tracker_dict.get(category)
        if jsonl_tracker is None:
            raise ValueError(f"No JsonlTracker found for category '{category}'")
        
        metric_path = jsonl_tracker.get_file_path()
        if (metric_path is None) or not metric_path.exists(): 
            raise FileNotFoundError(f"Metrics file does not exist: {metric_path}") 
        
        return metric_path

    # ---- trackers (internal) ----
    def _make_tracker_dict(
        self, 
        tracker_cls: type[JsonlTracker | TensorBoardTracker],
        log_dir: Path, 
        categories: Iterable[str]
    ) -> TrackerDict:
        tracker_configs = [
            TrackerConfig(
                enabled=True,
                name=cat,
                cls=tracker_cls,
                log_dir=log_dir
            )
            for cat in categories
        ]
        return TrackerDict(tracker_configs)

    def _get_metrics_trackers(self) -> TrackerDict:
        if self._metrics_tracker_dict is None:
            self._metrics_tracker_dict = self._make_tracker_dict(
                tracker_cls=JsonlTracker, 
                log_dir=self.metrics_dir, 
                categories=self.metric_categories
            )

        return self._metrics_tracker_dict

    def _get_tb_trackers(self) -> TrackerDict:
        if self._tb_tracker_dict is None:
            self._tb_tracker_dict = self._make_tracker_dict(
                tracker_cls=TensorBoardTracker, 
                log_dir=self.tb_dir, 
                categories=self.metric_categories
            )
        return self._tb_tracker_dict


