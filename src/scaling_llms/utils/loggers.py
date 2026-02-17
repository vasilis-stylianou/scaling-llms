from __future__ import annotations

import dataclasses
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import zoneinfo

from scaling_llms.constants import LOCAL_TIMEZONE


class TimezoneFormatter(logging.Formatter):
    """Custom formatter that converts timestamps to a specific timezone."""

    def __init__(self, fmt: str = None, datefmt: str = None, timezone: str = LOCAL_TIMEZONE):
        super().__init__(fmt, datefmt)
        self.timezone = zoneinfo.ZoneInfo(timezone)

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=self.timezone)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def setup_root_logging(
    *,
    level: int = logging.INFO,
    stream=sys.stdout,
    fmt: str = "%(asctime)s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = True,
) -> None:
    """
    Works reliably in scripts + VSCode + Jupyter/Colab.

    - force=True replaces any existing handlers (needed in notebooks)
    - fmt includes timestamp with LOCAL_TIMEZONE
    """
    formatter = TimezoneFormatter(fmt=fmt, datefmt=datefmt)
    try:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logging.basicConfig(
            level=level,
            handlers=[handler],
            force=force,  # Python 3.8+
        )
    except TypeError:
        # Fallback for older Python (no force=)
        root = logging.getLogger()
        root.setLevel(level)
        for h in list(root.handlers):
            root.removeHandler(h)
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        root.addHandler(handler)


@dataclass(slots=True)
class BaseLogger:
    name: str = "Logger"
    log_dir: Path | None = None
    level: int = logging.INFO
    propagate: bool = True
    file_name: str | None = None  # if set, attach a file handler to log_dir/file_name
    fmt: str = "%(asctime)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    _logger: logging.Logger = dataclasses.field(init=False, repr=False)
    _file_handler: logging.Handler | None = dataclasses.field(default=None, init=False, repr=False)


    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)
        self._logger.propagate = self.propagate

        # Notebook-safe: remove any existing file handlers for this logger name
        for h in list(self._logger.handlers):
            if isinstance(h, logging.FileHandler):
                self._logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        if self.file_name is not None:
            if self.log_dir is None:
                raise ValueError("log_dir must be set when file_name is provided")
            self.log_dir = Path(self.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            path = self.log_dir / self.file_name

            # Eagerly create the file so it *definitely* exists
            path.touch(exist_ok=True)

            self._attach_file_handler(path)

            # If we still can't see it, fail loudly right here
            if not path.exists():
                raise RuntimeError(f"Failed to create log file at: {path}")

    def _attach_file_handler(self, path: Path) -> None:
        fh = logging.FileHandler(path, mode="a", encoding="utf-8", delay=False)
        fh.setLevel(self.level)
        formatter = TimezoneFormatter(fmt=self.fmt, datefmt=self.datefmt)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)
        self._file_handler = fh

    def flush(self) -> None:
        for h in self._logger.handlers:
            try:
                h.flush()
            except Exception:
                pass

    # def __post_init__(self) -> None:
    #     self._logger = logging.getLogger(self.name)
    #     self._logger.setLevel(self.level)
    #     self._logger.propagate = self.propagate

    #     if self.file_name is not None:
    #         if self.log_dir is None:
    #             raise ValueError("log_dir must be set when file_name is provided")
    #         self.log_dir = Path(self.log_dir)
    #         self.log_dir.mkdir(parents=True, exist_ok=True)
    #         self._attach_file_handler(self.log_dir / self.file_name)

    # def _attach_file_handler(self, path: Path) -> None:
    #     fh = logging.FileHandler(path, encoding="utf-8")
    #     fh.setLevel(self.level)
    #     formatter = TimezoneFormatter(fmt=self.fmt, datefmt=self.datefmt)
    #     fh.setFormatter(formatter)
    #     self._logger.addHandler(fh)
    #     self._file_handler = fh

    # ---- basic API ----
    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._logger.error(msg, *args)

    def exception(self, msg: str, *args: Any) -> None:
        self._logger.exception(msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(msg, *args)

    def close(self) -> None:
        if self._file_handler is not None:
            self._logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None


@dataclass(slots=True)
class TrainerLogger(BaseLogger):

    # ---- EXTENDED API ----
    def log_start(
        self,
        *,
        model_params: str,
        n_layer: int,
        n_embd: int,
        vocab_size: str,
        device: str,
        precision: str,
        num_steps: int,
        accum_steps: int,
        lr: float,
        step_idx: int,
    ) -> None:
        if step_idx == 0:
            self.info("Starting Training")
        else:
            self.info("Resuming Training from step_idx=%d", step_idx)

        self.info(
            "[model] params=%s | n_layer=%d | n_embd=%d | vocab_size=%s", 
            model_params, n_layer, n_embd, vocab_size
        )
        self.info("[device] device=%s | precision=%s", device, precision)
        self.info(
            "[optimization] num_steps=%d | accum_steps=%d | lr=%.3e",
            num_steps, accum_steps, lr,
        )

    def log_train_step(
        self,
        *,
        step: int,
        nll: float | None = None,
        ppl: float | None = None,
        lr: float | None = None,
        tokens_seen_total: int | None = None,
        tokens_per_sec: float | None = None,
        step_ms: float | None = None,
        peak_alloc_gb: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        parts: list[str] = [f"step={step}"]

        if tokens_seen_total is not None:
            parts.append(f"tokens_total={tokens_seen_total}")

        if nll is not None:
            parts.append(f"nll={nll:.4f}")

        if ppl is not None:
            parts.append(f"ppl={ppl:.2f}" if math.isfinite(ppl) else "ppl=inf")

        if lr is not None:
            parts.append(f"lr={lr:.3e}")

        if tokens_per_sec is not None:
            parts.append(f"tok_per_s={tokens_per_sec:.0f}")

        if step_ms is not None:
            parts.append(f"step_ms={step_ms:.1f}")

        if peak_alloc_gb is not None:
            parts.append(f"peak_alloc_gb={peak_alloc_gb:.3f}")

        if extra:
            for k, v in extra.items():
                parts.append(f"{k}={v}")

        self.info("[train] " + " | ".join(parts))

    def log_eval(
        self,
        *,
        step: int,
        nll: float,
        ppl: float,
        tokens: int | None = None,
    ) -> None:
        ppl_str = f"{ppl:.2f}" if math.isfinite(ppl) else "inf"
        parts = [f"step={step}", f"nll={nll:.4f}", f"ppl={ppl_str}"]
        if tokens is not None:
            parts.append(f"tokens={tokens}")
        self.info("[eval] " + " | ".join(parts))

    def log_checkpoint(self, msg) -> None:
        self.info("[checkpoint] " + msg)


@dataclass(slots=True)
class DataLogger(BaseLogger):

    def log_start(self, cfg) -> None:
        
        if cfg.start_sample_idx > 0: 
            self.info("Resuming Data Loading from sample_idx=%d", cfg.start_sample_idx) 
        else:
            self.info("Starting Data Loading") 

        parts = [f"name={cfg.dataset_name}"]
        if cfg.dataset_config is not None:
            parts.append(f"config={cfg.dataset_config}")
        if cfg.train_split is not None:
            parts.append(f"train_split={cfg.train_split}")
        if cfg.eval_split is not None:
            parts.append(f"eval_split={cfg.eval_split}")

        self.info("[dataset info] " + " | ".join(parts))


    def log_batch_info(
        self, 
        *, 
        vocab_size: int, 
        seq_len: int, 
        train_batch_size: int,
        eval_batch_size: int,
        dtype: str,
    ) -> None:
        self.info(
            "[batch info] vocab_size=%d | seq_len=%d | train_batch_size=%d | eval_batch_size=%d | dtype=%s", 
            vocab_size, seq_len, train_batch_size, eval_batch_size, dtype
        )
        
    def log_dataset_loading(self, msg: str) -> None: 
        self.info("[hf dataset loading] %s", msg)

    def log_tokenization(self, msg: str) -> None:
        self.info("[tokenization] %s", msg)

    def log_token_buffer_loading(self, msg: str) -> None:
        self.info("[token buffer loading] %s", msg)
        
    def log_dataloader_info(self, msg: str) -> None:
        self.info("[dataloader] %s", msg)
    
