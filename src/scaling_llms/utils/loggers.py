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


def setup_console_logging(
    *,
    level: int = logging.INFO,
    stream=sys.stdout,
    fmt: str = "%(asctime)s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = True,
) -> None:
    """
    Works reliably in scripts + VSCode + Jupyter/Colab.

    Root logger = console only.
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
        root = logging.getLogger()
        root.setLevel(level)
        for h in list(root.handlers):
            root.removeHandler(h)
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Silence noisy third-party libraries (AFTER basicConfig)
    for lib in (
        "httpx",
        "httpcore",
        "urllib3",
        "datasets",
        "huggingface_hub",
        "huggingface_hub.utils._http",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)

@dataclass(slots=True)
class BaseLogger:
    """
    Notebook-safe logger with:
      - optional file handler (always created eagerly)
      - optional console handler
      - separate handler levels (console_level / file_level)
      - runtime level adjustment
    """
    name: str = "Logger"
    log_dir: Path | None = None

    # Logger threshold (must be <= min(handler levels) to allow messages through)
    level: int = logging.DEBUG

    # If True, bubble to root logger (usually False for per-run loggers)
    propagate: bool = False

    # File logging
    file_name: str | None = None
    file_level: int = logging.DEBUG  # capture everything by default

    # Console logging (attach a StreamHandler to THIS logger)
    console: bool = False
    console_level: int = logging.INFO

    fmt: str = "%(asctime)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    _logger: logging.Logger = dataclasses.field(init=False, repr=False)
    _file_handler: logging.Handler | None = dataclasses.field(default=None, init=False, repr=False)
    _console_handler: logging.Handler | None = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)
        self._logger.propagate = self.propagate

        # Notebook-safe: remove existing handlers (same logger name persists across cell re-runs)
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        formatter = TimezoneFormatter(fmt=self.fmt, datefmt=self.datefmt)

        # ---- File handler ----
        if self.file_name is not None:
            if self.log_dir is None:
                raise ValueError("log_dir must be set when file_name is provided")
            self.log_dir = Path(self.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            path = self.log_dir / self.file_name
            path.touch(exist_ok=True)  # eager create (Drive/Colab friendly)

            fh = logging.FileHandler(path, mode="a", encoding="utf-8", delay=False)
            fh.setLevel(self.file_level)
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)
            self._file_handler = fh

            if not path.exists():
                raise RuntimeError(f"Failed to create log file at: {path}")

        # ---- Console handler ----
        if self.console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(self.console_level)
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)
            self._console_handler = ch

    # ---- configuration helpers ----
    def set_level(self, level: int) -> None:
        self.level = level
        self._logger.setLevel(level)

    def set_console_level(self, level: int) -> None:
        self.console_level = level
        if self._console_handler is not None:
            self._console_handler.setLevel(level)

    def set_file_level(self, level: int) -> None:
        self.file_level = level
        if self._file_handler is not None:
            self._file_handler.setLevel(level)

    def flush(self) -> None:
        for h in self._logger.handlers:
            try:
                h.flush()
            except Exception:
                pass

    # ---- basic API ----
    def debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(msg, *args)

    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._logger.error(msg, *args)

    def exception(self, msg: str, *args: Any) -> None:
        self._logger.exception(msg, *args)

    def close(self) -> None:
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        self._file_handler = None
        self._console_handler = None


@dataclass(slots=True)
class TrainerLogger(BaseLogger):
    """
    Logging policy for scaling-LLM experiments:

    - INFO: high-level lifecycle / milestones (start/resume, config summary, eval, checkpoint)
    - DEBUG: per-step progress + heavy diagnostics (perf, memory, grad norms, ratios)
    - WARNING: anomalies that may recover (non-finite loss, skipped step, grad overflow)
    - ERROR/EXCEPTION: run-ending failures
    """

    # ---- EXTENDED API ----
    def log_start(
        self,
        *,
        model_params: str,
        n_layer: int,
        n_embd: int,
        vocab_size: str,
        device: str,
        device_name: str,
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
        self.info(
            "[device] device=%s | device_name=%s | precision=%s", 
            device, device_name, precision
        )
        self.info(
            "[optimization] num_steps=%d | accum_steps=%d | lr=%.3e",
            num_steps, accum_steps, lr,
        )

    # High-frequency per-step logs should be DEBUG by default.
    def log_train_step(
        self,
        *,
        step: int,
        loss_sum: float | None = None,
        tokens: int | None = None,
        nll: float | None = None,
        ppl: float | None = None,
        lr: float | None = None,
        tokens_seen_total: int | None = None,
        tokens_per_sec: float | None = None,
        step_ms: float | None = None,
        peak_alloc_gb: float | None = None,
        extra: dict[str, Any] | None = None,
        level: int = logging.DEBUG,  # allow caller to override to INFO every N steps
    ) -> None:
        parts: list[str] = [f"step={step}"]

        if tokens_seen_total is not None:
            parts.append(f"tokens_total={tokens_seen_total:,}")

        if loss_sum is not None:
            parts.append(f"loss_sum={loss_sum:.4f}")

        if tokens is not None:
            parts.append(f"tokens={tokens:,}")
            
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

        msg = "[train] " + " | ".join(parts)
        self._logger.log(level, msg)

    def log_anomaly(self, msg: str, *args: Any) -> None:
        self.warning("[anomaly] " + msg, *args)

    def log_eval_step(
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
            parts.append(f"tokens={tokens:,}")
        self.info("[eval] " + " | ".join(parts))

    def log_checkpoint(self, msg: str) -> None:
        self.info("[checkpoint] " + msg)


@dataclass(slots=True)
class DataLogger(BaseLogger):
    """
    Data pipeline logging policy:

    - INFO: dataset identity, splits, cache hits/misses, token buffer paths, final counts
    - DEBUG: detailed per-shard/per-file progress, throughput, memmap specifics
    - WARNING: fallback paths, partial cache, mismatched tokenizer/vocab metadata
    """

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
