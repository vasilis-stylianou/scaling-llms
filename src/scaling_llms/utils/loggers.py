from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def setup_root_logging(
    *,
    level: int = logging.INFO,
    stream=sys.stdout,
    fmt: str = "%(message)s",
    force: bool = True,
) -> None:
    """
    Works reliably in scripts + VSCode + Jupyter/Colab.

    - force=True replaces any existing handlers (needed in notebooks)
    - fmt="%(message)s" removes INFO/level prefixes
    """
    try:
        logging.basicConfig(
            level=level,
            format=fmt,
            handlers=[logging.StreamHandler(stream)],
            force=force,  # Python 3.8+
        )
    except TypeError:
        # Fallback for older Python (no force=)
        root = logging.getLogger()
        root.setLevel(level)
        for h in list(root.handlers):
            root.removeHandler(h)
        h = logging.StreamHandler(stream)
        h.setFormatter(logging.Formatter(fmt))
        root.addHandler(h)


@dataclass(slots=True)
class BaseLogger:
    name: str = "Logger"
    log_dir: Path | None = None
    level: int = logging.INFO
    propagate: bool = True
    file_name: str | None = None  # if set, attach a file handler to log_dir/file_name
    fmt: str = "%(message)s"

    _logger: logging.Logger = dataclasses.field(init=False, repr=False)
    _file_handler: logging.Handler | None = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)
        self._logger.propagate = self.propagate

        if self.file_name is not None:
            if self.log_dir is None:
                raise ValueError("log_dir must be set when file_name is provided")
            self.log_dir = Path(self.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._attach_file_handler(self.log_dir / self.file_name)

    def _attach_file_handler(self, path: Path) -> None:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(self.level)
        fh.setFormatter(logging.Formatter(self.fmt))
        self._logger.addHandler(fh)
        self._file_handler = fh

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
    name: str = "Trainer"
    file_name: str | None = "train.log"

    # ---- EXTENDED API ----
    def log_start(
        self,
        *,
        run_root: Path | None,
        device: str,
        precision: str,
        num_steps: int,
        accum_steps: int,
        lr: float,
    ) -> None:
        self.info("starting training")
        if run_root is not None:
            self.info("run_dir=%s", str(run_root))
        self.info("device=%s precision=%s", device, precision)
        self.info("num_steps=%d accum_steps=%d lr=%.3e", num_steps, accum_steps, lr)

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
        extra: dict[str, Any] | None = None,
    ) -> None:
        parts: list[str] = [f"step={step}"]
        if nll is not None:
            parts.append(f"nll={nll:.4f}")
        if ppl is not None:
            parts.append(f"ppl={ppl:.2f}" if ppl != float("inf") else "ppl=inf")
        if lr is not None:
            parts.append(f"lr={lr:.3e}")
        if tokens_per_sec is not None:
            parts.append(f"tok/s={tokens_per_sec:.0f}")
        if step_ms is not None:
            parts.append(f"step_ms={step_ms:.1f}")
        if tokens_seen_total is not None:
            parts.append(f"tokens_seen={tokens_seen_total}")
        if extra:
            parts.append("extra=" + ", ".join(f"{k}={v}" for k, v in extra.items()))
        self.info(" | ".join(parts))

    def log_eval(
        self,
        *,
        step: int,
        nll: float,
        ppl: float,
        tokens: int | None = None,
    ) -> None:
        ppl_str = "inf" if ppl == float("inf") else f"{ppl:.2f}"
        msg = f"eval | step={step} | nll={nll:.4f} | ppl={ppl_str}"
        if tokens is not None:
            msg += f" | tokens={tokens}"
        self.info(msg)

    def log_checkpoint(self, *, step: int, path: Path, kind: str = "last") -> None:
        self.info("checkpoint | kind=%s | step=%d | path=%s", kind, step, str(path))
