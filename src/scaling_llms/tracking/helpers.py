
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal

from scaling_llms.utils.config import BaseJsonConfig


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


def log_as_json(obj: Any, path: str | Path) -> Path:
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


def _get_figure_backend(fig: Any) -> Literal["plotly", "matplotlib"]:
    try:
        import plotly.graph_objects as go

        if isinstance(fig, go.Figure):
            return "plotly"
    except Exception:
        pass

    try:
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        if isinstance(fig, Figure) or isinstance(fig, Axes):
            return "matplotlib"
        if hasattr(fig, "savefig") and hasattr(fig, "gca"):
            return "matplotlib"
    except Exception:
        pass

    raise ValueError(f"Could not determine figure backend; got type {type(fig)}")


def log_as_png(fig: Any, path: str | Path) -> Path:
    backend = _get_figure_backend(fig)
    path = Path(path)
    if backend == "plotly":
        fig.write_image(str(path))
    elif backend == "matplotlib":
        fig.savefig(str(path))
    else:
        raise ValueError(f"Unsupported figure backend: {backend}")
    return path


def log_as_html(fig: Any, path: str | Path) -> Path:
    backend = _get_figure_backend(fig)
    path = Path(path)
    if backend == "plotly":
        fig.write_html(str(path))
        return path
    raise ValueError(f"Unsupported figure backend: {backend}")