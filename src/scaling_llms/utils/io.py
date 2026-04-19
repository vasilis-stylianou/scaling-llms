import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np

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
    if isinstance(o, np.dtype):
        return {"__type__": "np.dtype", "name": o.name}

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
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

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


def _json_object_hook(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__type__") == "np.dtype":
            return np.dtype(obj["name"])

        return {k: _json_object_hook(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_json_object_hook(v) for v in obj]

    return obj


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return _json_object_hook(data)


def load_module_from_path(
    file_path: str | Path,
    module_name: str | None = None,
) -> ModuleType:

    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Module file not found: {path}")

    # Default module name: <parent_dir>__<file_stem>
    # Example: /a/b/foo/bar.py -> foo__bar
    name = module_name or f"{path.parent.name}__{path.stem}"

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")

    module = importlib.util.module_from_spec(spec)

    # Ensure imports of other files in the same directory
    # resolve correctly during execution
    # (e.g. `from constants import ...`) 
    parent_dir = str(path.parent)
    add_to_sys_path = parent_dir not in sys.path 
    if parent_dir not in sys.path:
        # Only modify sys.path if the parent directory is not already in it
        sys.path.insert(0, parent_dir)

    try:
        spec.loader.exec_module(module)
    finally:
        if add_to_sys_path:
            sys.path.remove(parent_dir)

    return module


def get_local_repo_dir() -> Path:
    """Get the local repository directory by running 'git rev-parse'."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path("."),
            text=True
        ).strip()
        return Path(output)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to get local repository directory. " \
            "Ensure this is run within a git repository."
            ) from e
