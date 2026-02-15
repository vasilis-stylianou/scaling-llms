from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any


class BaseJsonConfig:
    def to_json(self) -> dict[str, Any]:
        return self._to_json(self)

    @classmethod
    def from_json(cls, path: str | Path):
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        data = cls._postprocess_loaded_data(data)
        return cls(**data)
    
    @classmethod
    def _to_json(cls, obj: Any) -> Any:
        if isinstance(obj, BaseJsonConfig):
            if is_dataclass(obj):
                data = {}
                for f in fields(obj):
                    if f.init:  # only serialize init fields
                        data[f.name] = getattr(obj, f.name)
            else:
                data = obj.__dict__

            return cls._to_json(data)

        if is_dataclass(obj) and not isinstance(obj, type):
            return cls._to_json(asdict(obj))

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, dict):
            return {k: cls._to_json(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [cls._to_json(v) for v in obj]

        return obj

    @classmethod
    def _postprocess_loaded_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        return data

    
