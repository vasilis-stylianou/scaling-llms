from __future__ import annotations
from dataclasses import dataclass

RUN_IDENTITY_COLS = ("experiment_name", "run_name")


@dataclass(frozen=True)
class RunIdentity:
    experiment_name: str
    run_name: str

    def as_kwargs(self) -> dict[str, str]:
        return {"experiment_name": self.experiment_name, "run_name": self.run_name}

    def __str__(self) -> str:
        return f"({self.experiment_name}, {self.run_name})"
