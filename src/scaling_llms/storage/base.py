from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RegistryStorage:
    project_root: Path

    run_registry_root: Path
    runs_db_path: Path
    runs_artifacts_root: Path

    data_registry_root: Path
    datasets_db_path: Path
    datasets_artifacts_root: Path

    def __post_init__(self):
        # Validate that all paths are within the project root and have correct parent-child relationships
        child_parent_pairs = [
            ("run_registry_root", "project_root"),
            ("data_registry_root", "project_root"),
            ("runs_db_path", "run_registry_root"),
            ("runs_artifacts_root", "run_registry_root"),
            ("datasets_db_path", "data_registry_root"),
            ("datasets_artifacts_root", "data_registry_root"),
        ]

        for child_name, parent_name in child_parent_pairs:
            child = getattr(self, child_name)
            parent = getattr(self, parent_name)
            if child.parent != parent:
                raise ValueError(
                    f"{child_name} parent must be {parent_name}: {child.parent} != {parent}"
                )