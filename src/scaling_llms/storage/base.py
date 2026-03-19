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


@dataclass(frozen=True)
class DefaultRegistryStorage:
    """
    Canonical project storage layout builder.

    Given a project root, it materializes a `RegistryStorage` using the
    default registry names and subpaths.
    """

    project_root: str | Path

    run_registry_name: str = "run_registry"
    runs_db_name: str = "runs.db"
    runs_artifacts_subdir: str = "artifacts"

    data_registry_name: str = "data_registry"
    datasets_db_name: str = "datasets.db"
    datasets_artifacts_subdir: str = "tokenized_datasets"

    def to_registry_storage(
        self,
        *,
        create_dirs: bool = True,
        resolve_project_root: bool = True,
    ) -> RegistryStorage:
        project_root = Path(self.project_root).expanduser()
        if resolve_project_root:
            project_root = project_root.resolve()
        run_registry_root = project_root / self.run_registry_name
        data_registry_root = project_root / self.data_registry_name

        if create_dirs:
            project_root.mkdir(parents=True, exist_ok=True)
            run_registry_root.mkdir(parents=True, exist_ok=True)
            data_registry_root.mkdir(parents=True, exist_ok=True)

        return RegistryStorage(
            project_root=project_root,
            run_registry_root=run_registry_root,
            runs_db_path=run_registry_root / self.runs_db_name,
            runs_artifacts_root=run_registry_root / self.runs_artifacts_subdir,
            data_registry_root=data_registry_root,
            datasets_db_path=data_registry_root / self.datasets_db_name,
            datasets_artifacts_root=data_registry_root / self.datasets_artifacts_subdir,
        )