from __future__ import annotations

from pathlib import Path

from scaling_llms.utils.loggers import BaseLogger

class RunArtifacts:
    """
    Single source of truth for the run directory layout + path helpers.

    Owns:
      - root path
      - well-known subdirs
      - creation of directory structure
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.logger = BaseLogger(name="RunArtifacts")

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # ---- Define subdirs ----
    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def tb(self) -> Path:
        return self.root / "tb"


    # ---- API ----
    def get_all_paths(self) -> dict[str, Path]:
        """Returns a dictionary mapping subdir names to their Path values."""
        subdir_names = [
            attr_name for attr_name in dir(type(self)) 
            if isinstance(getattr(type(self), attr_name), property)
        ]
        # getattr(self, name) actually executes the property and gets the Path
        return {name: getattr(self, name) for name in subdir_names}
    
    def list_subdir_names(self) -> list[str]:
        """Returns a list of all subdir names."""
        return list(self.get_all_paths().keys())
    
    def list_subdir_paths(self) -> list[Path]:
        """Returns a list of all subdir Paths."""
        return list(self.get_all_paths().values())
    
    def ensure_dirs(self, *, exist_ok: bool = True) -> None:
        """
        Create root + all standard subdirs.
        If exist_ok=False, raises FileExistsError if any already exist.
        """
        self.root.mkdir(parents=True, exist_ok=exist_ok)
        
        # Iterate cleanly over the properties themselves
        for dir_path in self.list_subdir_paths():
            dir_path.mkdir(parents=True, exist_ok=exist_ok)

    def wipe(self) -> None:
        """Delete all contents of the run directory, but keep the directory itself."""
        if self.root.exists() and self.root.is_dir():
            for item in self.root.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for subitem in item.rglob("*"):
                        if subitem.is_file():
                            subitem.unlink()
                        elif subitem.is_dir():
                            subitem.rmdir()
                    item.rmdir()
        else:
            self.logger.warning(
                "Run directory %s does not exist or is not a directory; cannot wipe.", 
                self.root
            )

    # ---- paths ----
    def metadata_path(
        self,
        filename: str,
        subdir_name: str | None = None,
        ensure_dir: bool = False,
    ) -> Path:
        base = self.metadata
        if subdir_name is not None:
            base = base / subdir_name
            if ensure_dir:
                base.mkdir(parents=True, exist_ok=True)

        return base / filename

    def checkpoint_path(self, filename: str) -> Path:
        return self.checkpoints / filename
    
    def metric_path(self, category: str) -> Path:
        return self.metrics / f"{category}.jsonl"
