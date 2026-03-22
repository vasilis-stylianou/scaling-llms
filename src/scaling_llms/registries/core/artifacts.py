from __future__ import annotations

from pathlib import Path
import secrets
import shutil


class ArtifactsDir:
    """
    Base class for local artifact directories.
    """
    root: Path

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def __fspath__(self) -> str:
        return str(self.root)

    def __str__(self) -> str:
        return str(self.root)
    
    def exists(self) -> bool:
        return self.root.exists() and self.root.is_dir()
    
    def ensure_dirs(self, *, exist_ok: bool = True) -> None:
        """
        If exist_ok=False, raises FileExistsError if any already exist.
        """
        # Create all standard subdirs defined as properties on this class
        for attr_name in dir(type(self)):
            attr = getattr(type(self), attr_name)
            if isinstance(attr, property):
                dir_path = getattr(self, attr_name)
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
            raise FileNotFoundError(
                f"Run directory {self.root} does not exist or is not a directory; cannot wipe."
            )


class Artifacts:
    """
    TODO: Methods are allowed to use path-like objects but not Identity objects
    """
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def get_absolute_path(self, artifacts_path: str | Path) -> Path:
        path = Path(artifacts_path).expanduser()

        if path.is_absolute():
            abs_path = path.resolve()
            try:
                abs_path.relative_to(self.root)
            except ValueError:
                raise ValueError(
                    f"absolute artifacts_path {abs_path} is not within artifacts root {self.root}"
                )
            return abs_path

        rel = path
        if ".." in rel.parts:
            raise ValueError(f"artifacts_path cannot escape root via '..': {rel}")
        return (self.root / rel).resolve()
    
    def get_relative_path(self, absolute_path: str | Path) -> Path:
        abs_path = Path(absolute_path).expanduser().resolve()
        try:
            rel_path = abs_path.relative_to(self.root)
            return rel_path
        except ValueError:
            raise ValueError(f"absolute_path {abs_path} is not within artifacts root {self.root}")

    def ensure_root_dir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def make_unique_dir(
        self,
        *,
        prefix: str = "",
        parent_dir: Path,
        nbytes: int = 8,
        max_attempts: int = 64,
        create_dir: bool = False,
    ) -> Path:
        parent_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(max_attempts):
            key = secrets.token_hex(nbytes)
            directory = parent_dir / f"{prefix}{key}"
            if not directory.exists():
                if create_dir:
                    directory.mkdir(parents=False, exist_ok=False)
                return directory

        raise RuntimeError(
            f"Failed to allocate unique directory under {parent_dir} with prefix '{prefix}' "
            f"after {max_attempts} attempts"
        )