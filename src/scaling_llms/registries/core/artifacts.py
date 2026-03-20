from __future__ import annotations

from pathlib import Path
import secrets

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