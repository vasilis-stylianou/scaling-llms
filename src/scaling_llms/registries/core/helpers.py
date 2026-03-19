from datetime import datetime
from pathlib import Path
import secrets
import subprocess
from zoneinfo import ZoneInfo
from scaling_llms.constants import LOCAL_TIMEZONE


def get_next_id(prefix: str, parent_dir: Path) -> int:
    existing_ids: list[int] = []
    if parent_dir.exists():
        for p in parent_dir.iterdir():
            if p.is_dir() and p.name.startswith(prefix):
                suffix = p.name[len(prefix):]
                if suffix.isdigit():
                    existing_ids.append(int(suffix))
    return max(existing_ids, default=0) + 1


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


def get_local_iso_timestamp() -> str:
    return datetime.now(ZoneInfo(LOCAL_TIMEZONE)).isoformat()


def get_current_git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None