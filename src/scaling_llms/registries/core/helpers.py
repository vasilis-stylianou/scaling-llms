from datetime import datetime
from pathlib import Path
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


def get_local_iso_timestamp() -> str:
    return datetime.now(ZoneInfo(LOCAL_TIMEZONE)).isoformat()