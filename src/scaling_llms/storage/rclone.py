from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scaling_llms.storage.base import DefaultRegistryStorage, RegistryStorage


def _ensure_rclone_available(rclone_executable: str = "rclone") -> None:
    if shutil.which(rclone_executable) is None:
        raise RuntimeError(f"rclone executable not found: {rclone_executable}")


def _run_rclone(command: list[str], *, cwd: str | Path | None = None) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        raise RuntimeError(f"rclone command failed: {' '.join(command)}\n{details}") from exc


def build_remote_project_path(*, remote_name: str, remote_project_subdir: str) -> str:
    """
    Defines paths for rclone operations.

    Remote paths use the '<remote_name>:path/to/dir' format, where the colon (:)
    acts as the separator between the rclone config alias and the cloud
    directory. Local paths are standard Linux absolute or relative paths
    (e.g., '/workspace/runs') and must not contain a colon.
    """

    normalized_subdir = remote_project_subdir.strip().strip("/")
    if not normalized_subdir:
        raise ValueError("remote_project_subdir must be a non-empty path")
    return f"{remote_name}:{normalized_subdir}"


def sync_local_to_remote(
    local_path: str | Path,
    remote_path: str,
    *,
    mode: str = "copy",
    rclone_executable: str = "rclone",
    extra_args: list[str] | None = None,
) -> None:
    """Sync a local path to a remote via rclone.

    Parameters
    ----------
    local_path:
        Local file or directory path.
    remote_path:
        rclone remote path (e.g. ``gdrive:ml-experiments/scaling-llms-dev/run_registry``).
    mode:
        ``copy`` or ``sync``. ``copy`` is additive; ``sync`` makes destination mirror source.
    """
    if mode not in {"copy", "sync"}:
        raise ValueError("mode must be one of {'copy', 'sync'}")

    src = Path(local_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"local_path does not exist: {src}")

    _ensure_rclone_available(rclone_executable)
    command = [rclone_executable, mode, str(src), remote_path, "--create-empty-src-dirs"]
    if extra_args:
        command.extend(extra_args)
    _run_rclone(command)


def sync_remote_to_local(
    remote_path: str,
    local_path: str | Path,
    *,
    mode: str = "copy",
    rclone_executable: str = "rclone",
    extra_args: list[str] | None = None,
) -> None:
    """Sync a remote path into a local path via rclone."""
    if mode not in {"copy", "sync"}:
        raise ValueError("mode must be one of {'copy', 'sync'}")

    dst = Path(local_path).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    _ensure_rclone_available(rclone_executable)
    command = [rclone_executable, mode, remote_path, str(dst), "--create-empty-src-dirs"]
    if extra_args:
        command.extend(extra_args)
    _run_rclone(command)


@dataclass(frozen=True)
class RCloneDiskConfigs:
    remote_name: str
    remote_project_subdir: str

    @property
    def remote_project_root(self) -> str:
        return build_remote_project_path(
            remote_name=self.remote_name,
            remote_project_subdir=self.remote_project_subdir,
        )


def setup_rclone_storage(config: RCloneDiskConfigs) -> RegistryStorage:
    return DefaultRegistryStorage(
        project_root=config.remote_project_root,
    ).to_registry_storage(create_dirs=False, resolve_project_root=False)


def make_remote_storage(configs: RCloneDiskConfigs | None = None, **overrides) -> RegistryStorage:
    if configs is None:
        config = RCloneDiskConfigs(**overrides)
    else:
        config = configs
    return setup_rclone_storage(config)