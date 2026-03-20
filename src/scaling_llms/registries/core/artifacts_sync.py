from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class ArtifactsSyncHooks:
    def push_local_to_remote(self, relative_artifacts_path: str | Path) -> None:
        raise NotImplementedError

    def pull_remote_to_local(self, relative_artifacts_path: str | Path) -> None:
        raise NotImplementedError


class RCloneArtifactsSyncHooks(ArtifactsSyncHooks):
    def __init__(
        self,
        *,
        local_artifacts_root: str | Path,
        remote_rclone_name: str,
        remote_artifacts_root: str,
        mode: str = "copy",
        rclone_executable: str = "rclone",
        extra_args: list[str] | None = None,
    ):
        remote_name = remote_rclone_name.strip()
        remote_root = remote_artifacts_root.strip().strip("/")

        if not remote_name:
            raise ValueError("remote_rclone_name must be a non-empty rclone remote name")
        if not remote_root:
            raise ValueError("remote_artifacts_root must be a non-empty remote path")
        if mode not in {"copy", "sync"}:
            raise ValueError("mode must be one of {'copy', 'sync'}")

        self.local_artifacts_root = Path(local_artifacts_root).expanduser().resolve()
        self.remote_rclone_name = remote_name
        self.remote_artifacts_root = f"{remote_name}:{remote_root}"
        self.mode = mode
        self.rclone_executable = rclone_executable
        self.extra_args = extra_args

        self._ensure_rclone_available()

    # ---------- internal helpers ----------
    def _ensure_rclone_available(self) -> None:
        if shutil.which(self.rclone_executable) is None:
            raise RuntimeError(f"rclone executable not found: {self.rclone_executable}")

    def _run_rclone(self, command: list[str]) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or str(exc)
            raise RuntimeError(
                f"rclone command failed: {' '.join(command)}\n{details}"
            ) from exc

    def _validate_relative_artifacts_path(self, relative_artifacts_path: str | Path) -> Path:
        rel = Path(relative_artifacts_path)
        if rel.is_absolute():
            raise ValueError(f"relative_artifacts_path must be relative, got absolute path: {rel}")
        if ".." in rel.parts:
            raise ValueError(f"relative_artifacts_path cannot escape root via '..': {rel}")
        return rel

    def _local_path_for(self, relative_artifacts_path: Path) -> Path:
        return (self.local_artifacts_root / relative_artifacts_path).resolve()

    def _remote_path_for(self, relative_artifacts_path: Path) -> str:
        rel = relative_artifacts_path.as_posix()
        return f"{self.remote_artifacts_root}/{rel}" if rel else self.remote_artifacts_root

    def _sync_local_to_remote(self, local_artifacts_path: Path, remote_path: str) -> None:
        src = Path(local_artifacts_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"local_artifacts_path does not exist: {src}")

        command = [
            self.rclone_executable,
            self.mode,
            str(src),
            remote_path,
            "--create-empty-src-dirs",
        ]
        if self.extra_args:
            command.extend(self.extra_args)

        self._run_rclone(command)

    def _sync_remote_to_local(self, remote_path: str, local_artifacts_path: Path) -> None:
        dst = Path(local_artifacts_path).expanduser().resolve()
        dst.mkdir(parents=True, exist_ok=True)

        command = [
            self.rclone_executable,
            self.mode,
            remote_path,
            str(dst),
            "--create-empty-src-dirs",
        ]
        if self.extra_args:
            command.extend(self.extra_args)

        self._run_rclone(command)

    # ---------- public API ----------

    def push_local_to_remote(self, relative_artifacts_path: str | Path) -> None:
        rel = self._validate_relative_artifacts_path(relative_artifacts_path)
        local_artifacts_path = self._local_path_for(rel)
        remote_path = self._remote_path_for(rel)
        self._sync_local_to_remote(local_artifacts_path, remote_path)

    def pull_remote_to_local(self, relative_artifacts_path: str | Path) -> None:
        rel = self._validate_relative_artifacts_path(relative_artifacts_path)
        local_artifacts_path = self._local_path_for(rel)
        remote_path = self._remote_path_for(rel)
        self._sync_remote_to_local(remote_path, local_artifacts_path)