from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class ArtifactsSyncHooks:
    def push_local_to_remote(self, relative_artifacts_path: str | Path) -> None:
        raise NotImplementedError

    def pull_remote_to_local(self, relative_artifacts_path: str | Path) -> None:
        raise NotImplementedError

    def pull_remote_file_to_local(self, relative_file_path: str | Path) -> None:
        raise NotImplementedError


class RCloneArtifactsSyncHooks(ArtifactsSyncHooks):
    _DEFAULT_COMMON_ARGS = [
        "--fast-list",
        "--checkers", "32",
        "--transfers", "16",
    ]
    _DEFAULT_PUSH_ARGS = [
        "--s3-upload-concurrency", "16",
        "--create-empty-src-dirs",
    ]
    _DEFAULT_PULL_ARGS = [
        "--create-empty-src-dirs",
    ]
    
    def __init__(
        self,
        *,
        local_artifacts_root: str | Path,
        remote_rclone_name: str,
        remote_artifacts_root: str,
        push_mode: str = "copy",
        pull_mode: str = "sync",
        rclone_executable: str = "rclone",
        global_args: list[str] | None = None,
        push_args: list[str] | None = None,
        pull_args: list[str] | None = None,
    ):
        remote_name = remote_rclone_name.strip()
        remote_root = remote_artifacts_root.strip().strip("/")

        if not remote_name:
            raise ValueError("remote_rclone_name must be a non-empty rclone remote name")
        if not remote_root:
            raise ValueError("remote_artifacts_root must be a non-empty remote path")
        if push_mode not in {"copy", "sync"}:
            raise ValueError("push_mode must be one of {'copy', 'sync'}")
        if pull_mode not in {"copy", "sync"}:
            raise ValueError("pull_mode must be one of {'copy', 'sync'}")

        self.local_artifacts_root = Path(local_artifacts_root).expanduser().resolve()
        self.remote_rclone_name = remote_name
        self.remote_artifacts_root = f"{remote_name}:{remote_root}"
        self.push_mode = push_mode
        self.pull_mode = pull_mode
        self.rclone_executable = rclone_executable

        self.global_args = list(global_args or [])
        self.push_args = list(push_args or [])
        self.pull_args = list(pull_args or [])

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

    def _build_command(
        self,
        *,
        mode: str,
        src: str,
        dst: str,
        op_args: list[str] | None = None,
    ) -> list[str]:
        command = [self.rclone_executable, mode, src, dst]
        command.extend(self._DEFAULT_COMMON_ARGS)
        command.extend(self.global_args)
        if op_args:
            command.extend(op_args)
        return command

    def _remote_exists(self, remote_path: str) -> bool:
        command = [
            self.rclone_executable,
            "lsjson",
            remote_path,
            "--max-depth",
            "1",
            *self._DEFAULT_COMMON_ARGS,
            *self.global_args,
        ]
        try:
            subprocess.run(command, check=True, text=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as exc:
            details = ((exc.stderr or "").strip() or (exc.stdout or "").strip()).lower()
            if (
                "directory not found" in details
                or "object not found" in details
                or "not found" in details
            ):
                return False
            raise RuntimeError(
                f"rclone existence check failed: {' '.join(command)}\n{details}"
            ) from exc

    def _sync_local_to_remote(self, local_artifacts_path: Path, remote_path: str) -> None:
        src = Path(local_artifacts_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"local_artifacts_path does not exist: {src}")

        command = self._build_command(
            mode=self.push_mode,
            src=str(src),
            dst=remote_path,
            op_args=[*self._DEFAULT_PUSH_ARGS, *self.push_args],
        )
        self._run_rclone(command)

    def _sync_remote_to_local(self, remote_path: str, local_artifacts_path: Path) -> None:
        if not self._remote_exists(remote_path):
            return

        dst = Path(local_artifacts_path).expanduser().resolve()
        dst.mkdir(parents=True, exist_ok=True)

        command = self._build_command(
            mode=self.pull_mode,
            src=remote_path,
            dst=str(dst),
            op_args=[*self._DEFAULT_PULL_ARGS, *self.pull_args],
        )
        self._run_rclone(command)

    def _copy_remote_file_to_local(self, remote_file_path: str, local_file_path: Path) -> None:
        if not self._remote_exists(remote_file_path):
            raise FileNotFoundError(f"Remote file does not exist: {remote_file_path}")

        local_file_path = Path(local_file_path).expanduser().resolve()
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        command = self._build_command(
            mode="copyto",
            src=remote_file_path,
            dst=str(local_file_path),
        )
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

    def pull_remote_file_to_local(self, relative_file_path: str | Path) -> None:
        rel = self._validate_relative_artifacts_path(relative_file_path)
        local_file_path = self._local_path_for(rel)
        remote_file_path = self._remote_path_for(rel)
        self._copy_remote_file_to_local(remote_file_path, local_file_path)


def make_sync_hooks(
    *,
    local_artifacts_root: str | Path,
    sync_hooks_type: str | None,
    sync_hooks_args: dict | None,
) -> ArtifactsSyncHooks | None:
    if sync_hooks_type is None:
        return None
    if sync_hooks_type == "rclone":
        if sync_hooks_args is None:
            raise ValueError("sync_hooks_args must be provided if sync_hooks_type is specified")
        return RCloneArtifactsSyncHooks(
            local_artifacts_root=local_artifacts_root,
            **sync_hooks_args,
        )
    raise ValueError(f"Unsupported sync_hooks_type: {sync_hooks_type}")