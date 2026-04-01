import subprocess
from pathlib import Path


class RcloneMount:
    def __init__(self, remote_name: str, remote_path: str, mount_point: str | Path = "./artifacts"):
        self._remote_name = remote_name
        self._remote_path = remote_path

        self.remote_path = f"{remote_name}:{remote_path}"
        self.mount_path = Path(mount_point).expanduser().resolve()

    def __repr__(self):
        return f"RcloneMount(remote_path='{self.remote_path}', mount_path='{self.mount_path}')"

    __str__ = __repr__

    # 👉 allows passing object directly where path is expected
    def __fspath__(self):
        return str(self.mount_path)

    # optional: makes Path(self) work
    def __truediv__(self, other):
        return self.mount_path / other

    # ------------------------
    # Core functionality
    # ------------------------

    def mount(self, vfs_cache_mode="full", daemon=True) -> "RcloneMount":
        self.mount_path.mkdir(parents=True, exist_ok=True)

        if self.is_mounted():
            return self
        cmd = [
            "rclone",
            "mount",
            self.remote_path,
            str(self.mount_path),
            "--vfs-cache-mode",
            vfs_cache_mode,
        ]

        if daemon:
            cmd.append("--daemon")

        subprocess.run(cmd, check=True)
        return self

    def is_mounted(self) -> bool:
        result = subprocess.run(
            ["mount"],
            capture_output=True,
            text=True,
        )
        return str(self.mount_path) in result.stdout

    def unmount(self, force=False) -> "RcloneMount":
        if not self.is_mounted():
            return self

        cmd = ["umount"]
        if force:
            cmd.append("-f")

        cmd.append(str(self.mount_path))
        subprocess.run(cmd, check=True)
        return self

    # ------------------------
    # Convenience helpers
    # ------------------------

    def ensure_mounted(self) -> "RcloneMount":
        if not self.is_mounted():
            self.mount()
        return self