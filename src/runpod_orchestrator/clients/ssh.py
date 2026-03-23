# runpod_orchestrator/clients/ssh.py

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path

from runpod_orchestrator.exceptions import SSHError
from runpod_orchestrator.specs import PodConnectionInfo


class SSHClient:
    """
    SSHClient is a wrapper around the SSH command-line tool that provides methods for
    executing commands and transferring files to remote pods. It handles SSH key
    management, connection timeouts, and error handling.
    """
    def __init__(self, identity_file: str, connect_timeout_s: int = 10) -> None:
        self.identity_file = str(Path(identity_file).expanduser())
        self.connect_timeout_s = connect_timeout_s

    def _base_ssh_args(self, conn: PodConnectionInfo) -> list[str]:
        return [
            "ssh",
            "-i",
            self.identity_file,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            f"ConnectTimeout={self.connect_timeout_s}",
            "-p",
            str(conn.ssh_port),
            f"root@{conn.public_ip}",
        ]

    def _base_scp_args(self, conn: PodConnectionInfo) -> list[str]:
        return [
            "scp",
            "-i",
            self.identity_file,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            f"ConnectTimeout={self.connect_timeout_s}",
            "-P",
            str(conn.ssh_port),
        ]

    def run(self, conn: PodConnectionInfo, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [*self._base_ssh_args(conn), command],
            text=True,
            capture_output=True,
            check=False,
        )
        if check and proc.returncode != 0:
            raise SSHError(
                f"Remote command failed with code {proc.returncode}\n"
                f"Command: {command}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return proc

    def upload(self, conn: PodConnectionInfo, local_path: str | Path, remote_path: str) -> None:
        src = str(Path(local_path).expanduser().resolve())
        if not Path(src).exists():
            raise SSHError(f"Local file does not exist: {src}")
        proc = subprocess.run(
            [*self._base_scp_args(conn), src, f"root@{conn.public_ip}:{remote_path}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SSHError(
                f"SCP upload failed with code {proc.returncode}\n"
                f"Source: {src}\n"
                f"Destination: {remote_path}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

    def wait_until_ready(
        self,
        conn: PodConnectionInfo,
        *,
        timeout_s: int,
        poll_s: int,
    ) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            proc = self.run(conn, "echo ssh-ready", check=False)
            if proc.returncode == 0 and "ssh-ready" in proc.stdout:
                return
            time.sleep(poll_s)
        raise SSHError(f"SSH did not become ready within {timeout_s}s for pod {conn.pod_id}")

    @staticmethod
    def require_executable(name: str) -> None:
        if shutil.which(name) is None:
            raise SSHError(f"Required executable not found: {name}")

    @staticmethod
    def quote(value: str) -> str:
        return shlex.quote(value)