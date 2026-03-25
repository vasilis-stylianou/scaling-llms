from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from runpod_orchestrator.specs import PodConnectionInfo

logger = logging.getLogger(__name__)


class SSHClient:
    """
    SSH/SCP wrapper for executing commands and transferring files to remote pods.

    Args:
    -----
    identity_file: str 
        - path to SSH private key for authentication (supports ~ expansion)

    Methods:
    --------
    run_command(conn, command, check=True, timeout_s=None) -> CompletedProcess[str]
        - execute a shell command on the remote pod over SSH and return result
    scp_to_pod(conn, local_path, remote_path, check=True, timeout_s=None) -> CompletedProcess[str]
        - upload a local file to the remote pod using SCP
    probe_connectivity(conn, timeout_s=10) -> bool
        - check if the pod is reachable over SSH by running a simple command
    shell_quote(value) -> str
        - utility method to shell-quote a string for safe remote execution
    """

    def __init__(self, identity_file: str) -> None:
        self.identity_file = str(Path(identity_file).expanduser())

    # -- command helpers --

    def _validate(self, conn: PodConnectionInfo) -> None:
        if not conn.is_ssh_ready:
            raise RuntimeError(f"Pod {conn.pod_id} is not SSH ready")
        if not Path(self.identity_file).exists():
            raise FileNotFoundError(
                f"SSH identity file not found: {self.identity_file}"
            )

    def _base_ssh_args(self, conn: PodConnectionInfo) -> list[str]:
        """
        Construct the base `ssh` command args for the given connection.

        Validates SSH readiness and the presence of the expanded identity file,
        then returns the list of command arguments for `ssh`.
        """
        self._validate(conn)
        return [
            "ssh",
            "-i", self.identity_file,
            "-p", str(conn.ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"root@{conn.public_ip}",
        ]

    # -- public API --

    def run_command(
        self,
        conn: PodConnectionInfo,
        command: str,
        *,
        check: bool = True,
        timeout_s: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute a shell `command` on the remote pod over SSH and return result.

        Builds the SSH invocation, quotes the remote command for `bash -lc`, and
        executes it returning the completed process object.
        """
        ssh_cmd = self._base_ssh_args(conn)
        remote_cmd = f"bash -lc {shlex.quote(command)}"
        return subprocess.run(
            ssh_cmd + [remote_cmd],
            text=True,
            capture_output=True,
            check=check,
            timeout=timeout_s,
        )

    def scp_to_pod(
        self,
        conn: PodConnectionInfo,
        local_path: str | Path,
        remote_path: str,
        *,
        check: bool = True,
        timeout_s: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Copy a local file to the remote pod using `scp` and return result.

        Validates SSH readiness and identity file presence, then runs `scp` to
        transfer the resolved local path to the remote destination.
        """
        
        self._validate(conn)
        resolved_local_path = Path(local_path).expanduser().resolve()
        cmd = [
            "scp",
            "-i", self.identity_file,
            "-P", str(conn.ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            str(resolved_local_path),
            f"root@{conn.public_ip}:{remote_path}",
        ]
        return subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=check,
            timeout=timeout_s,
        )

    def probe_connectivity(
        self,
        conn: PodConnectionInfo,
        timeout_s: int = 10,
    ) -> bool:
        """
        Probe whether SSH connectivity to the pod succeeds within a timeout.

        Performs a lightweight `true` command over SSH and returns True on success.
        """
        try:
            self.run_command(conn, "true", timeout_s=timeout_s)
            return True
        except Exception as exc:
            logger.debug("SSH probe failed for pod %s: %s", conn.pod_id, exc)
            return False
