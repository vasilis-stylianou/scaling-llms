from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from runpod_orchestrator.models import PodConnectionInfo

logger = logging.getLogger(__name__)


def _ssh_base_args(conn: PodConnectionInfo, identity_file: str) -> list[str]:
    """Construct the base `ssh` command args for the given connection.

    Validates SSH readiness and the presence of the expanded identity file,
    then returns the list of command arguments for `ssh`.
    """

    if not conn.is_ssh_ready:
        raise RuntimeError(f"Pod {conn.pod_id} is not SSH ready")

    expanded_identity = str(Path(identity_file).expanduser())
    if not Path(expanded_identity).exists():
        raise FileNotFoundError(f"SSH identity file not found: {expanded_identity}")

    return [
        "ssh",
        "-i",
        expanded_identity,
        "-p",
        str(conn.ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{conn.public_ip}",
    ]


def run_ssh_command(
    conn: PodConnectionInfo,
    command: str,
    *,
    identity_file: str,
    check: bool = True,
    timeout_s: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute a shell `command` on the remote pod over SSH and return result.

    Builds the SSH invocation, quotes the remote command for `bash -lc`, and
    executes it returning the completed process object.
    """

    ssh_cmd = _ssh_base_args(conn, identity_file)
    remote_cmd = f"bash -lc {shlex.quote(command)}"

    result = subprocess.run(
        ssh_cmd + [remote_cmd],
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout_s,
    )
    return result


def scp_to_pod(
    conn: PodConnectionInfo,
    local_path: Path,
    remote_path: str,
    *,
    identity_file: str,
    check: bool = True,
    timeout_s: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Copy a local file to the remote pod using `scp` and return result.

    Validates SSH readiness and identity file presence, then runs `scp` to
    transfer the resolved local path to the remote destination.
    """

    if not conn.is_ssh_ready:
        raise RuntimeError(f"Pod {conn.pod_id} is not SSH ready")

    expanded_identity = str(Path(identity_file).expanduser())
    if not Path(expanded_identity).exists():
        raise FileNotFoundError(f"SSH identity file not found: {expanded_identity}")

    local_path = Path(local_path).expanduser().resolve()

    cmd = [
        "scp",
        "-i",
        expanded_identity,
        "-P",
        str(conn.ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        str(local_path),
        f"root@{conn.public_ip}:{remote_path}",
    ]

    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout_s,
    )


def shell_quote(value: str) -> str:
    """Return a shell-quoted version of the given string.

    Delegates to `shlex.quote` for safe shell escaping.
    """

    return shlex.quote(value)


def probe_ssh_connectivity(
    conn: PodConnectionInfo,
    *,
    identity_file: str,
    timeout_s: int = 10,
) -> bool:
    """Probe whether SSH connectivity to the pod succeeds within a timeout.

    Performs a lightweight `true` command over SSH and returns True on success.
    """

    try:
        run_ssh_command(
            conn,
            "true",
            identity_file=identity_file,
            timeout_s=timeout_s,
        )
        return True
    except Exception as exc:
        logger.debug("SSH probe failed for pod %s: %s", conn.pod_id, exc)
        return False
