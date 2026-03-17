from __future__ import annotations

import logging
from pathlib import Path

from runpod_orchestrator.models import PodConnectionInfo, SetupSpec
from runpod_orchestrator.ssh import run_ssh_command, scp_to_pod, shell_quote

logger = logging.getLogger(__name__)


def copy_rclone_config(conn: PodConnectionInfo, spec: SetupSpec) -> None:
    """Copy a local rclone config into the remote pod when configured.

    Creates the remote parent directory and uses `scp` to transfer the file.
    """

    local_rclone = spec.expanded_rclone_local
    if local_rclone is None:
        return

    if not local_rclone.exists():
        raise FileNotFoundError(f"rclone config not found: {local_rclone}")

    remote_parent = str(Path(spec.rclone_config_remote).parent)
    run_ssh_command(
        conn,
        f"mkdir -p {shell_quote(remote_parent)}",
        identity_file=spec.expanded_identity_file,
    )
    logger.info("Copying rclone config to %s", spec.rclone_config_remote)
    scp_to_pod(
        conn,
        local_path=local_rclone,
        remote_path=spec.rclone_config_remote,
        identity_file=spec.expanded_identity_file,
    )


def clone_or_update_repo(conn: PodConnectionInfo, spec: SetupSpec) -> None:
    """Ensure the repository is present on the remote pod, cloning or updating.

    Either clones the specified repo (and branch) or performs a fetch/pull to
    keep an existing checkout up to date.
    """

    repo_url_value = spec.repo_url.strip()
    if not repo_url_value:
        raise ValueError("SetupSpec.repo_url must be provided")

    repo_dir = shell_quote(spec.repo_dir)
    repo_url = shell_quote(repo_url_value)
    repo_branch_value = spec.repo_branch.strip() if spec.repo_branch else None
    branch = shell_quote(repo_branch_value) if repo_branch_value else None

    if branch:
        base_cmd = (
            f"mkdir -p {shell_quote(str(Path(spec.repo_dir).parent))} && "
            f"if [ -d {repo_dir}/.git ]; then "
            f"git -C {repo_dir} fetch origin {branch} --prune && "
            f"git -C {repo_dir} checkout {branch} && "
            f"git -C {repo_dir} reset --hard origin/{branch}; "
            f"else git clone --branch {branch} --single-branch {repo_url} {repo_dir}; fi"
        )
    else:
        base_cmd = (
            f"mkdir -p {shell_quote(str(Path(spec.repo_dir).parent))} && "
            f"if [ -d {repo_dir}/.git ]; then "
            f"git -C {repo_dir} fetch --all --prune && "
            "current_branch=$(git -C "
            f"{repo_dir} rev-parse --abbrev-ref HEAD) && "
            "if [ \"$current_branch\" = \"HEAD\" ]; then "
            "default_branch=$(git -C "
            f"{repo_dir} remote show origin | sed -n 's/.*HEAD branch: //p') && "
            f"git -C {repo_dir} checkout \"$default_branch\"; "
            "fi && "
            f"git -C {repo_dir} pull --ff-only; "
            f"else git clone {repo_url} {repo_dir}; fi"
        )

    logger.info("Syncing repository at %s", spec.repo_dir)
    run_ssh_command(conn, base_cmd, identity_file=spec.expanded_identity_file)


def poetry_install_remote(conn: PodConnectionInfo, spec: SetupSpec) -> None:
    """Run `poetry install` in the checked-out repository on the pod.

    Invokes `poetry install` remotely with optional extra args supplied in the
    setup spec.
    """

    repo_dir = shell_quote(spec.repo_dir)
    extra_args = " ".join(shell_quote(arg) for arg in spec.poetry_install_args)
    cmd = f"cd {repo_dir} && poetry install"
    if extra_args:
        cmd = f"{cmd} {extra_args}"

    run_ssh_command(conn, cmd, identity_file=spec.expanded_identity_file)


def create_jupyter_kernel_remote(conn: PodConnectionInfo, spec: SetupSpec) -> None:
    """Install a Jupyter kernel in the remote environment when requested.

    When `create_jupyter_kernel` is True, installs an ipykernel entry for the
    project inside the pod's environment.
    """

    if not spec.create_jupyter_kernel:
        return

    repo_dir = shell_quote(spec.repo_dir)
    kernel_name = shell_quote(spec.kernel_name)
    kernel_display_name = shell_quote(spec.kernel_display_name)
    cmd = (
        f"cd {repo_dir} && "
        "poetry run python -m ipykernel install --user "
        f"--name {kernel_name} --display-name {kernel_display_name}"
    )
    run_ssh_command(conn, cmd, identity_file=spec.expanded_identity_file)


def setup_pod(conn: PodConnectionInfo, spec: SetupSpec) -> None:
    """Perform the full provisioning sequence on the remote pod.

    Executes rclone config copy, repository sync, dependency install, and
    optional kernel installation in order.
    """

    copy_rclone_config(conn, spec)
    clone_or_update_repo(conn, spec)
    poetry_install_remote(conn, spec)
    create_jupyter_kernel_remote(conn, spec)
