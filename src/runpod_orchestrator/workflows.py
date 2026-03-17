from __future__ import annotations

from pathlib import Path

from runpod_orchestrator import lifecycle
from runpod_orchestrator.models import PodConnectionInfo, PodSpec, RetryPolicy, SetupSpec, TrainSpec
from runpod_orchestrator.provision import setup_pod
from runpod_orchestrator.ssh import run_ssh_command, shell_quote


def _build_tmux_training_command(spec: TrainSpec) -> str:
    """Build a shell command that launches the training command inside tmux.

    Produces a single shell string that creates the log dir, starts a detached
    tmux session and appends output to the configured log path.
    """

    log_path = shell_quote(spec.log_path)
    log_dir = shell_quote(str(Path(spec.log_path).parent))
    repo_dir = shell_quote(spec.repo_dir)
    session_name = shell_quote(spec.tmux_session_name)

    train_cmd = f"cd {repo_dir} && {spec.command} 2>&1 | tee -a {log_path}"
    train_cmd_quoted = shell_quote(train_cmd)

    return (
        f"mkdir -p {log_dir} && "
        f"tmux kill-session -t {session_name} >/dev/null 2>&1 || true && "
        f"tmux new-session -d -s {session_name} bash -lc {train_cmd_quoted}"
    )


def train_on_pod(conn: PodConnectionInfo, spec: TrainSpec) -> str:
    """Launch the training command on the pod inside a tmux session.

    Executes the tmux-start command remotely and returns the configured log
    path where training output will be written.
    """

    remote_cmd = _build_tmux_training_command(spec)
    run_ssh_command(conn, remote_cmd, identity_file=spec.expanded_identity_file)
    return spec.log_path


def up_workflow(
    *,
    pod_spec: PodSpec,
    bootstrap_script: Path,
    setup_spec: SetupSpec,
    reuse_if_exists: bool,
    timeout_s: int,
    poll_s: int,
    retry_policy: RetryPolicy,
) -> PodConnectionInfo:
    """Create or reuse a pod, run provisioning, and return its connection info.

    Uses the provided bootstrap script to build docker args, ensures the pod
    exists and SSH is available, then runs the provisioning sequence.
    """

    docker_args = lifecycle.load_bootstrap_as_docker_args(bootstrap_script)
    conn = lifecycle.resolve_or_create_pod(
        pod_spec,
        docker_args,
        reuse_if_exists=reuse_if_exists,
        timeout_s=timeout_s,
        poll_s=poll_s,
        retry_policy=retry_policy,
    )
    setup_pod(conn, setup_spec)
    return conn


def run_workflow(
    *,
    pod_spec: PodSpec,
    bootstrap_script: Path,
    setup_spec: SetupSpec,
    train_spec: TrainSpec,
    reuse_if_exists: bool,
    timeout_s: int,
    poll_s: int,
    retry_policy: RetryPolicy,
    terminate_after_launch: bool,
    terminate_on_failure: bool,
) -> PodConnectionInfo:
    """Run create+setup+train workflow and optionally terminate after launch.

    Coordinates `up_workflow`, remote training launch, and optional
    termination semantics for success or failure cases.
    """
    conn: PodConnectionInfo | None = None
    try:
        conn = up_workflow(
            pod_spec=pod_spec,
            bootstrap_script=bootstrap_script,
            setup_spec=setup_spec,
            reuse_if_exists=reuse_if_exists,
            timeout_s=timeout_s,
            poll_s=poll_s,
            retry_policy=retry_policy,
        )
        train_on_pod(conn, train_spec)
        if terminate_after_launch:
            lifecycle.terminate_pod(conn.pod_id, retry_policy=retry_policy)
        return conn
    except Exception:
        if conn is not None and terminate_on_failure:
            lifecycle.terminate_pod(conn.pod_id, retry_policy=retry_policy)
        raise


def down_workflow(
    *,
    pod_id: str,
    retry_policy: RetryPolicy,
) -> None:
    """Terminate a pod by id using the lifecycle terminate helper.

    This is a small convenience wrapper that delegates termination to the
    lifecycle logic.
    """

    lifecycle.terminate_pod(pod_id, retry_policy=retry_policy)
