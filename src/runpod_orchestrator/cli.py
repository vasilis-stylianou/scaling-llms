from __future__ import annotations

import argparse
from pathlib import Path

from runpod_orchestrator import lifecycle, workflows
from runpod_orchestrator.models import RetryPolicy, SetupSpec, TrainSpec
from runpod_orchestrator.provision import setup_pod


def _add_retry_args(parser: argparse.ArgumentParser) -> None:
    """Add retry policy related CLI arguments to the given parser.

    Injects `--max-attempts` and `--retry-base-s` options for retry tuning.
    """

    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-base-s", type=float, default=2.0)


def _add_wait_args(parser: argparse.ArgumentParser) -> None:
    """Add wait-related CLI arguments for timeout and poll interval.

    Adds `--timeout-s` and `--poll-s` options to control waiting behavior.
    """

    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--poll-s", type=int, default=5)


def _add_create_args(parser: argparse.ArgumentParser) -> None:
    """Add pod creation related CLI arguments to a parser.

    Adds flags for pod spec path, bootstrap script, and reuse semantics.
    """

    parser.add_argument("--pod-spec", type=Path, required=True, help="Path to pod_spec.yaml")
    parser.add_argument(
        "--bootstrap-script",
        type=Path,
        required=True,
        help="Path to bootstrap.sh; injected into docker_args",
    )
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-if-exists",
        dest="reuse_if_exists",
        action="store_true",
        help="Reuse pod with same name if it exists",
    )
    reuse_group.add_argument(
        "--no-reuse",
        dest="reuse_if_exists",
        action="store_false",
        help="Force fresh pod creation",
    )
    parser.set_defaults(reuse_if_exists=True)


def _add_setup_args(parser: argparse.ArgumentParser, *, require_repo_url: bool) -> None:
    """Add provisioning/setup related CLI arguments to a parser.

    Adds identity, repo, rclone and kernel related options for setup operations.
    """

    parser.add_argument("--identity-file", default="~/.ssh/runpod_key")
    parser.add_argument("--repo-url", required=require_repo_url, help="Git repository URL")
    parser.add_argument("--repo-dir", default="/workspace/repos/scaling-llms")
    parser.add_argument("--repo-branch", default=None)
    parser.add_argument("--rclone-config-local", default=None)
    parser.add_argument("--rclone-config-remote", default="/root/.config/rclone/rclone.conf")
    parser.add_argument("--create-jupyter-kernel", action="store_true")
    parser.add_argument("--kernel-name", default="scaling-llms")
    parser.add_argument("--kernel-display-name", default="Python (scaling-llms)")
    parser.add_argument(
        "--poetry-install-args",
        nargs="*",
        default=[],
        help="Extra args passed to poetry install, e.g. --poetry-install-args --with dev",
    )


def _add_train_args(
    parser: argparse.ArgumentParser,
    *,
    include_identity_file: bool = True,
    include_repo_dir: bool = True,
) -> None:
    """Add training-related CLI arguments such as remote command and tmux.

    Adds `--cmd`, session and logging related options used for remote training.
    """

    parser.add_argument("--cmd", required=True, help="Training command to execute remotely")
    if include_identity_file:
        parser.add_argument("--identity-file", default="~/.ssh/runpod_key")
    if include_repo_dir:
        parser.add_argument("--repo-dir", default="/workspace/repos/scaling-llms")
    parser.add_argument("--tmux-session", default="train")
    parser.add_argument("--log-path", default="/workspace/runs/train.log")


def _identity_file_for_command(args: argparse.Namespace) -> str:
    """Return the identity file path selected by CLI args.

    Small utility that centralizes selection of the identity file option.
    """

    return args.identity_file


def _retry_policy_from_args(args: argparse.Namespace) -> RetryPolicy:
    """Build a `RetryPolicy` object from parsed CLI arguments.

    Extracts retry-related flags and returns the configured dataclass.
    """

    return RetryPolicy(max_attempts=args.max_attempts, retry_base_s=args.retry_base_s)


def _setup_spec_from_args(args: argparse.Namespace) -> SetupSpec:
    """Construct a `SetupSpec` from CLI arguments for provisioning operations.

    Converts CLI flags into a `SetupSpec` used by provisioning helpers.
    """

    return SetupSpec(
        identity_file=args.identity_file,
        repo_url=args.repo_url,
        repo_dir=args.repo_dir,
        repo_branch=args.repo_branch,
        rclone_config_local=args.rclone_config_local,
        rclone_config_remote=args.rclone_config_remote,
        create_jupyter_kernel=args.create_jupyter_kernel,
        kernel_name=args.kernel_name,
        kernel_display_name=args.kernel_display_name,
        poetry_install_args=args.poetry_install_args,
    )


def _train_spec_from_args(args: argparse.Namespace) -> TrainSpec:
    """Construct a `TrainSpec` from CLI arguments for launching training.

    Converts relevant CLI flags into the training specification dataclass.
    """

    return TrainSpec(
        command=args.cmd,
        identity_file=args.identity_file,
        repo_dir=args.repo_dir,
        tmux_session_name=args.tmux_session,
        log_path=args.log_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI parser with all subcommands.

    Registers subcommands for create/stop/terminate/setup/train/up/run and
    wires the appropriate argument groups for each.
    """

    parser = argparse.ArgumentParser(description="RunPod utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create pod from spec and wait for SSH")
    _add_create_args(create_parser)
    _add_wait_args(create_parser)
    _add_retry_args(create_parser)
    create_parser.add_argument("--dry-run", action="store_true")

    stop_parser = subparsers.add_parser("stop", help="Stop a pod")
    stop_parser.add_argument("--pod-id", required=True)
    _add_retry_args(stop_parser)

    terminate_parser = subparsers.add_parser("terminate", help="Terminate (delete) a pod")
    terminate_parser.add_argument("--pod-id", required=True)
    _add_retry_args(terminate_parser)

    down_parser = subparsers.add_parser("down", help="Alias for terminate")
    down_parser.add_argument("--pod-id", required=True)
    _add_retry_args(down_parser)

    setup_parser = subparsers.add_parser("setup", help="Provision an existing pod over SSH")
    setup_parser.add_argument("--pod-id", required=True)
    _add_wait_args(setup_parser)
    _add_retry_args(setup_parser)
    _add_setup_args(setup_parser, require_repo_url=True)

    train_parser = subparsers.add_parser("train", help="Launch remote training command")
    train_parser.add_argument("--pod-id", required=True)
    _add_wait_args(train_parser)
    _add_retry_args(train_parser)
    _add_train_args(train_parser)

    up_parser = subparsers.add_parser("up", help="Create pod and run provisioning")
    _add_create_args(up_parser)
    _add_wait_args(up_parser)
    _add_retry_args(up_parser)
    _add_setup_args(up_parser, require_repo_url=True)

    run_parser = subparsers.add_parser("run", help="Create pod, setup, and launch training")
    _add_create_args(run_parser)
    _add_wait_args(run_parser)
    _add_retry_args(run_parser)
    _add_setup_args(run_parser, require_repo_url=True)
    _add_train_args(run_parser, include_identity_file=False, include_repo_dir=False)
    run_parser.add_argument(
        "--terminate-after-launch",
        dest="terminate_after_launch",
        action="store_true",
        help="Terminate pod after training command is successfully launched",
    )
    run_parser.add_argument(
        "--terminate-on-success",
        dest="terminate_after_launch",
        action="store_true",
        help="Deprecated alias for --terminate-after-launch",
    )
    run_parser.add_argument("--terminate-on-failure", action="store_true")

    return parser


def _print_connection_info(conn) -> None:
    """Pretty-print pod connection information to stdout.

    Displays pod id, name, statuses, public IP and SSH command for the user.
    """

    print("==== POD READY ====")
    print(f"pod_id={conn.pod_id}")
    print(f"name={conn.name}")
    print(f"desired_status={conn.desired_status}")
    print(f"runtime_status={conn.runtime_status}")
    print(f"public_ip={conn.public_ip}")
    print(f"ssh_port={conn.ssh_port}")


def main() -> None:
    """CLI entrypoint that parses args and dispatches to the requested action.

    Interprets subcommands and calls into lifecycle, provision and workflow
    helpers to perform the requested operation.
    """

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create":
        pod_spec = lifecycle.load_pod_spec(args.pod_spec)
        docker_args = lifecycle.load_bootstrap_as_docker_args(args.bootstrap_script)
        if args.dry_run:
            print("==== DRY RUN ====")
            print(f"name={pod_spec.name}")
            print(f"image_name={pod_spec.image_name}")
            print(f"gpu_type_id={pod_spec.gpu_type_id}")
            print(f"cloud_type={pod_spec.cloud_type}")
            print(f"container_disk_in_gb={pod_spec.container_disk_in_gb}")
            print(f"volume_in_gb={pod_spec.volume_in_gb}")
            print(f"ports={pod_spec.ports}")
            print(f"identity_file={pod_spec.expanded_identity_file}")
            print(f"env={pod_spec.env}")
            print("---- docker_args ----")
            print(docker_args)
            return

        lifecycle.configure_api_key_from_env()
        retry_policy = _retry_policy_from_args(args)
        conn = lifecycle.resolve_or_create_pod(
            pod_spec,
            docker_args,
            reuse_if_exists=args.reuse_if_exists,
            timeout_s=args.timeout_s,
            poll_s=args.poll_s,
            retry_policy=retry_policy,
        )
        _print_connection_info(conn)
        print(f"ssh_command={conn.ssh_command(pod_spec.expanded_identity_file)}")
        return

    lifecycle.configure_api_key_from_env()
    retry_policy = _retry_policy_from_args(args)

    if args.command == "stop":
        lifecycle.stop_pod(args.pod_id, retry_policy=retry_policy)
        print(f"Stopped pod {args.pod_id}")
        return

    if args.command in {"terminate", "down"}:
        lifecycle.terminate_pod(args.pod_id, retry_policy=retry_policy)
        print(f"Terminated pod {args.pod_id}")
        return

    if args.command == "setup":
        setup_spec = _setup_spec_from_args(args)
        conn = lifecycle.get_connection_info_for_pod(
            args.pod_id,
            timeout_s=args.timeout_s,
            poll_s=args.poll_s,
            retry_policy=retry_policy,
            identity_file=setup_spec.expanded_identity_file,
        )
        setup_pod(conn, setup_spec)
        print(f"Setup complete for pod {args.pod_id}")
        return

    if args.command == "train":
        train_spec = _train_spec_from_args(args)
        conn = lifecycle.get_connection_info_for_pod(
            args.pod_id,
            timeout_s=args.timeout_s,
            poll_s=args.poll_s,
            retry_policy=retry_policy,
            identity_file=train_spec.expanded_identity_file,
        )
        log_path = workflows.train_on_pod(conn, train_spec)
        print(f"Training started in tmux session '{train_spec.tmux_session_name}'")
        print(f"log_path={log_path}")
        return

    if args.command == "up":
        pod_spec = lifecycle.load_pod_spec(args.pod_spec)
        setup_spec = _setup_spec_from_args(args)
        conn = workflows.up_workflow(
            pod_spec=pod_spec,
            bootstrap_script=args.bootstrap_script,
            setup_spec=setup_spec,
            reuse_if_exists=args.reuse_if_exists,
            timeout_s=args.timeout_s,
            poll_s=args.poll_s,
            retry_policy=retry_policy,
        )
        _print_connection_info(conn)
        print(f"ssh_command={conn.ssh_command(setup_spec.expanded_identity_file)}")
        return

    if args.command == "run":
        pod_spec = lifecycle.load_pod_spec(args.pod_spec)
        setup_spec = _setup_spec_from_args(args)
        train_spec = _train_spec_from_args(args)
        conn = workflows.run_workflow(
            pod_spec=pod_spec,
            bootstrap_script=args.bootstrap_script,
            setup_spec=setup_spec,
            train_spec=train_spec,
            reuse_if_exists=args.reuse_if_exists,
            timeout_s=args.timeout_s,
            poll_s=args.poll_s,
            retry_policy=retry_policy,
            terminate_after_launch=args.terminate_after_launch,
            terminate_on_failure=args.terminate_on_failure,
        )
        if not args.terminate_after_launch:
            _print_connection_info(conn)
        print("Run workflow finished")
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
