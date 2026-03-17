from __future__ import annotations

import argparse
from pathlib import Path

from runpod_orchestrator import lifecycle
from runpod_orchestrator.models import RetryPolicy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the legacy launcher compatibility wrapper.

    Returns the populated argparse namespace for the launcher tool.
    """

    parser = argparse.ArgumentParser(
        description="Legacy compatibility wrapper for create+wait flow."
    )
    parser.add_argument("--pod-spec", type=Path, required=True)
    parser.add_argument("--bootstrap-script", type=Path, required=True)

    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument("--reuse-if-exists", dest="reuse_if_exists", action="store_true")
    reuse_group.add_argument("--no-reuse", dest="reuse_if_exists", action="store_false")
    parser.set_defaults(reuse_if_exists=True)

    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--poll-s", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-base-s", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the legacy launcher flow: create/wait and print connection info.

    Implements a backwards-compatible wrapper around `lifecycle.resolve_or_create_pod`.
    """

    args = parse_args()
    retry_policy = RetryPolicy(
        max_attempts=args.max_attempts,
        retry_base_s=args.retry_base_s,
    )

    spec = lifecycle.load_pod_spec(args.pod_spec)
    docker_args = lifecycle.load_bootstrap_as_docker_args(args.bootstrap_script)

    if args.dry_run:
        print("==== DRY RUN ====")
        print(f"name={spec.name}")
        print(f"image_name={spec.image_name}")
        print(f"gpu_type_id={spec.gpu_type_id}")
        print(f"cloud_type={spec.cloud_type}")
        print(f"container_disk_in_gb={spec.container_disk_in_gb}")
        print(f"volume_in_gb={spec.volume_in_gb}")
        print(f"ports={spec.ports}")
        print(f"identity_file={spec.expanded_identity_file}")
        print(f"env={spec.env}")
        print("---- docker_args ----")
        print(docker_args)
        return

    lifecycle.configure_api_key_from_env()
    conn = lifecycle.resolve_or_create_pod(
        spec,
        docker_args,
        reuse_if_exists=args.reuse_if_exists,
        timeout_s=args.timeout_s,
        poll_s=args.poll_s,
        retry_policy=retry_policy,
    )

    print("==== POD READY ====")
    print(f"pod_id={conn.pod_id}")
    print(f"name={conn.name}")
    print(f"desired_status={conn.desired_status}")
    print(f"runtime_status={conn.runtime_status}")
    print(f"public_ip={conn.public_ip}")
    print(f"ssh_port={conn.ssh_port}")
    print(f"ssh_command={conn.ssh_command(spec.expanded_identity_file)}")


if __name__ == "__main__":
    main()