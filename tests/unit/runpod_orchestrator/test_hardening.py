from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from runpod_orchestrator_old import cli, lifecycle, provision, ssh, workflows
from runpod_orchestrator_old.models import PodConnectionInfo, PodSpec, RetryPolicy, SetupSpec, TrainSpec


def _conn() -> PodConnectionInfo:
    return PodConnectionInfo(
        pod_id="pod-1",
        name="pod",
        desired_status="RUNNING",
        runtime_status="RUNNING",
        public_ip="1.2.3.4",
        ssh_port=22022,
    )


def test_extract_ssh_port_from_ports_prefers_private_22() -> None:
    ports = [
        {"privatePort": 80, "publicPort": 10080},
        {"privatePort": 22, "publicPort": 10022},
        {"privatePort": 22, "publicPort": 10023},
    ]
    assert lifecycle._extract_ssh_port_from_ports(ports) == 10022


def test_resolve_or_create_pod_reuses_existing_ready_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = PodSpec(name="n", image_name="img", gpu_type_id="gpu", identity_file="~/.ssh/id")
    existing = {
        "id": "pod-x",
        "name": "n",
        "runtime": {
            "status": "RUNNING",
            "ports": [{"privatePort": 22, "publicPort": 10022, "isIpPublic": True, "ip": "1.2.3.4"}],
        },
    }

    monkeypatch.setattr(lifecycle, "find_pod_by_name", lambda *args, **kwargs: existing)
    create_mock = MagicMock()
    monkeypatch.setattr(lifecycle, "create_pod", create_mock)
    monkeypatch.setattr(lifecycle.ssh_utils, "probe_ssh_connectivity", lambda *args, **kwargs: True)

    result = lifecycle.resolve_or_create_pod(
        spec,
        docker_args="bash -lc true",
        reuse_if_exists=True,
        timeout_s=1,
        poll_s=0,
        retry_policy=RetryPolicy(max_attempts=1, retry_base_s=0),
    )

    assert result.pod_id == "pod-x"
    create_mock.assert_not_called()


def test_resolve_or_create_pod_creates_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = PodSpec(name="n", image_name="img", gpu_type_id="gpu")
    monkeypatch.setattr(lifecycle, "find_pod_by_name", lambda *args, **kwargs: None)
    monkeypatch.setattr(lifecycle, "create_pod", lambda *args, **kwargs: {"id": "pod-created"})
    monkeypatch.setattr(lifecycle, "wait_for_ssh_ready", lambda *args, **kwargs: _conn())

    result = lifecycle.resolve_or_create_pod(
        spec,
        docker_args="bash -lc true",
        reuse_if_exists=True,
        timeout_s=10,
        poll_s=1,
        retry_policy=RetryPolicy(),
    )
    assert result.pod_id == "pod-1"


def test_run_ssh_command_builds_expected_ssh_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = tmp_path / "id_rsa"
    key.write_text("k", encoding="utf-8")
    conn = _conn()

    run_mock = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(ssh.subprocess, "run", run_mock)

    cmd = "echo 'hello' && cat /tmp/x | grep hello > /tmp/y"
    ssh.run_ssh_command(conn, cmd, identity_file=str(key))

    called_cmd = run_mock.call_args.args[0]
    expected_remote = f"bash -lc {shlex.quote(cmd)}"
    assert called_cmd[-1] == expected_remote
    assert called_cmd[0] == "ssh"
    assert f"root@{conn.public_ip}" in called_cmd


def test_copy_rclone_config_creates_remote_dir_and_scp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = tmp_path / "id_rsa"
    key.write_text("k", encoding="utf-8")
    local_rc = tmp_path / "rclone.conf"
    local_rc.write_text("[x]", encoding="utf-8")

    spec = SetupSpec(
        identity_file=str(key),
        repo_url="https://example.com/repo.git",
        rclone_config_local=str(local_rc),
        rclone_config_remote="/root/.config/rclone/rclone.conf",
    )

    ssh_calls: list[str] = []
    monkeypatch.setattr(
        provision,
        "run_ssh_command",
        lambda _conn, command, **kwargs: ssh_calls.append(command),
    )
    scp_mock = MagicMock()
    monkeypatch.setattr(provision, "scp_to_pod", scp_mock)

    provision.copy_rclone_config(_conn(), spec)

    assert ssh_calls and "mkdir -p /root/.config/rclone" in ssh_calls[0]
    scp_mock.assert_called_once()


def test_clone_or_update_repo_with_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = tmp_path / "id_rsa"
    key.write_text("k", encoding="utf-8")
    spec = SetupSpec(
        identity_file=str(key),
        repo_url="https://example.com/repo.git",
        repo_dir="/workspace/repos/scaling-llms",
        repo_branch="main",
    )

    calls: list[str] = []
    monkeypatch.setattr(
        provision,
        "run_ssh_command",
        lambda _conn, command, **kwargs: calls.append(command),
    )

    provision.clone_or_update_repo(_conn(), spec)
    assert len(calls) == 1
    cmd = calls[0]
    assert "fetch origin main --prune" in cmd
    assert "checkout main" in cmd
    assert "reset --hard origin/main" in cmd
    assert "origin/HEAD" not in cmd


def test_setup_pod_calls_steps_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = tmp_path / "id_rsa"
    key.write_text("k", encoding="utf-8")
    spec = SetupSpec(identity_file=str(key), repo_url="https://example.com/repo.git")

    order: list[str] = []
    monkeypatch.setattr(provision, "copy_rclone_config", lambda *_args, **_kwargs: order.append("copy"))
    monkeypatch.setattr(provision, "clone_or_update_repo", lambda *_args, **_kwargs: order.append("clone"))
    monkeypatch.setattr(provision, "poetry_install_remote", lambda *_args, **_kwargs: order.append("poetry"))
    monkeypatch.setattr(
        provision,
        "create_jupyter_kernel_remote",
        lambda *_args, **_kwargs: order.append("kernel"),
    )

    provision.setup_pod(_conn(), spec)
    assert order == ["copy", "clone", "poetry", "kernel"]


def test_build_tmux_training_command() -> None:
    spec = TrainSpec(
        command="poetry run python train.py --x 1",
        repo_dir="/workspace/repos/scaling-llms",
        job_session_name="train",
        log_path="/workspace/runs/train.log",
    )
    cmd = workflows._build_tmux_training_command(spec)
    assert "tmux new-session -d -s" in cmd
    assert "tee -a" in cmd
    assert "/workspace/runs/train.log" in cmd


def test_run_workflow_terminates_on_failure_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _conn()
    monkeypatch.setattr(workflows, "up_workflow", lambda **kwargs: conn)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("train failed")

    monkeypatch.setattr(workflows, "train_on_pod", _boom)
    terminate_mock = MagicMock()
    monkeypatch.setattr(workflows.lifecycle, "terminate_pod", terminate_mock)

    with pytest.raises(RuntimeError):
        workflows.run_workflow(
            pod_spec=PodSpec(name="n", image_name="img", gpu_type_id="gpu"),
            bootstrap_script=Path("bootstrap.sh"),
            setup_spec=SetupSpec(repo_url="https://example.com/repo.git"),
            train_spec=TrainSpec(command="echo hi"),
            reuse_if_exists=True,
            timeout_s=1,
            poll_s=0,
            retry_policy=RetryPolicy(),
            terminate_after_launch=False,
            terminate_on_failure=True,
        )

    terminate_mock.assert_called_once_with(conn.pod_id, retry_policy=RetryPolicy())


def test_cli_run_dispatches_to_run_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "prog",
        "run",
        "--pod-spec",
        "./src/runpod_orchestrator/pod_spec.yaml",
        "--bootstrap-script",
        "./src/runpod_orchestrator/bootstrap.sh",
        "--repo-url",
        "https://example.com/repo.git",
        "--cmd",
        "echo hi",
    ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(cli.lifecycle, "configure_api_key_from_env", lambda: None)
    monkeypatch.setattr(
        cli.lifecycle,
        "load_pod_spec",
        lambda _path: PodSpec(name="n", image_name="img", gpu_type_id="gpu"),
    )
    monkeypatch.setattr(cli, "_print_connection_info", lambda *_args, **_kwargs: None)

    run_mock = MagicMock(return_value=_conn())
    monkeypatch.setattr(cli.workflows, "run_workflow", run_mock)

    cli.main()
    assert run_mock.call_count == 1
