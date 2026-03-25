from __future__ import annotations

import time
from dataclasses import replace

from runpod_orchestrator.clients.ssh import SSHClient


def test_create_pod(orchestrator, unique_config, pod_tracker):
    result = orchestrator.create(unique_config)
    pod_tracker(result.pod.pod_id)

    assert result.pod.pod_id
    assert result.pod.public_ip
    assert result.pod.ssh_port > 0
    assert result.pod.is_ssh_ready


def test_provision_pod(orchestrator, unique_config, pod_tracker):
    created = orchestrator.create(unique_config)
    pod_tracker(created.pod.pod_id)

    result = orchestrator.provision(unique_config, pod_id=created.pod.pod_id)

    ssh = SSHClient(unique_config.provisioning.expanded_identity_file)
    proc = ssh.run(
        result.pod,
        f"test -d {unique_config.provisioning.repo_dir} && echo repo-ready",
    )
    assert "repo-ready" in proc.stdout


def test_provision_poetry_can_import_scaling_llms(orchestrator, unique_config, pod_tracker):
    created = orchestrator.create(unique_config)
    pod_tracker(created.pod.pod_id)

    result = orchestrator.provision(unique_config, pod_id=created.pod.pod_id)

    ssh = SSHClient(unique_config.provisioning.expanded_identity_file)
    proc = ssh.run(
        result.pod,
        (
            "cd /workspace/repos/scaling-llms && "
            "poetry run python -c \"import scaling_llms; print('scaling-llms-import-ok')\""
        ),
    )
    assert "scaling-llms-import-ok" in proc.stdout


def test_submit_job(orchestrator, unique_config, pod_tracker):
    created = orchestrator.create(unique_config)
    pod_tracker(created.pod.pod_id)

    orchestrator.provision(unique_config, pod_id=created.pod.pod_id)

    config = replace(
        unique_config,
        job_spec=replace(
            unique_config.job_spec,
            command='python -c "print(\\"hello from pod\\")"',
            tmux_session_name="smoke-job",
            log_path="/workspace/jobs/smoke-job.log",
        ),
    )

    result = orchestrator.submit(config, pod_id=created.pod.pod_id)

    ssh = SSHClient(config.job_spec.expanded_identity_file)
    proc = ssh.run(result.pod, "sleep 2 && cat /workspace/jobs/smoke-job.log")
    assert "hello from pod" in proc.stdout


def test_stop_and_resume_pod(orchestrator, unique_config, pod_tracker):
    created = orchestrator.create(unique_config)
    pod_tracker(created.pod.pod_id)

    orchestrator.stop(created.pod.pod_id)
    time.sleep(5)

    resumed = orchestrator.resume(unique_config, pod_id=created.pod.pod_id)
    assert resumed.pod.pod_id == created.pod.pod_id
    assert resumed.pod.is_ssh_ready


def test_run_end_to_end(orchestrator, unique_config, pod_tracker):
    config = replace(
        unique_config,
        job_spec=replace(
            unique_config.job_spec,
            command='python -c "print(\\"end-to-end ok\\")"',
            tmux_session_name="e2e-job",
            log_path="/workspace/jobs/e2e-job.log",
        ),
    )

    result = orchestrator.run(config)
    pod_tracker(result.pod.pod_id)

    ssh = SSHClient(config.job_spec.expanded_identity_file)
    proc = ssh.run(result.pod, "sleep 2 && cat /workspace/jobs/e2e-job.log")
    assert "end-to-end ok" in proc.stdout