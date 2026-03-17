import os
import shutil
import json
import uuid

import pytest

import scaling_llms.experiments as experiments
from scaling_llms.experiments import make_gdrive_local_disk_nested_experiment_runner


@pytest.mark.integration
def test_nested_runner_syncs_artifacts_to_gdrive(
    monkeypatch,
    tmp_path,
):
    if shutil.which("rclone") is None:
        pytest.skip("rclone not installed")

    test_suffix = uuid.uuid4().hex[:8]
    exp_name = f"nested-sync-test-{test_suffix}"
    run_name = f"run-local-to-gdrive-{test_suffix}"
    local_project_root = tmp_path / f"local_project_{test_suffix}"

    class _FakeTrainer:
        class _Cfg:
            device_name = "cpu"

        cfg = _Cfg()

        def train(self, max_steps=None):
            return None

    def _fake_init_trainer(*, run, **kwargs):
        run.log_metadata({"synced": True}, "local_sync_marker", format="json")
        return _FakeTrainer()

    monkeypatch.setattr(experiments, "init_trainer", _fake_init_trainer)

    runner = make_gdrive_local_disk_nested_experiment_runner(
        exp_name=exp_name,
        project_root=str(local_project_root),
        is_dev=True,
    )

    try:
        trainer = runner.start(
            run_name=run_name,
            dataset_kwargs=dict(
                dataset_name="super_glue",
                dataset_config="cb",
                train_split="train[:1%]",
                eval_split="test[:1%]",
                tokenizer_name="gpt2_tiktoken",
                text_field="premise",
            ),
            dataloader_kwargs=dict(
                seq_len=16,
                train_batch_size=2,
                eval_batch_size=2,
                start_sample_idx=0,
                seed=1,
            ),
            gpt_hparams=dict(n_embd=16, n_layer=1, n_head=1),
            trainer_kwargs=dict(
                num_steps=1,
                lr=3e-4,
                accum_steps=1,
                lr_schedule="linear",
                enable_tb=False,
                net_log_freq=1,
                sys_log_freq=1,
                eval_log_freq=1,
                ckpt_log_freq=1,
            ),
            max_steps=1,
        )

        assert isinstance(trainer, _FakeTrainer)

        identity = experiments.RunIdentity(exp_name, run_name)
        assert runner.remote_run_registry.run_exists(identity)
        assert runner.local_run_registry.run_exists(identity)

        local_run = runner.local_run_registry.get_run(identity)
        remote_run = runner.remote_run_registry.get_run(identity)
        assert local_run.root != remote_run.root

        marker_path = remote_run.artifacts.metadata_path("local_sync_marker.json")
        assert marker_path.exists(), "Expected local artifact to be synced to remote (gdrive) run dir"
        assert json.loads(marker_path.read_text()) == {"synced": True}
    finally:
        for registry in (runner.remote_run_registry, runner.local_run_registry):
            try:
                registry.delete_experiment(exp_name, confirm=False)
            except FileNotFoundError:
                pass
