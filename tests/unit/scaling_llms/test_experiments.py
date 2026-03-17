import pytest
from contextlib import contextmanager

from scaling_llms.experiments import (
    make_gdrive_experiment_runner,
)
import scaling_llms.experiments as experiments
from scaling_llms.constants import CKPT_FILES, METADATA_FILES, METRIC_CATS
from scaling_llms.registries.datasets.identity import DatasetIdentity
from scaling_llms.trainer import Trainer


EXPERIMENT_NAME = "test_experiment_runner" 
RUN_NAME = "run_test"

@pytest.fixture(autouse=True)
def cleanup_experiments(dataset_kwargs):
    exp = make_gdrive_experiment_runner(EXPERIMENT_NAME, is_dev=True)
    try:
        exp.delete_experiment(confirm=False)
    except Exception:
        pass

    # Delete stale test dataset from data registry so it gets
    # re-registered with the current schema (incl. dataset_info.json)
    try:
        dataset_id = DatasetIdentity(**dataset_kwargs)
        exp.data_registry.delete_dataset(identity=dataset_id, confirm=False)
    except Exception:
        pass

    yield

    # Cleanup after test
    try:
        exp.delete_experiment(confirm=False)
    except Exception:
        pass


@pytest.fixture
def exp():
    return make_gdrive_experiment_runner(EXPERIMENT_NAME, is_dev=True)


@pytest.fixture
def dataset_kwargs():
    return dict(
        dataset_name="glue",
        dataset_config="sst2",
        train_split="train[:10%]",
        eval_split="test[:10%]",
        tokenizer_name="gpt2_tiktoken",
        text_field="sentence",
    )

@pytest.fixture
def dataloader_kwargs():
    return dict(
        seq_len=16,
        train_batch_size=8,
        eval_batch_size=8,
        start_sample_idx=0,
        seed=42,
    )

@pytest.fixture
def gpt_hparams():
    return dict(
        n_embd=32,
        n_layer=1,
        n_head=2,
    )

@pytest.fixture
def trainer_kwargs():
    return dict(
        num_steps=3, 
        lr=3e-4,
        accum_steps=2,
        lr_schedule="linear",
        enable_tb=True, 
        net_log_freq=2,
        sys_log_freq=2,
        eval_log_freq=2,
        ckpt_log_freq=2
    )


def test_experiment_runner_start(exp, dataset_kwargs, dataloader_kwargs, gpt_hparams, trainer_kwargs):

    trainer = exp.start(
        run_name=RUN_NAME,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        gpt_hparams=gpt_hparams,
        trainer_kwargs=trainer_kwargs,
    )
    assert isinstance(trainer, Trainer)

    runs_df = exp.run_registry.get_runs_as_df(EXPERIMENT_NAME)
    row = runs_df.query(f"run_name == '{RUN_NAME}'").iloc[0]
    assert row.device_name == trainer.cfg.device_name

    # Check that the expected metadata and checkpoint files have been created in the run directory
    run = trainer.run
    assert run is not None
    assert run.root.exists()

    run.start(resume=True)
    # Metadata
    for filename in METADATA_FILES.as_list():
        if filename.endswith(".json") or filename.endswith("log"):
            file_path = run.artifacts.metadata_path(filename)
            assert file_path.exists(), f"Expected metadata file {filename} to be logged"
        else:
            raise ValueError(f"Unexpected file name {filename} in METADATA_FILES")

    # Checkpoints
    for filename in CKPT_FILES.as_list():
        if filename.endswith(".pt"):
            file_path = run.artifacts.checkpoint_path(filename)
            assert file_path.exists(), f"Expected checkpoint file {filename} to be saved"
        else:
            raise ValueError(f"Unexpected file name {filename} in CKPT_FILES")

    # Metrics
    for cat in METRIC_CATS.as_list():
        metric_path = run.artifacts.metric_path(cat)
        assert metric_path.exists() and metric_path.is_file(), f"Expected metric file for category {cat} to be created"

    # TensorBoard logs
    tb_dir = run.tb_dir
    assert tb_dir.exists() and tb_dir.is_dir(), "TensorBoard log directory does not exist"
    assert any(tb_dir.iterdir()), "TensorBoard log directory is empty"

    run.close()


def test_experiment_runner_resume(exp, dataset_kwargs, dataloader_kwargs, gpt_hparams, trainer_kwargs):
    trainer = exp.start(
        run_name=RUN_NAME,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        gpt_hparams=gpt_hparams,
        trainer_kwargs=trainer_kwargs,
    )

    # Now attempt to resume the same run from checkpoints
    # Case 1: Attempt to resume from last checkpoint
    # With max_steps not provided, should fail since training is already complete
    with pytest.raises(ValueError, match=r"Training already complete or beyond target"):
        exp.resume(run_name=RUN_NAME, ckpt_filename=CKPT_FILES.last_ckpt)

    # Case 2: Resume from best checkpoint with explicit max_steps.
    # Checkpoints should load and training should resume from the next step after the checkpoint
    resumed = exp.resume(
        run_name=RUN_NAME,
        ckpt_filename=CKPT_FILES.best_ckpt,
        max_steps=trainer_kwargs["num_steps"] + 1,
    )
    assert resumed.step_idx == trainer_kwargs["num_steps"] + 1

    # Case 3: Now attempt to resume from last checkpoint but allow additional steps so it doesn't error out
    num_resume_steps = 2
    trainer_resumed = exp.resume(run_name=RUN_NAME, ckpt_filename=CKPT_FILES.last_ckpt, max_steps=trainer_kwargs["num_steps"] + num_resume_steps)
    assert trainer_resumed.step_idx - trainer.step_idx == num_resume_steps

   
def test_experiment_runner_start_from_checkpoint(exp, dataset_kwargs, dataloader_kwargs, gpt_hparams, trainer_kwargs):
    _ = exp.start(
        run_name=RUN_NAME,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        gpt_hparams=gpt_hparams,
        trainer_kwargs=trainer_kwargs,
    )

    # Now attempt to start a new run initialized from the best checkpoint of the previous run
    num_steps = 5
    trainer2 = exp.start_from_checkpoint(
        run_name="new_run",
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        ckpt_exp_name=EXPERIMENT_NAME,
        ckpt_run_name=RUN_NAME,
        ckpt_filename=CKPT_FILES.best_ckpt,
        max_steps=num_steps
    )
    assert trainer2.step_idx == num_steps, "Trainer initialized from checkpoint did not reset step index correctly"
    exp.delete_run("new_run", confirm=False)  # Clean up new run after test

    # Now attempt to start a new run initialized from the same checkpoint 
    # but with a different sequence length which should raise an error
    bad_dl_kwargs = {**dataloader_kwargs, "seq_len": dataloader_kwargs["seq_len"] // 2}
    with pytest.raises(ValueError, match=r"Sequence length mismatch between old run"):
        exp.start_from_checkpoint(
            run_name="new_run_2",
            dataset_kwargs=dataset_kwargs,
            dataloader_kwargs=bad_dl_kwargs,
            ckpt_exp_name=EXPERIMENT_NAME,
            ckpt_run_name=RUN_NAME,
            ckpt_filename=CKPT_FILES.best_ckpt,
        )


def test_make_gdrive_experiment_runner_uses_rclone_transfer_mode(monkeypatch):
    class _FakeRunRegistry:
        def run_exists(self, _identity):
            return False

        @contextmanager
        def managed_run(self, _identity, resume=False, overwrite=False):
            class _Run:
                metadata_dir = None

            yield _Run()

        def set_device_name(self, _identity, _device_name):
            return None

    class _FakeDataRegistry:
        pass

    class _FakeTrainer:
        class _Cfg:
            device_name = "cpu"

        cfg = _Cfg()

        def train(self, max_steps=None):
            return None

    captured: dict[str, str] = {}

    def _fake_init_trainer(*, transfer_mode, **kwargs):
        captured["transfer_mode"] = transfer_mode
        return _FakeTrainer()

    monkeypatch.setattr(experiments, "make_gdrive_run_registry", lambda **_: _FakeRunRegistry())
    monkeypatch.setattr(experiments, "make_gdrive_data_registry", lambda **_: _FakeDataRegistry())
    monkeypatch.setattr(experiments, "init_trainer", _fake_init_trainer)

    runner = make_gdrive_experiment_runner(
        exp_name="exp-test-rclone",
        is_dev=True,
        transfer_mode="rclone",
    )

    assert isinstance(runner, experiments.ExperimentRunner)
    assert runner.transfer_mode == "rclone"

    result = runner.start(
        run_name="run-test-rclone",
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

    assert result is not None
    assert captured["transfer_mode"] == "rclone"