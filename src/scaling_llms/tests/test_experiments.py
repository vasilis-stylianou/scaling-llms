import pytest

from scaling_llms.experiments import ExperimentRunner
from scaling_llms.constants import RUN_FILES, METRIC_CATS
from scaling_llms.trainer import Trainer


EXPERIMENT_NAME = "test_experiment_runner" 
RUN_NAME = "run_test"

@pytest.fixture(autouse=True)
def cleanup_experiments():
    exp = ExperimentRunner(EXPERIMENT_NAME, RUN_NAME, is_dev=True)

    yield  # Run the test

    # Cleanup after test
    exp.delete_experiment(confirm=False)


@pytest.fixture
def data_kwargs():
    return dict(
        dataset_name="wikitext",
        dataset_config="wikitext-103-v1",
        seq_len=16,
        train_batch_size=8,
        eval_batch_size=8,
        train_split="train[:1000]",
        eval_split="test[:1000]",
        train_tokens_budget=1024,
        start_sample_idx=0,
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
        train_log_freq=1,
        net_log_freq=2,
        sys_log_freq=2,
        eval_log_freq=2,
        ckpt_log_freq=2
    )


def test_experiment_runner_start(data_kwargs, gpt_hparams, trainer_kwargs):

    exp = ExperimentRunner(EXPERIMENT_NAME, RUN_NAME, is_dev=True)
    trainer = exp.start(
        data_kwargs=data_kwargs,
        gpt_hparams=gpt_hparams,
        trainer_kwargs=trainer_kwargs,
    )
    assert isinstance(trainer, Trainer)

    # Check that the expected metadata and checkpoint files have been created in the run directory
    run = trainer.run
    assert run is not None
    assert run.root.exists()

    run.start(resume=True)
    for filename in RUN_FILES.as_list():
        # Metadata
        if filename.endswith(".json") or filename.endswith("log"):
            file_path = run.get_metadata_path(filename)
            assert file_path.exists(), f"Expected metadata file {filename} to be logged"
        
        # Checkpoints
        elif filename.endswith(".pt"):
            file_path = run.get_checkpoint_path(filename)
            assert file_path.exists(), f"Expected checkpoint file {filename} to be saved"
        else:
            raise ValueError(f"Unexpected file name {filename} in RUN_FILES")

    # Metrics
    for cat in METRIC_CATS.as_list():
        metric_path = run.get_metric_path(cat)
        assert metric_path.exists() and metric_path.is_file(), f"Expected metric file for category {cat} to be created"

    # TensorBoard logs
    tb_dir = run.get_tb_dir()
    assert tb_dir.exists() and tb_dir.is_dir(), "TensorBoard log directory does not exist"
    assert any(tb_dir.iterdir()), "TensorBoard log directory is empty"


def test_experiment_runner_resume(data_kwargs, gpt_hparams, trainer_kwargs):
    exp = ExperimentRunner(EXPERIMENT_NAME, RUN_NAME, is_dev=True)
    trainer = exp.start(
        data_kwargs=data_kwargs,
        gpt_hparams=gpt_hparams,
        trainer_kwargs=trainer_kwargs,
    )

    # Now attempt to resume the same run from checkpoints
    # Case 1: Attempt to resume from last checkpoint
    # With max_steps not provided, should fail since training is already complete
    with pytest.raises(ValueError, match=r"Training already complete or beyond target"):
        exp.resume(RUN_FILES.last_ckpt)

    # Case 2: Now attempt to resume from best checkpoint which should work because
    # best_ckpt is saved at step 2 (eval_log_freq) and num_steps is 3, 
    # so there is still one step left to train
    resumed = exp.resume(RUN_FILES.best_ckpt)
    assert resumed.step_idx == trainer_kwargs["num_steps"], "Resumed trainer did not load the best checkpoint"

    # Case 3: Now attempt to resume from last checkpoint but allow additional steps so it doesn't error out
    num_resume_steps = 2
    trainer_resumed = exp.resume(RUN_FILES.last_ckpt, max_steps=trainer_kwargs["num_steps"] + num_resume_steps)
    assert trainer_resumed.step_idx - trainer.step_idx == num_resume_steps

   
def test_experiment_runner_start_from_checkpoint(data_kwargs, gpt_hparams, trainer_kwargs):
    exp = ExperimentRunner(EXPERIMENT_NAME, RUN_NAME, is_dev=True)
    _ = exp.start(
        data_kwargs=data_kwargs,
        gpt_hparams=gpt_hparams,
        trainer_kwargs=trainer_kwargs,
    )

    # Now attempt to start a new run initialized from the best checkpoint of the previous run
    new_exp = ExperimentRunner(EXPERIMENT_NAME, "new_run", is_dev=True)
    data_kwargs2 = data_kwargs.copy()
    num_steps = 5
    trainer2 = new_exp.start_from_checkpoint(
        data_kwargs=data_kwargs2,
        ckpt_exp_name=EXPERIMENT_NAME,
        ckpt_run_name=RUN_NAME,
        ckpt_filename=RUN_FILES.best_ckpt,
        max_steps=num_steps
    )
    assert trainer2.step_idx == num_steps, "Trainer initialized from checkpoint did not reset step index correctly"
    new_exp.delete_run(confirm=False)  # Clean up new run after test

    # Now attempt to start a new run initialized from the same checkpoint 
    # but with a different sequence length which should raise an error
    data_kwargs2["seq_len"] = data_kwargs2["seq_len"] // 2
    with pytest.raises(ValueError, match=r"Sequence length mismatch between old run"):
        new_exp.start_from_checkpoint(
            data_kwargs=data_kwargs2,
            ckpt_exp_name=EXPERIMENT_NAME,
            ckpt_run_name=RUN_NAME,
            ckpt_filename=RUN_FILES.best_ckpt,
        )