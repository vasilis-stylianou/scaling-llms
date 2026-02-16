import json
import shutil
import numpy as np
import pytest
import torch
from pathlib import Path

from scaling_llms.constants import (
    DATA_FILES,
    LOCAL_DEV_DATA_DIR,
    METRIC_CATS,
    PROJECT_DEV_NAME,
    RUN_DIRS,
    TOKENIZED_CACHE_DIR_NAME,
    METRIC_SCHEMA,
)
from scaling_llms.models import GPTConfig, GPTModel
from scaling_llms.tracking.checkpoint import CheckpointManager
from scaling_llms.tracking.registries import (
    GoogleDriveDataRegistry,
    GoogleDriveRunRegistry,
)

EXPERIMENT_NAME = "test_tracking"
RUN_NAME = "run_test_tracking"

# ============================================================
# FIXTURES FOR REGISTRIES AND CHECKPOINT MANAGER TESTS
# ============================================================
@pytest.fixture
def dev_run_registry():
    return GoogleDriveRunRegistry(project_subdir=PROJECT_DEV_NAME)


@pytest.fixture
def dev_data_registry():
    return GoogleDriveDataRegistry(project_subdir=PROJECT_DEV_NAME)


@pytest.fixture
def local_dev_tokenized_cache_dir():
    path = Path(LOCAL_DEV_DATA_DIR) / TOKENIZED_CACHE_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(autouse=True)
def clean_registries_and_cache(
    dev_run_registry, dev_data_registry, local_dev_tokenized_cache_dir
):
    def _clean():
        # Clean up datasets
        df = dev_data_registry.get_datasets_as_df()
        for _, row in df.iterrows():
            dev_data_registry.delete_dataset(
                dataset_path=row.dataset_path, confirm=False
            )
        for item in local_dev_tokenized_cache_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

        # Clean up  experiments
        df_runs = dev_run_registry.get_runs_as_df().query(f"experiment_name == '{EXPERIMENT_NAME}'")
        for _, row in df_runs.iterrows():
            dev_run_registry.delete_experiment(row.experiment_name, confirm=False)

    # Clean up before each unit test
    _clean()

    yield

    # Clean up after each unit test
    _clean()


# ============================================================
# TESTS FOR RUN REGISTRY
# ============================================================
def test_run_registry_paths(dev_run_registry):

    # Check registry paths are correctly set up
    assert dev_run_registry.registry_root.parent == dev_run_registry.project_root, (
        "Registry root should be a direct child of project root"
    )
    assert dev_run_registry.db_path.parent == dev_run_registry.registry_root, (
        "Database path should be inside registry root"
    )
    assert dev_run_registry.artifacts_root.parent == dev_run_registry.registry_root, (
        "Artifacts root should be inside registry root"
    )

    # Create a run
    exp_name = EXPERIMENT_NAME
    run_name = RUN_NAME
    run = dev_run_registry.start_run(exp_name, run_name)
    run.close()

    # Check the run is registered correctly
    df_runs =  dev_run_registry.get_runs_as_df().query(f"experiment_name == '{exp_name}'")
    row = df_runs.iloc[0]
    assert len(df_runs) == 1, f"Expected 1 run, found {len(df_runs)}"
    assert row.experiment_name == exp_name, (
        f"Expected experiment_name '{exp_name}', got '{row.experiment_name}'"
    )
    assert row.run_name == run_name, f"Expected run_name '{run_name}', got '{row.run_name}'"

    # Check the run paths are correct
    exp_dir = dev_run_registry.get_experiment_dir(row.experiment_name)
    run_dir = dev_run_registry.get_run_dir(row.experiment_name, row.run_name)
    run_abs_path = Path(row.run_absolute_path)

    assert run_abs_path == dev_run_registry.artifacts_root / row.artifacts_path, (
        f"Run absolute path mismatch: {run_abs_path} != {dev_run_registry.artifacts_root / row.artifacts_path}"
    )
    assert run_abs_path.parent == exp_dir, (
        f"Run directory parent should be experiment directory: {run_abs_path.parent} != {exp_dir}"
    )
    assert run_abs_path == run_dir, f"Run absolute path should match run_dir: {run_abs_path} != {run_dir}"
    assert run_abs_path == run.root, f"Run absolute path should match run.root: {run_abs_path} != {run.root}"

    for subdir in RUN_DIRS.as_list():
        subdir_path = run_dir / subdir
        assert subdir_path.exists(), f"Run subdirectory {subdir} should exist"
        assert subdir_path.is_dir(), f"Run subdirectory {subdir} should be a directory"
        assert subdir_path == run[subdir], f"Run should be accessible via run[{subdir}]"

def test_run_registry_delete(dev_run_registry):
    exp_name = EXPERIMENT_NAME
    run_name = RUN_NAME
    run = dev_run_registry.start_run(exp_name, run_name)
    run.close()

    # Delete the run and check it's removed
    dev_run_registry.delete_run(exp_name, run_name, confirm=False)
    with pytest.raises(FileNotFoundError):
        dev_run_registry.get_run_dir(exp_name, run_name)

    # Delete the experiment and check it's removed
    dev_run_registry.delete_experiment(exp_name, confirm=False)
    with pytest.raises(FileNotFoundError):
        dev_run_registry.get_experiment_dir(exp_name)


def test_run_registry_resume(dev_run_registry):
    exp_name = EXPERIMENT_NAME
    run_name = RUN_NAME
    run1 = dev_run_registry.start_run(exp_name, run_name)
    run1.close()

    with pytest.raises(ValueError):
        dev_run_registry.start_run(exp_name, run_name, resume=False)

    run2 = dev_run_registry.start_run(exp_name, run_name, resume=True)
    run2.close()

    assert run1.root == run2.root, (
        "Resumed run should have the same root path as the original run"
    )


def test_run_logging(dev_run_registry):
    exp_name = EXPERIMENT_NAME
    run_name = RUN_NAME
    run = dev_run_registry.start_run(exp_name, run_name)

    # Log metrics
    metric_cat = METRIC_CATS.train
    metrics = {"loss": 0.5, "accuracy": 0.8}
    step = 1
    run.log_metrics({metric_cat: metrics}, step=step)

    # Log metadata
    metadata = {"param1": "value1", "param2": 42}
    metadata_filename = "metadata.json"
    run.log_metadata(metadata, metadata_filename, format="json")

    run.close()

    # Check paths exist
    metrics_path = run[RUN_DIRS.metrics] / f"{metric_cat}.jsonl"
    metadata_path = run[RUN_DIRS.metadata] / metadata_filename
    assert metrics_path.exists(), "Metrics file should exist"
    assert metadata_path.exists(), "Metadata file should exist"

    # Validate metrics
    expected_metric_data = [
        {
            METRIC_SCHEMA.metric: key,
            METRIC_SCHEMA.value: value,
            METRIC_SCHEMA.step: step,
        }
        for key, value in metrics.items()
    ]
    logged_metric_data = [json.loads(line) for line in metrics_path.open()]
    assert logged_metric_data == expected_metric_data, (
        "Logged metrics do not match expected metrics"
    )

    # Validate metadata
    logged_metadata = json.loads(metadata_path.read_text())
    assert logged_metadata == metadata, (
        "Logged metadata does not match expected metadata"
    )


# ============================================================
# TESTS FOR DATA REGISTRY
# ============================================================
def test_data_registry_paths(dev_data_registry):

    assert dev_data_registry.registry_root.parent == dev_data_registry.project_root, (
        "Registry root should be a direct child of project root"
    )
    assert dev_data_registry.db_path.parent == dev_data_registry.registry_root, (
        "Database path should be inside registry root"
    )
    assert dev_data_registry.datasets_root.parent == dev_data_registry.registry_root, (
        "Datasets root should be inside registry root"
    )


def test_data_registry_register_find_copy_delete(
    dev_data_registry, local_dev_tokenized_cache_dir
):
    dataset_name = "dataset_test_1"
    dataset_config = "dataset_config_1"
    train_split = None
    eval_split = None

    with pytest.raises(FileNotFoundError):
        dev_data_registry.find_dataset_path(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            train_split=train_split,
            eval_split=eval_split,
            raise_if_not_found=True,
        )

    # Create dummy tokenized dataset files in the local tokenized cache dir
    local_dataset_dir = local_dev_tokenized_cache_dir / "test"
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    local_train_mmap_path = local_dataset_dir / DATA_FILES.train_tokens
    local_eval_mmap_path = local_dataset_dir / DATA_FILES.eval_tokens

    n_tokens = 1024
    train = np.random.randint(0, 50000, size=n_tokens, dtype=np.uint32)
    eval_ = np.random.randint(0, 50000, size=n_tokens, dtype=np.uint32)

    local_train_mmap_path.write_bytes(train.tobytes())
    local_eval_mmap_path.write_bytes(eval_.tobytes())

    # Register the dataset
    dataset_path = dev_data_registry.register_dataset(
        local_train_mmap_path,
        local_eval_mmap_path,
        dataset_name,
        dataset_config,
        train_split,
        eval_split,
    )

    # Check the dataset is registered correctly
    df_datasets = dev_data_registry.get_datasets_as_df()
    assert len(df_datasets) == 1, f"Expected 1 dataset, found {len(df_datasets)}"
    found_path = dev_data_registry.find_dataset_path(
        dataset_name,
        dataset_config,
        train_split,
        eval_split,
    )
    assert dataset_path == found_path, (
        f"Registered dataset path mismatch: {dataset_path} != {found_path}"
    )

    # Remove the local dummy files
    local_train_mmap_path.unlink()
    local_eval_mmap_path.unlink()
    assert not local_train_mmap_path.exists(), (
        f"Train tokens file should be deleted: {local_train_mmap_path}"
    )
    assert not local_eval_mmap_path.exists(), (
        f"Eval tokens file should be deleted: {local_eval_mmap_path}"
    )

    # Copy the dataset to a local cache directory and check the files exist
    dev_data_registry.copy_dataset_to_local(dataset_path, local_dataset_dir)
    assert local_train_mmap_path.exists(), (
        f"Train tokens file should exist after copy: {local_train_mmap_path}"
    )
    assert local_eval_mmap_path.exists(), (
        f"Eval tokens file should exist after copy: {local_eval_mmap_path}"
    )

    # Delete the dataset and check it's removed
    dev_data_registry.delete_dataset(dataset_path, confirm=False)
    df_datasets = dev_data_registry.get_datasets_as_df()
    assert len(df_datasets) == 0, f"Expected 0 datasets after deletion, found {len(df_datasets)}"
    assert not dataset_path.exists(), f"Dataset path should not exist after deletion: {dataset_path}"


# ============================================================
# TESTS FOR CHECKPOINT MANAGER
# ============================================================
def _model_param_norm(model: GPTModel) -> torch.Tensor:
    return torch.sqrt(sum(p.float().norm() ** 2 for p in model.parameters()))


def test_checkpoint_manager_save_load(dev_run_registry):
    # Create a run to store checkpoints
    run = dev_run_registry.start_run(EXPERIMENT_NAME, RUN_NAME, resume=False)
    run.close()

    # Create a model and save a checkpoint
    torch.manual_seed(1)
    model1 = GPTModel(GPTConfig())
    ckpt_manager = CheckpointManager(
        run[RUN_DIRS.checkpoints],
        model1,
        optimizer=None,
        scaler=None,
        lr_scheduler=None,
    )
    trainer_state = {"step_idx": 10, "tokens_seen_total": 10000}
    ckpt_path = ckpt_manager.save(trainer_state=trainer_state, name="test_ckpt.pt")

    # Create a new model and check that its parameters are different from the original model
    torch.manual_seed(2)
    model2 = GPTModel(GPTConfig())
    norm1 = _model_param_norm(model1)
    norm2 = _model_param_norm(model2)
    assert not torch.allclose(norm1, norm2), (
        f"Model parameters should differ before loading checkpoint: {norm1} ≈ {norm2}"
    )

    # Load the checkpoint into the new model and check that the parameters now match
    ckpt_manager = CheckpointManager(run[RUN_DIRS.checkpoints], model2)
    loaded_trainer_state = ckpt_manager.load(ckpt_path)

    # Check that the trainer state is loaded correctly
    assert loaded_trainer_state == trainer_state, (
        f"Loaded trainer state should match saved state: {loaded_trainer_state} != {trainer_state}"
    )   

    # Check that the model parameters match after loading the checkpoint
    norm1_final = _model_param_norm(model1)
    norm2_final = _model_param_norm(model2)
    assert torch.allclose(norm1_final, norm2_final), (
        f"Model parameters should match after loading checkpoint: {norm1_final} != {norm2_final}"
    )
