from dataclasses import fields
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from scaling_llms.constants import (
    DATA_FILES,
    METRIC_CATS,
    TOKENIZED_CACHE_DIR_NAME,
)
from scaling_llms.models import GPTConfig, GPTModel
from scaling_llms.registries.datasets.artifacts import DatasetArtifacts, TokenizedDatasetInfo
from scaling_llms.registries.datasets.identity import DATASET_IDENTITY_COLS, DatasetIdentity
from scaling_llms.registries.datasets.schema import DATASETS_TABLE
from scaling_llms.registries.runs.registry import RunRegistry
from scaling_llms.registries.runs.identity import RunIdentity
from scaling_llms.registries.datasets.registry import DataRegistry
from scaling_llms.tracking.trackers import METRIC_SCHEMA
from scaling_llms.checkpointing.manager import CheckpointManager
from scaling_llms.storage.base import RegistryStorage


EXPERIMENT_NAME = "test_tracking"
RUN_NAME = "run_test_tracking"

# ============================================================
# FIXTURES FOR REGISTRIES AND CHECKPOINT MANAGER TESTS
# ============================================================
@pytest.fixture
def storage(tmp_path: Path):

    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)

    run_registry_root = project_root / "run_registry"
    runs_artifacts_root = run_registry_root / "artifacts"
    runs_db_path = run_registry_root / "runs.db"

    data_registry_root = project_root / "data_registry"
    datasets_db_path = data_registry_root / "datasets.db"
    datasets_artifacts_root = data_registry_root / "tokenized_datasets"

    storage = RegistryStorage(
        project_root=project_root,
        run_registry_root=run_registry_root,
        runs_db_path=runs_db_path,
        runs_artifacts_root=runs_artifacts_root,
        data_registry_root=data_registry_root,
        datasets_db_path=datasets_db_path,
        datasets_artifacts_root=datasets_artifacts_root,
    )
    return storage

@pytest.fixture
def run_registry(storage):
    return RunRegistry.from_storage(storage)

@pytest.fixture
def data_registry(storage):
    return DataRegistry.from_storage(storage)

@pytest.fixture
def local_tokenized_cache_dir(tmp_path: Path):
    path = tmp_path / TOKENIZED_CACHE_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# TESTS FOR RUN REGISTRY
# ============================================================
def test_run_registry_paths(run_registry):

    # Check registry paths are correctly set up
    assert run_registry.root.parent == run_registry.storage.project_root, (
        "Registry root should be a direct child of project root"
    )
    assert run_registry.db_path.parent == run_registry.root, (
        "Database path should be inside registry root"
    )
    assert run_registry.artifacts_root.parent == run_registry.root, (
        "Artifacts root should be inside registry root"
    )

    # Create a run
    exp_name = EXPERIMENT_NAME
    run_name = RUN_NAME
    run = run_registry.create_run(RunIdentity(exp_name, run_name))

    # Check the run is registered correctly
    df_runs =  run_registry.get_runs_as_df().query(f"experiment_name == '{exp_name}'")
    row = df_runs.iloc[0]
    assert len(df_runs) == 1, f"Expected 1 run, found {len(df_runs)}"
    assert row.experiment_name == exp_name, (
        f"Expected experiment_name '{exp_name}', got '{row.experiment_name}'"
    )
    assert row.run_name == run_name, f"Expected run_name '{run_name}', got '{row.run_name}'"

    # Check the run paths are correct
    exp_dir = run_registry.get_experiment_dir(row.experiment_name)
    run_dir = run_registry.get_run_dir(RunIdentity(row.experiment_name, row.run_name))
    run_abs_path = Path(row.run_absolute_path)

    assert run_abs_path == run_registry.artifacts_root / row.artifacts_path, (
        f"Run absolute path mismatch: {run_abs_path} != {run_registry.artifacts_root / row.artifacts_path}"
    )
    assert run_abs_path.parent == exp_dir, (
        f"Run directory parent should be experiment directory: {run_abs_path.parent} != {exp_dir}"
    )
    assert run_abs_path == run_dir, f"Run absolute path should match run_dir: {run_abs_path} != {run_dir}"
    assert run_abs_path == run.root, f"Run absolute path should match run.root: {run_abs_path} != {run.root}"

    # Check the run subdirectories are correctly set up
    for subdir_name in run.artifacts.list_subdir_names():
        subdir_path = run_dir / subdir_name
        attr_name = f"{subdir_name}_dir"
        assert subdir_path.exists(), f"Run subdirectory {subdir_name} should exist"
        assert subdir_path.is_dir(), f"Run subdirectory {subdir_name} should be a directory"
        assert subdir_path == getattr(run, attr_name), f"Run subdirectory must be accessible by run.{attr_name}"

def test_run_registry_delete(run_registry):
    exp_name = EXPERIMENT_NAME
    run_name = RUN_NAME
    _ = run_registry.create_run(RunIdentity(exp_name, run_name))

    # Delete the run and check it's removed
    run_registry.delete_run(RunIdentity(exp_name, run_name), confirm=False)
    with pytest.raises(FileNotFoundError):
        run_registry.get_run_dir(RunIdentity(exp_name, run_name))

    # Delete the experiment and check it's removed
    run_registry.delete_experiment(exp_name, confirm=False)
    with pytest.raises(FileNotFoundError):
        run_registry.get_experiment_dir(exp_name)


def test_run_registry_resume(run_registry):
    exp_name = EXPERIMENT_NAME
    run_name = RUN_NAME
    run1 = run_registry.create_run(RunIdentity(exp_name, run_name))

    with pytest.raises(ValueError):
        run_registry.create_run(RunIdentity(exp_name, run_name), resume=False)

    run2 = run_registry.create_run(RunIdentity(exp_name, run_name), resume=True)

    assert run1.root == run2.root, (
        "Resumed run should have the same root path as the original run"
    )


def test_run_logging(run_registry):
    exp_name = EXPERIMENT_NAME
    run_name = RUN_NAME
    run = run_registry.create_run(RunIdentity(exp_name, run_name))
    run.start()

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
    metrics_path = run.metrics_dir / f"{metric_cat}.jsonl"
    metadata_path = run.metadata_dir / metadata_filename
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
def test_dataset_identity_matches_unique_index():
    ident_idx_spec = next(i for i in DATASETS_TABLE.indexes if i.name == "idx_datasets_identity")

    assert ident_idx_spec.unique is True
    assert tuple(ident_idx_spec.columns) == tuple(DATASET_IDENTITY_COLS)

    ident_field_names = tuple(f.name for f in fields(DatasetIdentity))
    assert ident_field_names == tuple(DATASET_IDENTITY_COLS)


def test_data_registry_paths(data_registry):

    assert data_registry.root.parent == data_registry.storage.project_root, (
        "Registry root should be a direct child of project root"
    )
    assert data_registry.db_path.parent == data_registry.root, (
        "Database path should be inside registry root"
    )
    assert data_registry.artifacts_root.parent == data_registry.root, (
        "Datasets artifacts root should be inside registry root"
    )


def test_data_registry_register_find_copy_delete(
    data_registry, 
    local_tokenized_cache_dir
):
    # Create a dummy dataset identity and check that it's not found in the registry
    ident = DatasetIdentity(
        dataset_name="dataset_test_1",
        dataset_config="dataset_config_1",
        train_split="train",
        eval_split="test",
        tokenizer_name="tokenizer_test_1",
        text_field="text",
    )

    with pytest.raises(FileNotFoundError):
        data_registry.find_dataset_path(ident, raise_if_not_found=True)

    # Create dummy tokenized dataset files in the local tokenized cache dir
    local_dataset_dir = local_tokenized_cache_dir / "test"
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    local_train_mmap_path = local_dataset_dir / DATA_FILES.train_tokens
    local_eval_mmap_path = local_dataset_dir / DATA_FILES.eval_tokens

    n_tokens = 1024
    train = np.random.randint(0, 50000, size=n_tokens, dtype=np.uint32)
    eval_ = np.random.randint(0, 50000, size=n_tokens, dtype=np.uint32)

    local_train_mmap_path.write_bytes(train.tobytes())
    local_eval_mmap_path.write_bytes(eval_.tobytes())

    # Register the dataset
    dataset_path = data_registry.register_dataset(
        local_train_mmap_path,
        local_eval_mmap_path,
        ident,
    )

    # Check the dataset is registered correctly
    df_datasets = data_registry.get_datasets_as_df()
    assert len(df_datasets) == 1, f"Expected 1 dataset, found {len(df_datasets)}"
    found_path = data_registry.find_dataset_path(ident)
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
    data_registry.copy_dataset_to_local(dataset_path, local_dataset_dir)
    assert local_train_mmap_path.exists(), (
        f"Train tokens file should exist after copy: {local_train_mmap_path}"
    )
    assert local_eval_mmap_path.exists(), (
        f"Eval tokens file should exist after copy: {local_eval_mmap_path}"
    )

    # Delete the dataset and check it's removed
    data_registry.delete_dataset(dataset_path=dataset_path, confirm=False)
    df_datasets = data_registry.get_datasets_as_df()
    assert len(df_datasets) == 0, f"Expected 0 datasets after deletion, found {len(df_datasets)}"
    assert not dataset_path.exists(), f"Dataset path should not exist after deletion: {dataset_path}"


def test_data_registry_dataset_info(data_registry, local_tokenized_cache_dir):
    """
    register_dataset with a TokenizedDatasetInfo should persist a JSON file
    in the dataset artifacts directory, and get_dataset_info should return
    the same data back.
    """
    ident = DatasetIdentity(
        dataset_name="dataset_info_test",
        dataset_config=None,
        train_split="train",
        eval_split="test",
        tokenizer_name="gpt2_tiktoken",
        text_field="text",
    )

    # Create dummy tokenized dataset files
    local_dataset_dir = local_tokenized_cache_dir / "info_test"
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    local_train_mmap = local_dataset_dir / DATA_FILES.train_tokens
    local_eval_mmap = local_dataset_dir / DATA_FILES.eval_tokens

    n_tokens = 512
    local_train_mmap.write_bytes(
        np.random.randint(0, 50000, size=n_tokens, dtype=np.uint16).tobytes()
    )
    local_eval_mmap.write_bytes(
        np.random.randint(0, 50000, size=n_tokens, dtype=np.uint16).tobytes()
    )

    dataset_info = TokenizedDatasetInfo(
        vocab_size=50257,
        eos_id=50256,
        dtype="uint16",
        total_train_tokens=n_tokens,
        total_eval_tokens=n_tokens,
    )

    # Register with dataset_info
    dataset_path = data_registry.register_dataset(
        local_train_mmap,
        local_eval_mmap,
        ident,
        dataset_info=dataset_info,
    )

    # The JSON file should exist inside the registered artifact directory
    artifacts = DatasetArtifacts(dataset_path)
    assert artifacts.dataset_info.exists(), (
        f"Expected dataset_info JSON at {artifacts.dataset_info} but it does not exist."
    )

    # get_dataset_info should return a TokenizedDatasetInfo matching the original
    loaded_info = data_registry.get_dataset_info(ident)
    assert loaded_info is not None, "get_dataset_info returned None for a registered dataset."
    assert isinstance(loaded_info, TokenizedDatasetInfo), (
        f"Expected TokenizedDatasetInfo, got {type(loaded_info)}"
    )
    assert loaded_info == dataset_info, (
        f"Loaded dataset info does not match original: {loaded_info} != {dataset_info}"
    )

    # get_dataset_info for an unknown identity should return None
    unknown_ident = DatasetIdentity(
        dataset_name="nonexistent",
        dataset_config=None,
        train_split="train",
        eval_split="test",
        tokenizer_name="gpt2_tiktoken",
        text_field="text",
    )
    assert data_registry.get_dataset_info(unknown_ident) is None, (
        "get_dataset_info should return None for an unregistered dataset."
    )


# ============================================================
# TESTS FOR CHECKPOINT MANAGER
# ============================================================
def _model_param_norm(model: GPTModel) -> torch.Tensor:
    return torch.sqrt(sum(p.float().norm() ** 2 for p in model.parameters()))


def test_checkpoint_manager_save_load(run_registry):
    # Create a run to store checkpoints
    run = run_registry.create_run(RunIdentity(EXPERIMENT_NAME, RUN_NAME), resume=False)

    # Create a model and save a checkpoint
    torch.manual_seed(1)
    model1 = GPTModel(GPTConfig())
    ckpt_manager = CheckpointManager(
        run.checkpoints_dir,
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
    ckpt_manager = CheckpointManager(run.checkpoints_dir, model2)
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
