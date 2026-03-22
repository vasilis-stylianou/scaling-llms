from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import pytest
import torch


from scaling_llms.checkpointing.manager import CheckpointManager
from scaling_llms.constants import DATASET_FILES, METRIC_CATS, TOKENIZED_CACHE_DIR_NAME
from scaling_llms.models import GPTConfig, GPTModel
from scaling_llms.registries import (
    DatasetIdentity,
    DatasetRegistry,
    TokenizedDatasetInfo, 
    RunIdentity,
    RunRegistry,
)
from scaling_llms.registries.datasets.schema import (
    DATASET_IDENTITY_COLS,
    make_datasets_table_spec,
)
from scaling_llms.tracking import METRIC_SCHEMA


@pytest.fixture
def local_tokenized_cache_dir(tmp_path: Path) -> Path:
    path = tmp_path / TOKENIZED_CACHE_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def run_identity() -> RunIdentity:
    return RunIdentity(
        experiment_name="experiment_test_tracking",
        run_name="run_test_tracking",
    )


# ============================================================
# TESTS FOR RUN REGISTRY
# ============================================================
def test_run_registry_paths(
    run_registry: RunRegistry,
    run_identity: RunIdentity,
    test_names: dict[str, str],
) -> None:
    assert run_registry.metadata.table_name == test_names["runs_table"]
    assert run_registry.artifacts.root.exists(), "Artifacts root should be created"

    exp_name = run_identity.experiment_name
    run_name = run_identity.run_name
    identity = RunIdentity(exp_name, run_name)
    run = run_registry.create_run(identity)

    df_runs = run_registry.get_runs_as_df(experiment_name=exp_name)
    row = df_runs.iloc[0]
    assert len(df_runs) == 1, f"Expected 1 run, found {len(df_runs)}"
    assert row.experiment_name == exp_name, (
        f"Expected experiment_name '{exp_name}', got '{row.experiment_name}'"
    )
    assert row.run_name == run_name, f"Expected run_name '{run_name}', got '{row.run_name}'"

    run_abs_path = run_registry.get_artifacts_path(identity)
    assert run_abs_path == run_registry.artifacts.get_absolute_path(row.artifacts_path), (
        "Run absolute path should match artifacts_path resolved by RunArtifacts"
    )
    assert run_abs_path == run.root, (
        f"Run absolute path should match run.root: {run_abs_path} != {run.root}"
    )
    assert run_abs_path.parent.name == exp_name, "Run directory should be nested under experiment dir"

    run_dirs = [run.metadata_dir, run.metrics_dir, run.checkpoints_dir, run.tb_dir]
    for subdir_path in run_dirs:
        assert subdir_path.exists(), f"Run subdirectory should exist: {subdir_path}"
        assert subdir_path.is_dir(), f"Run subdirectory should be a directory: {subdir_path}"
        assert subdir_path.parent == run.root, "Run subdirectory should be directly under run root"


def test_run_registry_delete(
    run_registry: RunRegistry,
    run_identity: RunIdentity,
) -> None:
    identity = RunIdentity(run_identity.experiment_name, run_identity.run_name)
    _ = run_registry.create_run(identity)

    run_registry.delete_run(identity, confirm=False)
    with pytest.raises(FileNotFoundError):
        run_registry.get_artifacts_path(identity)


def test_run_registry_resume(
    run_registry: RunRegistry,
    run_identity: RunIdentity,
) -> None:
    exp_name = run_identity.experiment_name
    run_name = run_identity.run_name

    run1 = run_registry.create_run(RunIdentity(exp_name, run_name))

    with pytest.raises(ValueError):
        run_registry.create_run(RunIdentity(exp_name, run_name), resume=False)

    run2 = run_registry.create_run(RunIdentity(exp_name, run_name), resume=True)

    assert run1.root == run2.root, "Resumed run should have the same root path as the original run"


def test_run_registry_logs_git_commit(
    run_registry: RunRegistry,
    monkeypatch: pytest.MonkeyPatch,
    run_identity: RunIdentity,
) -> None:
    fake_commit = "abc123def456"

    monkeypatch.setattr(
        "scaling_llms.registries.core.metadata.MetadataDB._get_current_git_commit_sha",
        lambda self: fake_commit,
    )

    identity = RunIdentity(run_identity.experiment_name, run_identity.run_name)
    _ = run_registry.create_run(identity)

    assert run_registry.metadata.get_git_commit(identity) == fake_commit


def test_run_registry_get_git_commit(
    run_registry: RunRegistry,
    monkeypatch: pytest.MonkeyPatch,
    run_identity: RunIdentity,
) -> None:
    fake_commit = "deadbeef"

    monkeypatch.setattr(
        "scaling_llms.registries.core.metadata.MetadataDB._get_current_git_commit_sha",
        lambda self: fake_commit,
    )

    identity = RunIdentity(run_identity.experiment_name, run_identity.run_name)
    _ = run_registry.create_run(identity)

    assert run_registry.metadata.get_git_commit(identity) == fake_commit


def test_run_registry_get_git_commit_missing_run_raises(
    run_registry: RunRegistry,
) -> None:
    with pytest.raises(FileNotFoundError):
        run_registry.metadata.get_git_commit(RunIdentity("missing-exp", "missing-run"))


def test_run_logging(
    run_registry: RunRegistry,
    run_identity: RunIdentity,
) -> None:
    run = run_registry.create_run(RunIdentity(run_identity.experiment_name, run_identity.run_name))
    run.start()

    metric_cat = METRIC_CATS.train
    metrics = {"loss": 0.5, "accuracy": 0.8}
    step = 1
    run.log_metrics({metric_cat: metrics}, step=step)

    metadata = {"param1": "value1", "param2": 42}
    metadata_filename = "metadata.json"
    run.log_metadata(metadata, metadata_filename, format="json")

    run.close()

    metrics_path = run.metrics_dir / f"{metric_cat}.jsonl"
    metadata_path = run.metadata_dir / metadata_filename
    assert metrics_path.exists(), "Metrics file should exist"
    assert metadata_path.exists(), "Metadata file should exist"

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

    logged_metadata = json.loads(metadata_path.read_text())
    assert logged_metadata == metadata, "Logged metadata does not match expected metadata"


# ============================================================
# TESTS FOR DATA REGISTRY
# ============================================================
def test_dataset_identity_matches_unique_index() -> None:
    datasets_table = make_datasets_table_spec()
    ident_idx_spec = next(i for i in datasets_table.indexes if i.name == "idx_datasets_identity")

    assert ident_idx_spec.unique is True
    assert tuple(ident_idx_spec.columns) == tuple(DATASET_IDENTITY_COLS)

    ident_field_names = tuple(f.name for f in fields(DatasetIdentity))
    assert ident_field_names == tuple(DATASET_IDENTITY_COLS)


def test_dataset_registry_paths(
    dataset_registry: DatasetRegistry,
    test_names: dict[str, str],
) -> None:
    assert dataset_registry.metadata.table_name == test_names["datasets_table"]
    assert dataset_registry.artifacts.root.exists(), "Datasets artifacts root should be initialized"


def test_dataset_registry_register_find_copy_delete(
    dataset_registry: DatasetRegistry,
    local_tokenized_cache_dir: Path,
) -> None:
    ident = DatasetIdentity(
        dataset_name="dataset_test_1",
        dataset_config="dataset_config_1",
        train_split="train",
        eval_split="test",
        tokenizer_name="tokenizer_test_1",
        text_field="text",
    )

    with pytest.raises(FileNotFoundError):
        dataset_registry.get_dataset_artifacts(ident, raise_if_not_found=True)

    local_dataset_dir = local_tokenized_cache_dir / "test"
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    local_train_mmap_path = local_dataset_dir / DATASET_FILES.train_tokens
    local_eval_mmap_path = local_dataset_dir / DATASET_FILES.eval_tokens

    n_tokens = 1024
    train = np.random.randint(0, 50000, size=n_tokens, dtype=np.uint32)
    eval_ = np.random.randint(0, 50000, size=n_tokens, dtype=np.uint32)

    local_train_mmap_path.write_bytes(train.tobytes())
    local_eval_mmap_path.write_bytes(eval_.tobytes())

    dataset_artifacts = dataset_registry.register_dataset(
        local_train_mmap_path,
        local_eval_mmap_path,
        ident,
    )

    df_datasets = dataset_registry.get_datasets_as_df()
    assert len(df_datasets) == 1, f"Expected 1 dataset, found {len(df_datasets)}"

    found_artifacts = dataset_registry.get_dataset_artifacts(ident)
    assert found_artifacts.root == dataset_artifacts.root, (
        f"Registered dataset path mismatch: {dataset_artifacts.root} != {found_artifacts.root}"
    )
    assert found_artifacts.train_bin.exists(), "Registered train tokens should exist"
    assert found_artifacts.eval_bin.exists(), "Registered eval tokens should exist"

    local_train_mmap_path.unlink()
    local_eval_mmap_path.unlink()
    assert not local_train_mmap_path.exists(), (
        f"Train tokens file should be deleted: {local_train_mmap_path}"
    )
    assert not local_eval_mmap_path.exists(), (
        f"Eval tokens file should be deleted: {local_eval_mmap_path}"
    )

    dataset_registry.delete_dataset(identity=ident, confirm=False)
    df_datasets = dataset_registry.get_datasets_as_df()
    assert len(df_datasets) == 0, f"Expected 0 datasets after deletion, found {len(df_datasets)}"
    assert not dataset_artifacts.root.exists(), (
        f"Dataset path should not exist after deletion: {dataset_artifacts.root}"
    )


def test_dataset_registry_dataset_info(
    dataset_registry: DatasetRegistry,
    local_tokenized_cache_dir: Path,
) -> None:
    ident = DatasetIdentity(
        dataset_name="dataset_info_test",
        dataset_config=None,
        train_split="train",
        eval_split="test",
        tokenizer_name="gpt2_tiktoken",
        text_field="text",
    )

    local_dataset_dir = local_tokenized_cache_dir / "info_test"
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    local_train_mmap = local_dataset_dir / DATASET_FILES.train_tokens
    local_eval_mmap = local_dataset_dir / DATASET_FILES.eval_tokens

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

    dataset_artifacts = dataset_registry.register_dataset(
        local_train_mmap,
        local_eval_mmap,
        ident,
        dataset_info=dataset_info,
    )

    assert dataset_artifacts.dataset_info.exists(), (
        f"Expected dataset_info JSON at {dataset_artifacts.dataset_info} but it does not exist."
    )

    loaded_info = dataset_registry.get_dataset_info(ident)
    assert loaded_info is not None, "get_dataset_info returned None for a registered dataset."
    assert isinstance(loaded_info, TokenizedDatasetInfo), (
        f"Expected TokenizedDatasetInfo, got {type(loaded_info)}"
    )
    assert loaded_info == dataset_info, (
        f"Loaded dataset info does not match original: {loaded_info} != {dataset_info}"
    )

    unknown_ident = DatasetIdentity(
        dataset_name="nonexistent",
        dataset_config=None,
        train_split="train",
        eval_split="test",
        tokenizer_name="gpt2_tiktoken",
        text_field="text",
    )
    assert dataset_registry.get_dataset_info(unknown_ident, raise_if_not_found=False) is None, (
        "get_dataset_info should return None for an unregistered dataset."
    )


# ============================================================
# TESTS FOR CHECKPOINT MANAGER
# ============================================================
def _model_param_norm(model: GPTModel) -> torch.Tensor:
    return torch.sqrt(sum(p.float().norm() ** 2 for p in model.parameters()))


def test_checkpoint_manager_save_load(
    run_registry: RunRegistry,
    run_identity: RunIdentity,
) -> None:
    run = run_registry.create_run(run_identity, resume=False)

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

    torch.manual_seed(2)
    model2 = GPTModel(GPTConfig())
    norm1 = _model_param_norm(model1)
    norm2 = _model_param_norm(model2)
    assert not torch.allclose(norm1, norm2), (
        f"Model parameters should differ before loading checkpoint: {norm1} ≈ {norm2}"
    )

    ckpt_manager = CheckpointManager(run.checkpoints_dir, model2)
    loaded_trainer_state = ckpt_manager.load(ckpt_path)

    assert loaded_trainer_state == trainer_state, (
        f"Loaded trainer state should match saved state: {loaded_trainer_state} != {trainer_state}"
    )

    norm1_final = _model_param_norm(model1)
    norm2_final = _model_param_norm(model2)
    assert torch.allclose(norm1_final, norm2_final), (
        f"Model parameters should match after loading checkpoint: {norm1_final} != {norm2_final}"
    )