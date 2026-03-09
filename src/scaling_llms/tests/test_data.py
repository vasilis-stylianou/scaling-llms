import pytest
import torch
from dataclasses import replace
from pathlib import Path

from scaling_llms.data import DataLoaderConfig, LocalDataPaths, get_dataloaders
from scaling_llms.registries.datasets.identity import DatasetIdentity
from scaling_llms.registries.datasets.registry import DataRegistry
from scaling_llms.storage.base import RegistryStorage


# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture
def local_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "local_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def dataset_id() -> DatasetIdentity:
    return DatasetIdentity(
        dataset_name="super_glue",
        dataset_config="cb",
        train_split="train[:10%]",
        eval_split="test[:10%]",
        tokenizer_name="gpt2_tiktoken",
        text_field="premise",
    )


@pytest.fixture
def dl_config() -> DataLoaderConfig:
    return DataLoaderConfig(
        seq_len=32,
        train_batch_size=8,
        eval_batch_size=8,
        start_sample_idx=0,
        seed=42,
    )


@pytest.fixture
def data_registry(tmp_path: Path) -> DataRegistry:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)

    run_registry_root = project_root / "run_registry"
    runs_artifacts_root = run_registry_root / "artifacts"
    runs_db_path = run_registry_root / "runs.db"

    data_registry_root = project_root / "data_registry"
    datasets_db_path = data_registry_root / "datasets.db"
    tokenized_datasets_root = data_registry_root / "tokenized_datasets"

    storage = RegistryStorage(
        project_root=project_root,
        run_registry_root=run_registry_root,
        runs_db_path=runs_db_path,
        runs_artifacts_root=runs_artifacts_root,
        data_registry_root=data_registry_root,
        datasets_db_path=datasets_db_path,
        datasets_artifacts_root=tokenized_datasets_root,
    )

    return DataRegistry.from_storage(storage)


# ============================================================
# HELPERS
# ============================================================
def _get_mmap_paths(dataset_id: DatasetIdentity, local_data_dir: Path):
    paths = LocalDataPaths(local_data_dir=local_data_dir)
    return paths.train_mmap_path(dataset_id), paths.eval_mmap_path(dataset_id)


# ============================================================
# TESTS
# ============================================================
def test_data_loaders(dataset_id, dl_config, data_registry, local_data_dir):
    dl_dict = get_dataloaders(dataset_id, data_registry, dl_config, local_data_dir=local_data_dir)
    train_dl, eval_dl = dl_dict["train"], dl_dict["eval"]

    # Verify shapes
    xb_train, yb_train = next(iter(train_dl))
    xb_eval, yb_eval = next(iter(eval_dl))
    for tensor, name, expected_shape in [
        (xb_train, "idx_train", (dl_config.train_batch_size, dl_config.seq_len)),
        (yb_train, "targets_train", (dl_config.train_batch_size, dl_config.seq_len)),
        (xb_eval, "idx_eval", (dl_config.eval_batch_size, dl_config.seq_len)),
        (yb_eval, "targets_eval", (dl_config.eval_batch_size, dl_config.seq_len)),
    ]:
        assert tensor.shape == expected_shape, f"Expected {name} shape {expected_shape} but got {tensor.shape}"


def test_dataloader_configs(dataset_id, dl_config, data_registry, local_data_dir):
    """
    Dataloaders with the same HuggingFace dataset configs should reuse
    the same local memmaps and not create new datasets in the Data Registry,
    even if other config parameters (e.g. batch size, seq_len, start_sample_idx)
    are different.
    This tests verifies that the caching mechanism for token memmaps is working correctly
    and that local memmaps are reused when appropriate.
    """

    # Create dataloaders to populate Data Registry and local cache
    _ = get_dataloaders(dataset_id, data_registry, dl_config, local_data_dir=local_data_dir)

    # Check that memmaps exist locally after dataloader creation
    train_mmap, eval_mmap = _get_mmap_paths(dataset_id, local_data_dir)
    assert train_mmap.exists(), f"Expected local train memmap at {train_mmap} but it does not exist."
    assert eval_mmap.exists(), f"Expected local eval memmap at {eval_mmap} but it does not exist."

    # Get file IDs (e.g. inodes) of local memmaps to check for reuse later
    train_file_id = train_mmap.stat().st_ino
    eval_file_id = eval_mmap.stat().st_ino

    # Create dataloaders with various DataLoaderConfig changes and check that local memmaps are reused
    # and no new datasets are created in the Data Registry
    msg2overrides = {
        "same configs": {},
        "different batch size": dict(train_batch_size=16),
        "different seq_len": dict(seq_len=16),
        "different start_sample_idx": dict(start_sample_idx=100),
    }
    for msg, overrides in msg2overrides.items():
        cfg = replace(dl_config, **overrides)
        _ = get_dataloaders(dataset_id, data_registry, cfg, local_data_dir=local_data_dir)

        assert train_mmap.stat().st_ino == train_file_id, "Expected local train memmap to be reused but it was not."
        assert eval_mmap.stat().st_ino == eval_file_id, "Expected local eval memmap to be reused but it was not."
        assert len(data_registry.get_datasets_as_df()) == 1, (
            f"Expected only 1 dataset in Data Registry but found {len(data_registry.get_datasets_as_df())} "
            f"after creating dataloaders with {msg}."
        )


def test_dataset_creation_configs(dataset_id, dl_config, data_registry, local_data_dir):
    """
    Dataloaders with different HuggingFace dataset configs should create new datasets in the Data Registry.
    This tests verifies that the dataset creation mechanism in the Data Registry is working correctly
    and that new datasets are created when appropriate.
    """

    # Create dataloaders to populate Data Registry and local cache
    _ = get_dataloaders(dataset_id, data_registry, dl_config, local_data_dir=local_data_dir)

    # Check that memmaps exist locally after dataloader creation
    train_mmap, eval_mmap = _get_mmap_paths(dataset_id, local_data_dir)
    assert train_mmap.exists(), f"Expected local train memmap at {train_mmap} but it does not exist."
    assert eval_mmap.exists(), f"Expected local eval memmap at {eval_mmap} but it does not exist."

    # Get file IDs (e.g. inodes) of local memmaps to check for reuse later
    train_file_id = train_mmap.stat().st_ino
    eval_file_id = eval_mmap.stat().st_ino

    # Create dataloaders with various DatasetIdentity changes
    # and check that new datasets are created in the Data Registry
    msg2overrides = {
        "different train_split": dict(train_split="train[10%:20%]"),
        "different eval_split": dict(eval_split="test[10%:20%]"),
        "different dataset_config": dict(dataset_config="copa"),
        "different text_field": dict(text_field="hypothesis"),
    }
    num_registered_datasets = len(data_registry.get_datasets_as_df())
    for msg, overrides in msg2overrides.items():
        new_id = replace(dataset_id, **overrides)
        _ = get_dataloaders(new_id, data_registry, dl_config, local_data_dir=local_data_dir)
        new_train_mmap, new_eval_mmap = _get_mmap_paths(new_id, local_data_dir)

        num_registered_datasets += 1
        assert new_train_mmap.stat().st_ino != train_file_id, "Expected local train memmap to be different but it was not."
        assert new_eval_mmap.stat().st_ino != eval_file_id, "Expected local eval memmap to be different but it was not."
        assert len(data_registry.get_datasets_as_df()) == num_registered_datasets, (
            f"Expected a new dataset to be created in the Data Registry but it was not "
            f"after creating dataloaders with {msg}."
        )


def test_dataset_offset(dataset_id, dl_config, data_registry, local_data_dir):

    # Create dataloaders with no offset
    dl_dict = get_dataloaders(dataset_id, data_registry, dl_config, local_data_dir=local_data_dir)
    train_dl1, eval_dl1 = dl_dict["train"], dl_dict["eval"]
    train_iter = iter(train_dl1)
    train_batch_1 = next(train_iter)
    train_batch_2 = next(train_iter)
    eval_batch_1 = next(iter(eval_dl1))

    # Create dataloaders with offset equal to 'train_batch_size' samples
    offset_cfg = replace(dl_config, start_sample_idx=dl_config.train_batch_size)
    dl_dict = get_dataloaders(dataset_id, data_registry, offset_cfg, local_data_dir=local_data_dir)
    train_dl2, eval_dl2 = dl_dict["train"], dl_dict["eval"]
    xb_train_offset1, yb_train_offset1 = next(iter(train_dl2))
    xb_eval_offset1, yb_eval_offset1 = next(iter(eval_dl2))

    # Check that the FIRST batch from the dataloader with offset
    # IS NOT the same as the FIRST batch from the dataloader without offset
    msg = "Expected the first batch from the dataloder with offset to be different from the first batch from the dataloader without offset, but they were the same."
    assert not (torch.equal(train_batch_1[0], xb_train_offset1) and torch.equal(train_batch_1[1], yb_train_offset1)), msg

    # Check that the FIRST batch from the dataloader with offset
    # IS the same as the SECOND batch from the dataloader without offset
    msg = "Expected the first batch from the dataloder with offset to be the same as the second batch from the dataloader without offset, but they were different."
    assert torch.equal(train_batch_2[0], xb_train_offset1) and torch.equal(train_batch_2[1], yb_train_offset1), msg

    # Check that the eval batches are the same since offset should not affect eval dataloader
    msg = "Expected the first batch from the eval dataloder with offset to be the same as the first batch from the eval dataloader without offset, but they were different."
    assert torch.equal(xb_eval_offset1, eval_batch_1[0]) and torch.equal(yb_eval_offset1, eval_batch_1[1]), msg