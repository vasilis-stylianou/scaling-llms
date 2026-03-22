import pytest
import torch
import numpy as np
import shutil
from dataclasses import replace
from pathlib import Path

import scaling_llms.data as data_module
from scaling_llms.data import DataLoaderConfig, LocalCachePaths, get_dataloaders
from scaling_llms.constants import DATASET_FILES
from scaling_llms.registries import (
    DatasetRegistry,
    DatasetArtifactsDir,
    TokenizedDatasetInfo,
    DatasetIdentity,
)


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


# ============================================================
# HELPERS
# ============================================================
def _get_mmap_paths(dataset_id: DatasetIdentity, local_data_dir: Path):
    paths = LocalCachePaths(cache_dir=local_data_dir)
    return paths.train_mmap_path(dataset_id), paths.eval_mmap_path(dataset_id)


def _write_dummy_token_mmaps(
    train_path: Path,
    eval_path: Path,
    *,
    dtype: np.dtype = np.dtype(np.uint16),
    n_train_tokens: int = 512,
    n_eval_tokens: int = 256,
) -> None:
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    train_arr = np.arange(n_train_tokens, dtype=dtype)
    eval_arr = np.arange(n_eval_tokens, dtype=dtype)
    train_arr.tofile(train_path)
    eval_arr.tofile(eval_path)


# ============================================================
# TESTS
# ============================================================
def test_data_loaders(dataset_id, dl_config, dataset_registry, local_data_dir):
    dl_dict = get_dataloaders(dataset_id, dataset_registry, dl_config, cache_dir=local_data_dir)
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


def test_dataloader_configs(dataset_id, dl_config, dataset_registry, local_data_dir):
    """
    Dataloaders with the same HuggingFace dataset configs should reuse
    the same local memmaps and not create new datasets in the Data Registry,
    even if other config parameters (e.g. batch size, seq_len, start_sample_idx)
    are different.
    This tests verifies that the caching mechanism for token memmaps is working correctly
    and that local memmaps are reused when appropriate.
    """

    # Create dataloaders to populate Data Registry and local cache
    _ = get_dataloaders(dataset_id, dataset_registry, dl_config, cache_dir=local_data_dir)

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
        _ = get_dataloaders(dataset_id, dataset_registry, cfg, cache_dir=local_data_dir)

        assert train_mmap.stat().st_ino == train_file_id, "Expected local train memmap to be reused but it was not."
        assert eval_mmap.stat().st_ino == eval_file_id, "Expected local eval memmap to be reused but it was not."
        assert len(dataset_registry.get_datasets_as_df()) == 1, (
            f"Expected only 1 dataset in Data Registry but found {len(dataset_registry.get_datasets_as_df())} "
            f"after creating dataloaders with {msg}."
        )


def test_dataset_creation_configs(dataset_id, dl_config, dataset_registry, local_data_dir):
    """
    Dataloaders with different HuggingFace dataset configs should create new datasets in the Data Registry.
    This tests verifies that the dataset creation mechanism in the Data Registry is working correctly
    and that new datasets are created when appropriate.
    """

    # Create dataloaders to populate Data Registry and local cache
    _ = get_dataloaders(dataset_id, dataset_registry, dl_config, cache_dir=local_data_dir)

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
    num_registered_datasets = len(dataset_registry.get_datasets_as_df())
    for msg, overrides in msg2overrides.items():
        new_id = replace(dataset_id, **overrides)
        _ = get_dataloaders(new_id, dataset_registry, dl_config, cache_dir=local_data_dir)
        new_train_mmap, new_eval_mmap = _get_mmap_paths(new_id, local_data_dir)

        num_registered_datasets += 1
        assert new_train_mmap.stat().st_ino != train_file_id, "Expected local train memmap to be different but it was not."
        assert new_eval_mmap.stat().st_ino != eval_file_id, "Expected local eval memmap to be different but it was not."
        assert len(dataset_registry.get_datasets_as_df()) == num_registered_datasets, (
            f"Expected a new dataset to be created in the Data Registry but it was not "
            f"after creating dataloaders with {msg}."
        )


def test_dataset_offset(dataset_id, dl_config, dataset_registry, local_data_dir):

    # Create dataloaders with no offset
    dl_dict = get_dataloaders(dataset_id, dataset_registry, dl_config, cache_dir=local_data_dir)
    train_dl1, eval_dl1 = dl_dict["train"], dl_dict["eval"]
    train_iter = iter(train_dl1)
    train_batch_1 = next(train_iter)
    train_batch_2 = next(train_iter)
    eval_batch_1 = next(iter(eval_dl1))

    # Create dataloaders with offset equal to 'train_batch_size' samples
    offset_cfg = replace(dl_config, start_sample_idx=dl_config.train_batch_size)
    dl_dict = get_dataloaders(dataset_id, dataset_registry, offset_cfg, cache_dir=local_data_dir)
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


def test_get_dataloaders_hydrates_from_remote_when_local_missing(
    dataset_id,
    dl_config,
    dataset_registry_with_sync_hooks,
    local_data_dir,
    tmp_path: Path,
):
    registry: DatasetRegistry = dataset_registry_with_sync_hooks["registry"]  # type: ignore[assignment]
    remote_root: Path = dataset_registry_with_sync_hooks["remote_root"]  # type: ignore[assignment]
    prepare_calls: list[DatasetIdentity] = dataset_registry_with_sync_hooks["prepare_calls"]  # type: ignore[assignment]

    src_train = tmp_path / "src" / DATASET_FILES.train_tokens
    src_eval = tmp_path / "src" / DATASET_FILES.eval_tokens
    _write_dummy_token_mmaps(src_train, src_eval)

    dataset_info = TokenizedDatasetInfo(
        vocab_size=50257,
        eos_id=50256,
        dtype="uint16",
        total_train_tokens=512,
        total_eval_tokens=256,
    )
    dataset_artifacts = registry.register_dataset(
        src_path_train_bin=src_train,
        src_path_eval_bin=src_eval,
        identity=dataset_id,
        dataset_info=dataset_info,
        vocab_size=dataset_info.vocab_size,
        total_train_tokens=dataset_info.total_train_tokens,
        total_eval_tokens=dataset_info.total_eval_tokens,
    )

    rel = dataset_artifacts.root.relative_to(registry.artifacts.root)
    remote_dataset_path = remote_root / rel
    assert remote_dataset_path.exists(), "Expected registered dataset to be synced to remote artifacts store."

    shutil.rmtree(dataset_artifacts.root)
    assert not dataset_artifacts.root.exists(), "Expected local dataset artifacts to be removed for hydration test."

    dl_dict = get_dataloaders(dataset_id, registry, dl_config, cache_dir=local_data_dir)
    assert dl_dict["train"] is not None and dl_dict["eval"] is not None
    assert dataset_artifacts.root.exists(), "Expected dataset artifacts to be hydrated locally from remote."
    assert (dataset_artifacts.root / DATASET_FILES.train_tokens).exists()
    assert (dataset_artifacts.root / DATASET_FILES.eval_tokens).exists()
    assert len(prepare_calls) >= 1, "Expected prepare hook to run for dataset hydration."


def test_get_dataloaders_creates_registers_and_syncs_when_dataset_missing(
    dataset_id,
    dl_config,
    dataset_registry_with_sync_hooks,
    local_data_dir,
    monkeypatch,
):
    registry: DatasetRegistry = dataset_registry_with_sync_hooks["registry"]  # type: ignore[assignment]
    remote_root: Path = dataset_registry_with_sync_hooks["remote_root"]  # type: ignore[assignment]
    sync_calls: list[DatasetIdentity] = dataset_registry_with_sync_hooks["sync_calls"]  # type: ignore[assignment]

    created_calls = {"count": 0}

    def _fake_make_tokenized_dataset(
        dataset_id: DatasetIdentity,
        dataset_registry: DatasetRegistry,
        local_train_mmap_path: Path,
        local_eval_mmap_path: Path,
        local_hf_cache_dir: Path,
        local_tokenized_cache_dir: Path,
    ) -> DatasetArtifactsDir:
        created_calls["count"] += 1
        _write_dummy_token_mmaps(local_train_mmap_path, local_eval_mmap_path)

        dataset_info = TokenizedDatasetInfo(
            vocab_size=50257,
            eos_id=50256,
            dtype="uint16",
            total_train_tokens=512,
            total_eval_tokens=256,
        )
        return dataset_registry.register_dataset(
            src_path_train_bin=local_train_mmap_path,
            src_path_eval_bin=local_eval_mmap_path,
            identity=dataset_id,
            dataset_info=dataset_info,
            vocab_size=dataset_info.vocab_size,
            total_train_tokens=dataset_info.total_train_tokens,
            total_eval_tokens=dataset_info.total_eval_tokens,
        )

    monkeypatch.setattr(data_module, "make_tokenized_dataset", _fake_make_tokenized_dataset)

    assert not registry.dataset_exists(dataset_id)
    dl_dict = get_dataloaders(dataset_id, registry, dl_config, cache_dir=local_data_dir)

    assert dl_dict["train"] is not None and dl_dict["eval"] is not None
    assert created_calls["count"] == 1, "Expected dataset creation path to run exactly once."
    assert registry.dataset_exists(dataset_id), "Expected created dataset to be registered in metadata DB."
    assert len(sync_calls) >= 1, "Expected created dataset to be synced to remote artifacts store."

    local_artifacts = registry.get_dataset_artifacts(dataset_id, raise_if_not_found=True)
    local_dataset_path = local_artifacts.root
    rel = local_dataset_path.relative_to(registry.artifacts.root)
    remote_dataset_path = remote_root / rel
    assert remote_dataset_path.exists(), "Expected remotely synced dataset directory to exist."
    assert (remote_dataset_path / DATASET_FILES.train_tokens).exists()
    assert (remote_dataset_path / DATASET_FILES.eval_tokens).exists()