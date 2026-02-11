import pytest
import shutil
import torch
from pathlib import Path

from scaling_llms.constants import LOCAL_DEV_DATA_DIR, PROJECT_DEV_NAME
from scaling_llms.data import DataConfig, get_dataloaders
from scaling_llms.tracking.registries import GoogleDriveDataRegistry

@pytest.fixture
def base_configs():
    return dict(
        dataset_name="wikitext",
        dataset_config="wikitext-103-v1",
        seq_len=512,
        train_batch_size=8,
        eval_batch_size=8,
        local_data_dir=LOCAL_DEV_DATA_DIR,
        train_split="train[:1000]",
        eval_split="test[:1000]",
        train_tokens_budget=1_000_000,
        start_sample_idx=0,
    )

@pytest.fixture
def gdrive_overrides():
    return dict(project_subdir=PROJECT_DEV_NAME)

@pytest.fixture
def dev_data_registry():
    return GoogleDriveDataRegistry(project_subdir=PROJECT_DEV_NAME)

@pytest.fixture(autouse=True)
def clean_data_registry_and_cache(dev_data_registry, base_configs):
    def _clean():
        df = dev_data_registry.get_datasets_as_df()
        for _, row in df.iterrows():
            dev_data_registry.delete_dataset(dataset_path=row.dataset_path, confirm=False)
        for item in Path(base_configs["local_data_dir"]).iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)

    # Clean up before each unit test
    _clean()

    yield

    # Clean up after each unit test
    _clean()


def test_data_loaders(base_configs, gdrive_overrides):
    cfg = DataConfig(**base_configs)
    train_dl, eval_dl = get_dataloaders(cfg, **gdrive_overrides)

    # Verify number of training tokens
    expected_num_train_tokens = len(train_dl.dataset) * cfg.seq_len
    actual_num_train_tokens = sum(b[0].numel() for b in train_dl) 
    msg = f"Expected {expected_num_train_tokens:,} train tokens but got {actual_num_train_tokens:,}."
    assert expected_num_train_tokens == actual_num_train_tokens, msg

    msg = f"Expected at most {cfg.train_tokens_budget:,} train tokens but got {actual_num_train_tokens:,}."
    assert actual_num_train_tokens <= cfg.train_tokens_budget, msg

    # Verify shapes
    xb_train, yb_train = next(iter(train_dl))
    xb_eval, yb_eval = next(iter(eval_dl))
    for tensor, name, expected_shape in [
        (xb_train, "idx_train", (cfg.train_batch_size, cfg.seq_len)), 
        (yb_train, "targets_train", (cfg.train_batch_size, cfg.seq_len)), 
        (xb_eval, "idx_eval", (cfg.eval_batch_size, cfg.seq_len)), 
        (yb_eval, "targets_eval", (cfg.eval_batch_size, cfg.seq_len))
    ]:
        assert tensor.shape == expected_shape, f"Expected {name} shape {expected_shape} but got {tensor.shape}"


def test_dataloader_configs(base_configs, gdrive_overrides, dev_data_registry):
    """
    Dataloaders with the same HuggingFace dataset configs should reuse 
    the same local memmaps and not create new datasets in the Data Registry, 
    even if other config parameters (e.g. batch size, seq_len, train_tokens_budget, start_sample_idx) 
    are different. 
    This tests verifies that the caching mechanism for token memmaps is working correctly 
    and that local memmaps are reused when appropriate.
    """

    BASE_CONFIGS = base_configs
    data_registry = dev_data_registry

    # Create dataloaders to populate Data Registry and local cache
    cfg = DataConfig(**BASE_CONFIGS)
    _ = get_dataloaders(cfg, **gdrive_overrides)

    # Check that memmaps exist locally after dataloader creation
    assert cfg.local_train_mmap_path.exists(), f"Expected local train memmap at {cfg.local_train_mmap_path} but it does not exist."
    assert cfg.local_eval_mmap_path.exists(), f"Expected local eval memmap at {cfg.local_eval_mmap_path} but it does not exist."

    # Get file IDs (e.g. inodes) of local memmaps to check for reuse later
    train_file_id = cfg.local_train_mmap_path.stat().st_ino
    eval_file_id = cfg.local_eval_mmap_path.stat().st_ino

    # Create dataloaders with various config changes and check that local memmaps are reused (i.e. file IDs are the same) 
    # and no new datasets are created in the Data Registry
    msg2diff_configs = {
        "same configs": {},
        "different batch size": dict(train_batch_size=16),
        "different seq_len": dict(seq_len=256),
        "different train_tokens_budget": dict(train_tokens_budget=500_000),
        "different start_sample_idx": dict(start_sample_idx=100),
    }
    for msg,diff_configs in msg2diff_configs.items():
        cfg = DataConfig(**{**BASE_CONFIGS, **diff_configs})
        _ = get_dataloaders(cfg, **gdrive_overrides)
        train_file_id_2 = cfg.local_train_mmap_path.stat().st_ino
        eval_file_id_2 = cfg.local_eval_mmap_path.stat().st_ino

        assert train_file_id == train_file_id_2, "Expected local train memmap to be reused but it was not."
        assert eval_file_id == eval_file_id_2, "Expected local eval memmap to be reused but it was not."
        assert len(data_registry.get_datasets_as_df()) == 1, f"Expected only 1 dataset in Data Registry but found {len(data_registry.get_datasets_as_df())} after creating dataloaders with {msg}."


def test_dataset_creation_configs(base_configs, gdrive_overrides, dev_data_registry):
    """
    Dataloaders with different HuggingFace dataset configs should create new datasets in the Data Registry. 
    This tests verifies that the dataset creation mechanism in the Data Registry is working correctly 
    and that new datasets are created when appropriate.
    """

    BASE_CONFIGS = base_configs
    GDRIVE_OVERRIDES = gdrive_overrides
    data_registry = dev_data_registry

    # Create dataloaders to populate Data Registry and local cache
    cfg = DataConfig(**BASE_CONFIGS)
    _ = get_dataloaders(cfg, **GDRIVE_OVERRIDES)
    
    # Check that memmaps exist locally after dataloader creation
    assert cfg.local_train_mmap_path.exists(), f"Expected local train memmap at {cfg.local_train_mmap_path} but it does not exist."
    assert cfg.local_eval_mmap_path.exists(), f"Expected local eval memmap at {cfg.local_eval_mmap_path} but it does not exist."

    # Get file IDs (e.g. inodes) of local memmaps to check for reuse later
    train_file_id = cfg.local_train_mmap_path.stat().st_ino
    eval_file_id = cfg.local_eval_mmap_path.stat().st_ino


    # Create dataloaders with various HuggingFace dataset config changes 
    # and check that new datasets are created in the Data Registry
    msg2diff_configs = {
        "different train_split": dict(train_split="train[1000:2000]"),
        "different eval_split": dict(eval_split="test[1000:2000]"),
        "different dataset_config": dict(dataset_config="wikitext-2-v1"),
    }
    num_registered_datasets = len(data_registry.get_datasets_as_df())
    for msg,diff_configs in msg2diff_configs.items():
        cfg = DataConfig(**{**BASE_CONFIGS, **diff_configs})
        _ = get_dataloaders(cfg, **GDRIVE_OVERRIDES)
        train_file_id_2 = cfg.local_train_mmap_path.stat().st_ino
        eval_file_id_2 = cfg.local_eval_mmap_path.stat().st_ino

        num_registered_datasets += 1
        assert train_file_id != train_file_id_2, "Expected local train memmap to be different but it was not."
        assert eval_file_id != eval_file_id_2, "Expected local eval memmap to be different but it was not."
        assert len(data_registry.get_datasets_as_df()) == num_registered_datasets, f"Expected a new dataset to be created in the Data Registry but it was not after creating dataloaders with {msg}."

    
def test_dataset_offset(base_configs, gdrive_overrides, dev_data_registry):

    BASE_CONFIGS = base_configs
    GDRIVE_OVERRIDES = gdrive_overrides

    # Create dataloaders with no offset
    cfg1 = DataConfig(**BASE_CONFIGS)
    train_dl1, eval_dl1 = get_dataloaders(cfg1, **GDRIVE_OVERRIDES)
    train_iter = iter(train_dl1)
    train_batch_1 = next(train_iter)
    train_batch_2 = next(train_iter)
    eval_batch_1 = next(iter(eval_dl1))
    
    # Create dataloaders with offset equal to 'train_batch_size' samples
    cfg2 = DataConfig(**{**BASE_CONFIGS, "start_sample_idx": BASE_CONFIGS["train_batch_size"]})
    train_dl2, eval_dl2 = get_dataloaders(cfg2, **GDRIVE_OVERRIDES)
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