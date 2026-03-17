from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import logging


import numpy as np
import tiktoken
import torch
from datasets import load_dataset, load_dataset_builder
import shutil
from torch.utils.data import DataLoader, Dataset

from scaling_llms.constants import (
    DATASET_FILES,
    LOCAL_DATA_DIR,
    HF_CACHE_DIR_NAME,
    MAX_CACHE_GB,
    METADATA_FILES,
    TOKENIZED_CACHE_DIR_NAME,
)
from scaling_llms.registries.datasets.artifacts import TokenizedDatasetInfo
from scaling_llms.registries.datasets.identity import DatasetIdentity
from scaling_llms.registries.datasets.registry import DataRegistry
from scaling_llms.tracking.run import Run
from scaling_llms.utils.config import BaseJsonConfig
from scaling_llms.utils.loggers import DataLogger


# ============================================================
# DETERMINISM (single GPU)
# ============================================================
def ensure_local_dataset_cache_cap(
    cache_dir: Path,
    dataset_name: str,
    dataset_config: str | None,
    cap_size_gb: float,
) -> None:
    """
    If (current_cache_size + expected_dataset_size) > cap_size_gb,
    empties the local dataset directory before loading.

    Returns the cache dir Path.
    """
    CAP_BYTES = int(cap_size_gb * 1024**3)

    # Resolve local data dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Get expected dataset size from HF metadata ----
    builder = load_dataset_builder(dataset_name, dataset_config)
    expected_size = (
        builder.info.dataset_size
        or builder.info.size_in_bytes
        or builder.info.download_size
        or 0
    )   
    if expected_size > CAP_BYTES:
        raise ValueError(
            f"Expected dataset size {expected_size / 1024**3:.2f} GB "
            f"already exceeds cap of {cap_size_gb} GB, cannot proceed with loading dataset. "
            f"Consider increasing cap_size_gb or using a smaller dataset."
        )

    # ---- 2. Compute current cache size ----
    current_cache_size = 0
    for p in cache_dir.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                current_cache_size += p.stat().st_size
        except FileNotFoundError:
            pass
        
    # ---- 3. Decide whether to wipe data ----
    if current_cache_size + expected_size > CAP_BYTES:
        for p in cache_dir.iterdir():
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    return

# ============================================================
# DETERMINISM (single GPU)
# ============================================================
def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Strict determinism (can throw if you use nondeterministic ops)
    # Comment out if it blocks you during early bring-up.
    torch.use_deterministic_algorithms(True)


# ============================================================
# TOKENIZER (tiktoken preferred)
# ============================================================
def load_tokenizer(name: str):
    
    if name == "gpt2_tiktoken":
        enc = tiktoken.get_encoding("gpt2")
        def encode_fn(text: str) -> list[int]:
            return enc.encode(text)
        
        def decode_fn(ids: list[int]) -> str:
            return enc.decode(ids)
        
        return {
            "encode": encode_fn,
            "decode": decode_fn,
            "eot_token": enc.eot_token,
            "vocab_size": enc.n_vocab,
            "raw": enc,
        }

    else:
        raise ValueError(f"Unsupported tokenizer library: {name}")


# ============================================================
# MEMORY MAP BUILDER (tokenize once, reuse forever)
# ============================================================
def build_memmap_tokens(
    token_mmap_path: str | Path,
    texts: list[str],
    encode_fn,
    eos_id: int,
    dtype: np.dtype,
    append_eos: bool,
) -> int:
    """
    Tokenize texts once into a contiguous, file-backed token buffer for
    efficient reading.

    Uses a NumPy memmap to write tokens to disk so subsequent reads can
    memory-map the file and access slices on demand without loading the
    full array into RAM. This enables fast, random window access during
    training while keeping memory usage low.
    """
    token_mmap_path = Path(token_mmap_path).expanduser().resolve()
    token_mmap_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary file during construction so interrupted runs leave no
    # partially-built memmap at the final path. The tmp path uses the final
    # suffix + '.tmp' to mirror other parts of the codebase.
    tmp_path = token_mmap_path.with_suffix(token_mmap_path.suffix + ".tmp")

    # If a previous temp file exists, remove it (stale from interrupted run)
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    # Skip tokenization if final memmap already exists
    if token_mmap_path.exists():
        print(f"Found existing token memmap at {token_mmap_path}, skipping tokenization.")
        existing_n_tokens = token_mmap_path.stat().st_size // np.dtype(dtype).itemsize
        return int(existing_n_tokens)  

    print(f"Token memmap not found at {token_mmap_path}, building now...")

    # Pass 1: count tokens to pre-allocate memmap (append EOS token if requested)
    is_valid_text = lambda t: (t is not None) and bool(t.strip())  # noqa: E731

    total_tokens = 0
    for text in texts:
        # Skip empty or whitespace-only texts to avoid unnecessary tokens
        if not is_valid_text(text):
            continue
        total_tokens += len(encode_fn(text))
        if append_eos:
            total_tokens += 1

    # Pass 2: tokenize and write to a temporary memmap, then atomically move
    # it into place. This avoids leaving the final memmap in a partial state
    # if the process is interrupted.
    try:
        arr = np.memmap(tmp_path, mode="w+", dtype=dtype, shape=(total_tokens,))
        write_index = 0
        for text in texts:
            if not is_valid_text(text):
                continue
            ids = encode_fn(text)
            n = len(ids)
            arr[write_index : write_index + n] = ids
            write_index += n
            if append_eos:
                arr[write_index] = eos_id
                write_index += 1

        arr.flush()  # ensure data is written to disk
        assert write_index == total_tokens  # sanity check

        # Atomically replace the final file with the tmp file
        os.replace(tmp_path, token_mmap_path)

    except Exception:
        # Clean up tmp file on error to avoid leaving partial artifacts
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

    return int(total_tokens)


def load_memmap_tokens(local_mmap_path: str | Path, dtype: np.dtype) -> np.memmap:
    return np.memmap(local_mmap_path, mode="r", dtype=dtype)

# ============================================================
# HUGGINGFACE DATASETS
# ============================================================
def load_text_splits_from_hf(
    dataset_name: str,
    train_split: str,
    eval_split: str,
    dataset_config: str | None = None,
    text_field: str = "text",
    cache_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    
    kwargs = dict(
        path=dataset_name,
        name=dataset_config,
        cache_dir=cache_dir,
    )
    train_ds = load_dataset(split=train_split, **kwargs)  # type: ignore[index]
    eval_ds = load_dataset(split=eval_split, **kwargs)  # type: ignore[index]
    train_texts = [x[text_field] for x in train_ds]  # type: ignore[index]
    eval_texts = [x[text_field] for x in eval_ds]  # type: ignore[index]

    return train_texts, eval_texts


# ============================================================
# DETERMINISTIC SAMPLING: SAMPLE_IDX -> WINDOW START
# ============================================================
def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


class DeterministicTokenWindows(Dataset):
    """
    Deterministic "random windows" over a contiguous token buffer.
    Reproducible + resumable: window depends only on (seed, global_sample_idx).
    Windows are not guaranteed to be unique or non-overlapping.
    """

    def __init__(
        self,
        tokens_mmap_buffer: np.memmap,
        seq_len: int,
        seed: int,
    ):
        self.tokens_memmap_buffer = tokens_mmap_buffer
        self.seq_len = int(seq_len)
        self.seed = int(seed)

        # Init max valid start index for a window of length seq_len
        self.max_idx = len(self.tokens_memmap_buffer) - (self.seq_len + 1)
        if self.max_idx <= 0:
            raise ValueError("Token buffer too small for seq_len")

    def __len__(self) -> int:
        return 2**63 - 1 # effectively infinite (max int64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Hash sample idx to get a pseudo-random but deterministic window start
        hashed_index = _splitmix64(self.seed ^ int(idx))
        start_index = int(
            hashed_index % (self.max_idx + 1)
        )  # start index must be in [0, max_idx]

        # Return input and target windows as PyTorch tensors
        x = np.asarray(
            self.tokens_memmap_buffer[start_index : start_index + self.seq_len], dtype=np.int64
        )
        y = np.asarray(
            self.tokens_memmap_buffer[start_index + 1 : start_index + 1 + self.seq_len],
            dtype=np.int64,
        )
        return torch.from_numpy(x), torch.from_numpy(y)


class OffsetDataset(Dataset):
    """View a dataset starting at a global sample index (for resume)."""

    def __init__(self, dataset: Dataset, offset_start: int):
        self.dataset = dataset
        self.offset = int(offset_start)
        if self.offset < 0 or self.offset > len(self.dataset):
            raise ValueError("Invalid start")

    def __len__(self) -> int:
        return 2**63 - 1 # effectively infinite

    def __getitem__(self, i: int):
        return self.dataset[self.offset + i]


class SequentialTokenChunks(Dataset):
    """
    Deterministic eval: sequential non-overlapping chunks over val buffer.
    """

    def __init__(self, tokens_mmap_buffer: np.memmap, seq_len: int):
        self.tokens_memmap_buffer = tokens_mmap_buffer
        self.seq_len = int(seq_len)
        self.num_sequences = (len(self.tokens_memmap_buffer) - 1) // self.seq_len
        if self.num_sequences <= 0:
            raise ValueError("Token buffer too small for seq_len")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.seq_len  # start index of the chunk
        x = np.asarray(self.tokens_memmap_buffer[i : i + self.seq_len], dtype=np.int64)
        y = np.asarray(self.tokens_memmap_buffer[i + 1 : i + 1 + self.seq_len], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================
# TOKENIZED DATASET FACTORY
# ============================================================
def make_tokenized_dataset(
    dataset_id: DatasetIdentity,
    data_registry: DataRegistry,
    local_train_mmap_path: Path,
    local_eval_mmap_path: Path,
    local_hf_cache_dir: Path,
    local_tokenized_cache_dir: Path,
    transfer_mode: str,
) -> Path:
    """
    TODO
    """
    # Ensure local dataset cache size does not exceed maximum allowed size
    for cache_dir in [local_hf_cache_dir, local_tokenized_cache_dir]:
        ensure_local_dataset_cache_cap(
            cache_dir=cache_dir,
            dataset_name=dataset_id.dataset_name,
            dataset_config=dataset_id.dataset_config,
            cap_size_gb=MAX_CACHE_GB,
        )

    # Load tokenizer
    tokenizer = load_tokenizer(dataset_id.tokenizer_name)
    encode_fn, eos_id, vocab_size = tokenizer["encode"], tokenizer["eot_token"], tokenizer["vocab_size"]

    # Use smallest unsigned dtype that can represent all token ids to save memory
    dtype = np.dtype(np.uint16 if vocab_size <= 65535 else np.uint32)
    
    # STEP 1: Load raw text splits from HuggingFace datasets (optionally sliced) 
    train_texts, eval_texts = load_text_splits_from_hf(
        dataset_name=dataset_id.dataset_name,
        train_split=dataset_id.train_split,
        eval_split=dataset_id.eval_split,
        dataset_config=dataset_id.dataset_config,
        text_field=dataset_id.text_field,
        cache_dir=local_hf_cache_dir,
    )

    # STEP 2: Tokenization 
    # Tokenize to memmap once and reuse forever 
    # (efficient random access without loading all tokens into RAM)
    # Note: for large datasets, this may take time and disk space on first run,
    # but subsequent runs will be fast and efficient due to memory-mapping.
    local_train_mmap_path = local_train_mmap_path
    local_eval_mmap_path = local_eval_mmap_path

    total_train_tokens = build_memmap_tokens(
        token_mmap_path=local_train_mmap_path,
        texts=train_texts,
        encode_fn=encode_fn,
        eos_id=eos_id,
        dtype=dtype,
        append_eos=True,
    )

    total_eval_tokens = build_memmap_tokens(
        token_mmap_path=local_eval_mmap_path,
        texts=eval_texts,
        encode_fn=encode_fn,
        eos_id=eos_id,
        dtype=dtype,
        append_eos=True,
    )

    # STEP 3: Register dataset in Data Registry
    dataset_info = TokenizedDatasetInfo(
        vocab_size=vocab_size,
        eos_id=eos_id,
        dtype=dtype.name,
        total_train_tokens=total_train_tokens,
        total_eval_tokens=total_eval_tokens
    )

    registered_dataset_path = data_registry.register_dataset(
        src_path_train_bin=local_train_mmap_path,
        src_path_eval_bin=local_eval_mmap_path,
        identity=dataset_id,
        dataset_info=dataset_info,
        mode=transfer_mode,
        vocab_size=vocab_size,
        total_train_tokens=total_train_tokens,
        total_eval_tokens=total_eval_tokens,
    )

    return registered_dataset_path


# ============================================================
# DATALOADER FACTORY
# ============================================================
@dataclass(frozen=True)
class DataLoaderConfig(BaseJsonConfig):
    seq_len: int
    train_batch_size: int
    eval_batch_size: int
    start_sample_idx: int
    seed: int

    # (Optional) DataLoader performance
    num_workers: int = 1
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2

    def __post_init__(self):
        if self.start_sample_idx < 0:
            raise ValueError(f"start_sample_idx must be >= 0, got {self.start_sample_idx}")


def make_dataloaders(
    # Token Mem-Map Info
    local_train_mmap_path: Path,
    local_eval_mmap_path: Path,
    dtype: np.dtype,
    # Dataloader Config
    dataloader_config: DataLoaderConfig
) -> dict[str, Any]:

    # STEP 1: Load token buffers using memory-mapping for efficient random access without loading all tokens into RAM
    train_tokens_buffer = load_memmap_tokens(local_train_mmap_path, dtype=dtype)
    eval_tokens_buffer = load_memmap_tokens(local_eval_mmap_path, dtype=dtype)

    # STEP 2: Create deterministic train schedule (token-budget driven)
    ## a) Train DS: deterministic random windows, with global sample index offset for resume
    deterministic_train_ds = DeterministicTokenWindows(
        tokens_mmap_buffer=train_tokens_buffer,
        seq_len=dataloader_config.seq_len,
        seed=dataloader_config.seed,
    )
    train_ds = OffsetDataset(
        deterministic_train_ds,
        offset_start=dataloader_config.start_sample_idx,
    )

    ## b) Eval DS: sequential non-overlapping chunks
    eval_ds = SequentialTokenChunks(
        eval_tokens_buffer,
        seq_len=dataloader_config.seq_len,
    )

    # STEP 3: Create PyTorch dataloaders
    dl_kwargs = dict(
        shuffle=False, # windows are already deterministic/randomized
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        drop_last=dataloader_config.drop_last,
        persistent_workers=dataloader_config.persistent_workers,
        prefetch_factor=dataloader_config.prefetch_factor
    )

    train_dl = DataLoader(train_ds, batch_size=dataloader_config.train_batch_size, **dl_kwargs)
    eval_dl = DataLoader(eval_ds, batch_size=dataloader_config.eval_batch_size, **dl_kwargs)

    return {
        "train": train_dl,
        "eval": eval_dl,
    }


# ============================================================
# HIGH-LEVEL DATALOADER FACTORY WITH REGISTRY INTEGRATION
# ============================================================
@dataclass(frozen=True)
class LocalDataPaths:
    local_data_dir: Path = LOCAL_DATA_DIR
    hf_cache_dir: Path = field(init=False)
    tokenized_cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hf_cache_dir", Path(self.local_data_dir) / HF_CACHE_DIR_NAME)
        object.__setattr__(self, "tokenized_cache_dir", Path(self.local_data_dir) / TOKENIZED_CACHE_DIR_NAME)

    def dataset_dir(self, dataset_id: DatasetIdentity) -> Path:
        return self.tokenized_cache_dir / dataset_id.slug()

    def train_mmap_path(self, dataset_id: DatasetIdentity) -> Path:
        return self.dataset_dir(dataset_id) / DATASET_FILES.train_tokens

    def eval_mmap_path(self, dataset_id: DatasetIdentity) -> Path:
        return self.dataset_dir(dataset_id) / DATASET_FILES.eval_tokens
    

def get_dataloaders(
    dataset_id: DatasetIdentity,
    data_registry: DataRegistry,
    dataloader_config: DataLoaderConfig,
    local_data_dir: Path | None = None,
    dataset_info: TokenizedDatasetInfo | None = None,
    run: Run | None = None,
    transfer_mode: str = "shutil",
) -> dict[str, Any]:

    if transfer_mode not in {"shutil", "rclone"}:
        raise ValueError(
            f"Invalid transfer_mode: {transfer_mode}. Expected one of: 'shutil', 'rclone'."
        )

    logger = DataLogger(
        name="DataLoader",
        file_name=str(METADATA_FILES.data_log) if run is not None else None,
        log_dir=run.metadata_dir if run is not None else None,
        level=logging.INFO,
    )

    logger.log_dataset_id(dataset_id)
    logger.log_dataloader_config(dataloader_config)    

    # STEP 1: Prepare local paths for token memmaps 
    # NOTE: these are the source for dataloader creation and also used for registry registration, 
    # but the canonical "source of truth" for the dataset is the Data Registry.
    override_kwargs = {} if local_data_dir is None else {"local_data_dir": local_data_dir}
    local_data_paths = LocalDataPaths(**override_kwargs)
    local_train_mmap_path = local_data_paths.train_mmap_path(dataset_id)
    local_eval_mmap_path = local_data_paths.eval_mmap_path(dataset_id)
    local_dataset_dir = local_data_paths.dataset_dir(dataset_id)

    # STEP 2: Create tokenized dataset and register in Data Registry if not already present
    if data_registry.dataset_exists(dataset_id):
        logger.log_tokenization("Token memmaps already exist in Data Registry, proceeding to create dataloaders...")
        registered_dataset_path = data_registry.find_dataset_path(dataset_id, raise_if_not_found=True)
    else:
        logger.log_tokenization("Token memmaps not found in Data Registry, proceeding with local preparation and upload...")
        registered_dataset_path = make_tokenized_dataset(
            dataset_id=dataset_id,
            data_registry=data_registry,
            local_train_mmap_path=local_train_mmap_path,
            local_eval_mmap_path=local_eval_mmap_path,
            local_hf_cache_dir=local_data_paths.hf_cache_dir,
            local_tokenized_cache_dir=local_data_paths.tokenized_cache_dir,
            transfer_mode=transfer_mode,
        )

    # STEP 3: Ensure local memmaps are available for dataloader creation
    if local_train_mmap_path.exists() and local_eval_mmap_path.exists():
        logger.log_tokenization("Memmaps already exist locally, skipping copy from Data Registry.")
    else: 
        logger.log_tokenization("Memmaps not found locally, copying locally from Data Registry...")
        data_registry.copy_dataset_to_local(
            registered_dataset_path,
            local_dataset_dir,
            mode=transfer_mode,
        )

    # Load dataset info from Data Registry
    dataset_info = dataset_info or data_registry.get_dataset_info(dataset_id)
    dtype = np.dtype(dataset_info.dtype)
    logger.log_dataset_info(dataset_info)
    
    # STEP 4: Create dataloaders
    logger.log_dataloader_creation(
        "Creating deterministic training dataset with random windows "
        "and sequential evaluation dataset with non-overlapping chunks..."
    )
    dls = make_dataloaders(
        local_train_mmap_path=local_train_mmap_path,
        local_eval_mmap_path=local_eval_mmap_path,
        dtype=dtype,
        dataloader_config=dataloader_config
    )

    logger.log_dataloader_creation(
        f"Prepared dataloaders with {len(dls['train'])} training batches and "
        f"{len(dls['eval'])} evaluation batches."
    )

    # STEP 5: Prepare output info
    output_info = dict(**asdict(dataset_info), **asdict(dataloader_config))

    return {
        "train": dls["train"],
        "eval": dls["eval"],
        "info": output_info,
    }



