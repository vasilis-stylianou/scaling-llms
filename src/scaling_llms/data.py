from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    DATA_FILES,
    LOCAL_DATA_DIR,
    HF_CACHE_DIR_NAME,
    MAX_CACHE_GB,
    TOKENIZED_CACHE_DIR_NAME,
    RUN_FILES
)
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
    *,
    token_mmap_path: str | Path,
    texts: list[str],
    encode_fn,
    eos_id: int,
    dtype=np.uint16,
    append_eos: bool = True,
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
        return # TODO: return existing token count from registry instead of None

    print(f"Token memmap not found at {token_mmap_path}, building now...")

    # Pass 1: count tokens to pre-allocate memmap (append EOS token if requested)
    total_tokens = 0
    for text in texts:
        # Skip empty or whitespace-only texts to avoid unnecessary tokens
        if (text is None) or (not text.strip()):
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
            if (text is None) or (not text.strip()):
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

    return total_tokens


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
# DATA CONFIG
# ============================================================
@dataclass
class DataConfig(BaseJsonConfig):
    # --- Registry identity (unique key) ---
    dataset_name: str
    train_split: str
    eval_split: str
    dataset_config: str | None
    tokenizer_name: str 
    text_field: str

    # --- Dataloader params ---
    seq_len: int
    train_batch_size: int
    eval_batch_size: int
    start_sample_idx: int = 0
    # Resume cursor (global sample index) for deterministic train schedule; 
    # effectively an offset into the infinite stream of random windows.

    # --- Local cache roots ---
    local_data_dir: Path = LOCAL_DATA_DIR

    # --- Other params ---
    num_workers: int = 0
    seed: int = 1234

    # --- Derived paths (set in __post_init__) ---
    hf_cache_dir: Path = field(init=False)
    tokenized_cache_dir: Path = field(init=False)
    local_train_mmap_path: Path = field(init=False)
    local_eval_mmap_path: Path = field(init=False)

    def __post_init__(self):
        hf_cache_dir = Path(self.local_data_dir) / HF_CACHE_DIR_NAME
        tokenized_cache_dir = Path(self.local_data_dir) / TOKENIZED_CACHE_DIR_NAME
        object.__setattr__(self, "hf_cache_dir", hf_cache_dir)
        object.__setattr__(self, "tokenized_cache_dir", tokenized_cache_dir)

        self._validate_start_sample_idx()
        self._configure_local_mmap_paths()

    def identity(self) -> DatasetIdentity:
        return DatasetIdentity(
            dataset_name=self.dataset_name,
            train_split=self.train_split,
            eval_split=self.eval_split,
            dataset_config=self.dataset_config,
            tokenizer_name=self.tokenizer_name,
            text_field=self.text_field,
        )

    def _validate_start_sample_idx(self):
        if self.start_sample_idx < 0:
            raise ValueError(f"start_sample_idx must be >= 0, got {self.start_sample_idx}")

    def _configure_local_mmap_paths(self):
        ident = self.identity()

        local_dataset_path = self.tokenized_cache_dir / ident.slug()

        object.__setattr__(
            self,
            "local_train_mmap_path",
            local_dataset_path / DATA_FILES.train_tokens,
        )

        object.__setattr__(
            self,
            "local_eval_mmap_path",
            local_dataset_path / DATA_FILES.eval_tokens,
        )

# ============================================================
# DATALOADER FACTORY
# ============================================================
def get_dataloaders(
    cfg: DataConfig, 
    data_registry: DataRegistry, 
    run : Run | None = None
) -> dict[str, Any]:
        
    """
    Raw texts are downloaded and cached locally in `hf_cache_dir`
    Tokenized memmaps are stored in `tokenized_cache_dir` and copied to Data Registry for persistence.
        - data_registry/dataset_name/[dataset_config]/train.bin
        - data_registry/dataset_name/[dataset_config]/val.bin
    """

    logger = DataLogger(
        name="DataLoader",
        file_name=str(RUN_FILES.data_log) if run is not None else None,
        log_dir=run.metadata_dir if run is not None else None,  # writes metadata/train.log
        level=logging.INFO,
    )
    logger.log_start(cfg)

    set_determinism(cfg.seed)

    # Load tokenizer
    tokenizer = load_tokenizer(cfg.tokenizer_name)
    encode_fn, eos_id, vocab_size = tokenizer["encode"], tokenizer["eot_token"], tokenizer["vocab_size"]

    # Use smallest unsigned dtype that can represent all token ids to save memory
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    
    logger.log_batch_info(
        vocab_size=vocab_size, 
        seq_len=cfg.seq_len, 
        train_batch_size=cfg.train_batch_size, 
        eval_batch_size=cfg.eval_batch_size, 
        dtype=f"np.{np.dtype(dtype).name}", 
    )

    # Get dataset identity and define unique key for Data Registry lookup
    ident = cfg.identity()
    dataset_key = ident.as_kwargs()
    
    # SKIP Steps 1-2 if memmaps already exist in Data Registry
    if data_registry.dataset_exists(**dataset_key):
        logger.log_dataset_loading("Token memmaps already exist in Data Registry, proceeding to create dataloaders...")
        registered_dataset_path = data_registry.find_dataset_path(**dataset_key, raise_if_not_found=True)
    else:
        logger.log_dataset_loading("Token memmaps not found in Data Registry, proceeding with local preparation and upload...")

        # Ensure local dataset cache size does not exceed maximum allowed size
        for cache_dir in [cfg.hf_cache_dir, cfg.tokenized_cache_dir]:
            ensure_local_dataset_cache_cap(
                cache_dir=cache_dir,
                dataset_name=ident.dataset_name,
                dataset_config=ident.dataset_config,
                cap_size_gb=MAX_CACHE_GB, # per cache dir
            )

        # --- STEP 1: Load raw text splits (optionally sliced) ---
        # NOTE: if not using HuggingFace datasets, you can condition on (e.g. cfg.dataset_name in HF_DATASET_NAMES)
        # and implement your own loading logic here to produce train_texts and eval_texts as lists of strings.
        logger.log_dataloader_info("Loading raw text splits from HuggingFace...")
        train_texts, eval_texts = load_text_splits_from_hf(
            dataset_name=ident.dataset_name,
            train_split=ident.train_split,
            eval_split=ident.eval_split,
            dataset_config=ident.dataset_config,
            text_field=ident.text_field,
            cache_dir=cfg.hf_cache_dir,
        )   

        # --- STEP 2: Tokenization ---
        # Tokenize to memmap once and reuse forever 
        # (efficient random access without loading all tokens into RAM)
        # Note: for large datasets, this may take time and disk space on first run,
        # but subsequent runs will be fast and efficient due to memory-mapping.
        for mmap_path,texts in [(cfg.local_train_mmap_path, train_texts), (cfg.local_eval_mmap_path, eval_texts)]:
            logger.log_tokenization(f"Tokenizing texts and building memmap at {mmap_path}...")
            total_tokens = build_memmap_tokens(
                token_mmap_path=mmap_path,
                texts=texts,
                encode_fn=encode_fn,
                eos_id=eos_id,
                dtype=dtype,
            )
            if mmap_path == cfg.local_train_mmap_path:
                total_train_tokens = total_tokens
            else:
                total_eval_tokens = total_tokens

        # Copy local memmaps to Data Registry
        logger.log_tokenization("Registering token memmaps to Data Registry...")
        registered_dataset_path = data_registry.register_dataset(
            src_path_train_bin=cfg.local_train_mmap_path,
            src_path_eval_bin=cfg.local_eval_mmap_path,
            vocab_size=vocab_size,
            total_train_tokens=total_train_tokens,
            total_eval_tokens=total_eval_tokens,
            **dataset_key
        )

    # --- STEP 3: Create token buffers for training and evaluation ---
    if cfg.local_train_mmap_path.exists() and cfg.local_eval_mmap_path.exists():
        logger.log_token_buffer_loading("Memmaps already exist locally, skipping copy from Data Registry.")
    else: 
        logger.log_token_buffer_loading("Memmaps not found locally, copying locally from Data Registry...")
        data_registry.copy_dataset_to_local(registered_dataset_path, cfg.local_train_mmap_path.parent)        

    logger.log_token_buffer_loading(f"Loading token memmaps from {cfg.local_train_mmap_path.parent}...")
    train_tokens_buffer = np.memmap(cfg.local_train_mmap_path, mode="r", dtype=dtype)
    eval_tokens_buffer = np.memmap(cfg.local_eval_mmap_path, mode="r", dtype=dtype)

    # --- STEP 4: Create deterministic train schedule (token-budget driven) ---
    # Train DS: deterministic random windows, with global sample index offset for resume
    logger.log_dataloader_info("Creating deterministic training dataset with random windows...")
    deterministic_train_ds = DeterministicTokenWindows(
        tokens_mmap_buffer=train_tokens_buffer,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
    )
    train_ds = OffsetDataset(deterministic_train_ds, offset_start=cfg.start_sample_idx)

    # Eval DS: sequential non-overlapping chunks
    logger.log_dataloader_info("Creating sequential evaluation dataset with non-overlapping chunks...")
    eval_ds = SequentialTokenChunks(eval_tokens_buffer, seq_len=cfg.seq_len)

    # --- STEP 5: Create PyTorch dataloaders ---
    dl_kwargs = dict(
        shuffle=False,  # windows are already deterministic/randomized
        num_workers=cfg.num_workers, 
        pin_memory=True,
        drop_last=False,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.train_batch_size, **dl_kwargs)
    eval_dl = DataLoader(eval_ds, batch_size=cfg.eval_batch_size, **dl_kwargs)

    logger.log_dataloader_info(f"Prepared dataloaders with {len(train_ds)} training samples and {len(eval_ds)} evaluation samples.")

    info = {
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        # "total_train_tokens": total_train_tokens,
        # "total_eval_tokens": total_eval_tokens,
    }
    return {
        "train": train_dl, 
        "eval": eval_dl, 
        "info": info
    }