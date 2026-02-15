from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
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
from scaling_llms.tracking.registries import GoogleDriveDataRegistry
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
# TOKENIZER (tiktoken preferred; fallback to HF GPT2)
# ============================================================
def make_gpt2_tokenizer():
    enc = tiktoken.get_encoding("gpt2")

    def encode_fn(text: str) -> list[int]:
        return enc.encode(text)

    return encode_fn, enc.eot_token, enc.n_vocab


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
) -> Path:
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

    # Skip tokenization if already exists
    
    if token_mmap_path.exists():
        print(f"Found existing token memmap at {token_mmap_path}, skipping tokenization.")
        return token_mmap_path

    print(f"Token memmap not found at {token_mmap_path}, building now...")

    # Pass 1: count tokens to pre-allocate memmap (append EOS token if requested)
    total_tokens = 0
    for text in texts:
        # Skip empty or whitespace-only texts to avoid unnecessary tokens
        if not text.strip():
            continue
        total_tokens += len(encode_fn(text))
        if append_eos:
            total_tokens += 1

    # Pass 2: tokenize and write to memmap
    arr = np.memmap(token_mmap_path, mode="w+", dtype=dtype, shape=(total_tokens,))
    write_index = 0
    for text in texts:
        if not text.strip():
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

    return token_mmap_path


# ============================================================
# HUGGINGFACE DATASETS
# ============================================================
def load_text_splits_from_hf(
    dataset_name: str,
    train_split: str,
    eval_split: str,
    dataset_config: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    
    kwargs = dict(
        path=dataset_name,
        name=dataset_config,
        cache_dir=cache_dir,
    )
    train_ds = load_dataset(split=train_split, **kwargs)  # type: ignore[index]
    eval_ds = load_dataset(split=eval_split, **kwargs)  # type: ignore[index]
    train_texts = [x["text"] for x in train_ds]  # type: ignore[index]
    eval_texts = [x["text"] for x in eval_ds]  # type: ignore[index]

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
        num_samples: int,
    ):
        self.tokens_memmap_buffer = tokens_mmap_buffer
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.num_samples = int(num_samples)

        # Init max valid start index for a window of length seq_len
        self.max_idx = len(self.tokens_memmap_buffer) - (self.seq_len + 1)
        if self.max_idx <= 0:
            raise ValueError("Token buffer too small for seq_len")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be > 0")

    def __len__(self) -> int:
        return self.num_samples

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
        return len(self.dataset) - self.offset

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
HF_DATASET_NAMES = Literal[
    "wikitext-103", 
    "openwebtext"
]

@dataclass
class DataConfig(BaseJsonConfig):

    dataset_name: HF_DATASET_NAMES
    seq_len: int 
    train_batch_size: int 
    eval_batch_size: int 

    # Data dir to setup cache directories for HF datasets and tokenized memmaps.
    local_data_dir: Path = LOCAL_DATA_DIR

    # Training budget (tokens) + resume cursor (samples)
    train_tokens_budget: int = 1_000_000
    start_sample_idx: int = 0  # resume cursor (global sample index)

    # Always start with 0 for deterministic bring-up
    num_workers: int = 0

    # Optional slicing and revision for HuggingFace datasets
    train_split: str | None = "train"
    eval_split: str | None = "test"
    dataset_config: str | None = None

    seed: int = 1234

    def __post_init__(self):
        self.train_split = self.train_split or "train"
        self.eval_split = self.eval_split or "test"
        self.hf_cache_dir = Path(self.local_data_dir) / HF_CACHE_DIR_NAME
        self.tokenized_cache_dir = Path(self.local_data_dir) / TOKENIZED_CACHE_DIR_NAME

        self._configure_local_mmap_paths()
        self._validate_num_samples()

    def _validate_num_samples(self):
        # Validate that start_sample_idx is within the total number of samples given the train_tokens_budget and seq_len
        num_samples = max(1, self.train_tokens_budget // self.seq_len)
        if self.start_sample_idx >= num_samples:
            raise ValueError(
                f"start_sample_idx={self.start_sample_idx} >= num_samples={num_samples} based on train_tokens_budget={self.train_tokens_budget} and seq_len={self.seq_len}"
            )
        
    def _configure_local_mmap_paths(self):
        local_dataset_path = self.tokenized_cache_dir / self.dataset_name
        if self.dataset_config is not None:
            local_dataset_path = local_dataset_path / self.dataset_config
        suffix = ""
        if self.train_split is not None:
            suffix += f"train={self.train_split.replace('/', '_').replace(':', '_')}"
        if self.eval_split is not None:
            suffix += f"__eval={self.eval_split.replace('/', '_').replace(':', '_')}"
        if suffix:
            local_dataset_path = local_dataset_path / f"_{suffix}"

        self.local_train_mmap_path = local_dataset_path / DATA_FILES.train_tokens
        self.local_eval_mmap_path = local_dataset_path / DATA_FILES.eval_tokens
        

# ============================================================
# DATALOADER FACTORY
# ============================================================
def get_dataloaders(cfg: DataConfig, run=None, **gdrive_overrides) -> dict[str, Any]:
        
    """
    Raw texts are downloaded and cached locally in `hf_cache_dir`
    Tokenized memmaps are stored in `tokenized_cache_dir` and copied to Google Drive for persistence.
        - gdrive_data_root/dataset_name/[dataset_config]/train.bin
        - gdrive_data_root/dataset_name/[dataset_config]/val.bin
    """

    logger = DataLogger(
        name="DataLoader",
        file_name=str(RUN_FILES.data_log) if run else None,
        log_dir=run.get_metadata_dir() if run else None,  # writes metadata/train.log
        level=logging.INFO,
    )
    logger.log_start(cfg)

    set_determinism(cfg.seed)

    encode_fn, eos_id, vocab_size = make_gpt2_tokenizer()

    # Use smallest unsigned dtype that can represent all token ids to save memory
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    
    logger.log_batch_info(
        vocab_size=vocab_size, 
        seq_len=cfg.seq_len, 
        train_batch_size=cfg.train_batch_size, 
        eval_batch_size=cfg.eval_batch_size, 
        dtype=f"np.{np.dtype(dtype).name}", 
    )

    # Init Google Drive data registry for managing data paths and copying between local and drive
    data_registry = GoogleDriveDataRegistry(**gdrive_overrides)
    gdrive_dataset_path = data_registry.find_dataset_path(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        train_split=cfg.train_split,
        eval_split=cfg.eval_split,
        raise_if_not_found=False,   
    )
    
    # SKIP Steps 1-2 if memmaps already exist in Data Registry
    if (gdrive_dataset_path is not None) and (gdrive_dataset_path.exists()):
        logger.log_dataset_loading("Token memmaps already exist in Data Registry, proceeding to create dataloaders...")
    else:
        logger.log_dataset_loading("Token memmaps not found in Data Registry, proceeding with local preparation and upload...")

        # Ensure local dataset cache size does not exceed maximum allowed size
        for cache_dir in [cfg.hf_cache_dir, cfg.tokenized_cache_dir]:
            ensure_local_dataset_cache_cap(
                cache_dir=cache_dir,
                dataset_name=cfg.dataset_name,
                dataset_config=cfg.dataset_config,
                cap_size_gb=MAX_CACHE_GB, # per cache dir
            )

        # --- STEP 1: Load raw text splits (optionally sliced) ---
        # NOTE: if not using HuggingFace datasets, you can condition on (cfg.dataset_name in HF_DATASET_NAMES)
        # and implement your own loading logic here to produce train_texts and eval_texts as lists of strings.
        logger.log_dataloader_info("Loading raw text splits from HuggingFace...")
        train_texts, eval_texts = load_text_splits_from_hf(
            dataset_name=cfg.dataset_name,
            train_split=cfg.train_split,
            eval_split=cfg.eval_split,
            dataset_config=cfg.dataset_config,
            cache_dir=cfg.hf_cache_dir,
        )   

        # --- STEP 2: Tokenization ---
        # Tokenize to memmap once and reuse forever 
        # (efficient random access without loading all tokens into RAM)
        # Note: for large datasets, this may take time and disk space on first run,
        # but subsequent runs will be fast and efficient due to memory-mapping.
        for mmap_path,texts in [(cfg.local_train_mmap_path, train_texts), (cfg.local_eval_mmap_path, eval_texts)]:
            logger.log_tokenization(f"Tokenizing texts and building memmap at {mmap_path}...")
            build_memmap_tokens(
                token_mmap_path=mmap_path,
                texts=texts,
                encode_fn=encode_fn,
                eos_id=eos_id,
                dtype=dtype,
            )

        # Copy local memmaps to Data Registry
        logger.log_tokenization("Registering token memmaps to Data Registry...")
        gdrive_dataset_path = data_registry.register_dataset(
            src_path_train_bin=cfg.local_train_mmap_path,
            src_path_eval_bin=cfg.local_eval_mmap_path,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            train_split=cfg.train_split,
            eval_split=cfg.eval_split,
        )

    # --- STEP 3: Create token buffers for training and evaluation ---
    if cfg.local_train_mmap_path.exists() and cfg.local_eval_mmap_path.exists():
        logger.log_token_buffer_loading("Memmaps already exist locally, skipping copy from Data Registry.")
    else: 
        logger.log_token_buffer_loading("Memmaps not found locally, copying locally from Data Registry...")
        data_registry.copy_dataset_to_local(gdrive_dataset_path, cfg.local_train_mmap_path.parent)        

    logger.log_token_buffer_loading(f"Loading token memmaps from {cfg.local_train_mmap_path.parent}...")
    train_tokens_buffer = np.memmap(cfg.local_train_mmap_path, mode="r", dtype=dtype)
    eval_tokens_buffer = np.memmap(cfg.local_eval_mmap_path, mode="r", dtype=dtype)

    # --- STEP 4: Create deterministic train schedule (token-budget driven) ---
    # 1 sample == 1 sequence of length seq_len

    # Train DS: deterministic random windows, with global sample index offset for resume
    logger.log_dataloader_info("Creating deterministic training dataset with random windows...")
    deterministic_train_ds = DeterministicTokenWindows(
        tokens_mmap_buffer=train_tokens_buffer,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        num_samples=max(1, cfg.train_tokens_budget // cfg.seq_len),
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
    }
    return {
        "train": train_dl, 
        "eval": eval_dl, 
        "info": info
    }











    # info: Dict[str, Any] = {
    #     "dataset_name": dataset_name,
    #     "out_root": str(out_root),
    #     "train_mmap_path": str(train_mmap_path),
    #     "val_mmap_path": str(val_mmap_path),
    #     "train_tokens_in_buffer": int(len(train_tokens_buffer)),
    #     "eval_tokens_in_buffer": int(len(eval_tokens_buffer)),
    #     "vocab_size": vocab_size,
    #     "eos_id": eos_id,
    #     "dtype": str(dtype),
    #     "seq_len": cfg.seq_len,
    #     "seed": cfg.seed,
    #     "train_tokens_budget": cfg.train_tokens_budget,
    #     "total_samples": max(1, cfg.train_tokens_budget // cfg.seq_len),
    #     "start_sample_idx": cfg.start_sample_idx,
    #     "num_workers": cfg.num_workers,
    #     "wikitext_train_split": (
    #         cfg.wikitext_train_split if dataset_name.startswith("wikitext") else None
    #     ),
    #     "wikitext_val_split": (
    #         cfg.wikitext_val_split if dataset_name.startswith("wikitext") else None
    #     ),
    #     "wikitext_revision": (
    #         cfg.wikitext_revision if dataset_name.startswith("wikitext") else None
    #     ),
    #     "owt_split": (cfg.owt_split if dataset_name == "openwebtext" else None),
    #     "owt_revision": (cfg.owt_revision if dataset_name == "openwebtext" else None),
    #     "owt_val_size": (cfg.owt_val_size if dataset_name == "openwebtext" else None),
    # }

    # return train_dl, eval_dl, info

