from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from scaling_llms.constants import LOCAL_DATA_DIR_NAME
from scaling_llms.tracking.managers import GoogleDriveDataRegistry


# ============================================================
# Determinism (single GPU)
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
# Tokenizer (tiktoken preferred; fallback to HF GPT2)
# ============================================================
def make_gpt2_tokenizer():
    enc = tiktoken.get_encoding("gpt2")

    def encode_fn(text: str) -> list[int]:
        return enc.encode(text)

    return encode_fn, enc.eot_token, enc.n_vocab


# ============================================================
# Memmap builder (tokenize once, reuse forever)
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
# HuggingFace Datasets
# ============================================================
def load_text_splits_from_hf(
    dataset_name: str,
    train_split: str | None = None,
    val_split: str | None = None,
    dataset_config: str | None = None,
    dataset_revision: str | None = None,
) -> tuple[list[str], list[str]]:
    
    kwargs = dict(
        path=dataset_name,
        name=dataset_config,
        revision=dataset_revision,
    )
    train_ds = load_dataset(split=train_split or "train", **kwargs)  # type: ignore[index]
    val_ds = load_dataset(split=val_split or "validation", **kwargs)  # type: ignore[index]
    train_texts = [x["text"] for x in train_ds]  # type: ignore[index]
    val_texts = [x["text"] for x in val_ds]  # type: ignore[index]

    return train_texts, val_texts




# ============================================================
# Deterministic sampling: sample_idx -> window start
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

@dataclass(frozen=True)
class DataConfig:

    dataset_name: HF_DATASET_NAMES
    seq_len: int 
    train_batch_size: int 
    eval_batch_size: int 

    # Data and cache directories (can be shared across datasets and runs)
    local_data_dir: str | Path = LOCAL_DATA_DIR_NAME

    # Training budget (tokens) + resume cursor (samples)
    train_tokens_budget: int = 1_000_000
    start_sample_idx: int = 0  # resume cursor (global sample index)

    # Always start with 0 for deterministic bring-up
    num_workers: int = 0

    # Optional slicing and revision for HuggingFace datasets
    train_split: str | None = None
    val_split: str | None = None
    dataset_config: str | None = None
    dataset_revision: str | None = None

    seed: int = 1234

    def __post_init__(self):
        self._configure_mmap_paths()
        self._validate_num_samples()

    def _validate_num_samples(self):
        # Validate that start_sample_idx is within the total number of samples given the train_tokens_budget and seq_len
        num_samples = max(1, self.train_tokens_budget // self.seq_len)
        if self.start_sample_idx >= num_samples:
            raise ValueError(
                f"start_sample_idx={self.start_sample_idx} >= num_samples={num_samples} based on train_tokens_budget={self.train_tokens_budget} and seq_len={self.seq_len}"
            )
        
    def _configure_mmap_paths(self):
        self.local_data_dir = Path(self.local_data_dir).expanduser().resolve()
        out_root = self.local_data_dir / "tokenized" / self.dataset_name  # type: ignore[union-attr]
        if self.dataset_config is not None:
            out_root = out_root / self.dataset_config
        # TODO: f"{train_split.replace('/', '_').replace(':', '_')}__{val_split.replace('/', '_').replace(':', '_')}"
        self.train_mmap_path = out_root / "train.bin"
        self.val_mmap_path = out_root / "val.bin"
        
# ============================================================
# DATALOADER FACTORY
# ============================================================
def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    set_determinism(cfg.seed)

    # Init Google Drive data registry for managing data paths and copying between local and drive
    data_registry = GoogleDriveDataRegistry()

    gdrive_data_root = data_registry.get_data_root()
    gdrive_train_mmap_path = gdrive_data_root / cfg.train_mmap_path.relative_to(cfg.local_data_dir)
    gdrive_val_mmap_path = gdrive_data_root / cfg.val_mmap_path.relative_to(cfg.local_data_dir)

    # SKIP Steps 1-2 if memmaps already exist in Google Drive
    if (not gdrive_train_mmap_path.exists()) and (not gdrive_val_mmap_path.exists()):

        # --- STEP 1: Load raw text splits (optionally sliced) ---
        # NOTE: if not using HuggingFace datasets, you can condition on (cfg.dataset_name in HF_DATASET_NAMES)
        # and implement your own loading logic here to produce train_texts and val_texts as lists of strings.
        train_texts, val_texts = load_text_splits_from_hf(
            dataset_name=cfg.dataset_name,
            train_split=cfg.train_split,
            val_split=cfg.val_split,
            dataset_config=cfg.dataset_config,
            dataset_revision=cfg.dataset_revision,
        )   

        # --- STEP 2: Tokenization ---
        encode_fn, eos_id, vocab_size = make_gpt2_tokenizer()

        # Use smallest unsigned dtype that can represent all token ids to save memory
        dtype = np.uint16 if vocab_size <= 65535 else np.uint32

        # Tokenize to memmap once and reuse forever 
        # (efficient random access without loading all tokens into RAM)
        # Note: for large datasets, this may take time and disk space on first run,
        # but subsequent runs will be fast and efficient due to memory-mapping.
        for mmap_path,texts in [(cfg.train_mmap_path, train_texts), (cfg.val_mmap_path, val_texts)]:
            build_memmap_tokens(
                token_mmap_path=mmap_path,
                texts=texts,
                encode_fn=encode_fn,
                eos_id=eos_id,
                dtype=dtype,
            )

        # Copy local memmaps to Google Drive
        data_registry.copy_local_to_data_root(cfg.train_mmap_path, gdrive_train_mmap_path, overwrite=True)
        data_registry.copy_local_to_data_root(cfg.val_mmap_path, gdrive_val_mmap_path, overwrite=True)

    # --- STEP 3: Create token buffers for training and evaluation ---
    if (not cfg.train_mmap_path.exists()) and (not cfg.val_mmap_path.exists()):
        print("Memmaps not found locally, copying locally from Google Drive...")
        data_registry.copy_data_root_to_local(gdrive_train_mmap_path, cfg.train_mmap_path, overwrite=True)
        data_registry.copy_data_root_to_local(gdrive_val_mmap_path, cfg.val_mmap_path, overwrite=True)

    train_tokens_buffer = np.memmap(cfg.train_mmap_path, mode="r", dtype=dtype)
    val_tokens_buffer = np.memmap(cfg.val_mmap_path, mode="r", dtype=dtype)

    # --- STEP 4: Create deterministic train schedule (token-budget driven) ---
    # 1 sample == 1 sequence of length seq_len

    # Train DS: deterministic random windows, with global sample index offset for resume
    deterministic_train_ds = DeterministicTokenWindows(
        tokens_mmap_buffer=train_tokens_buffer,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        num_samples=max(1, cfg.train_tokens_budget // cfg.seq_len),
    )
    train_ds = OffsetDataset(deterministic_train_ds, offset_start=cfg.start_sample_idx)

    # Eval DS: sequential non-overlapping chunks
    eval_ds = SequentialTokenChunks(val_tokens_buffer, seq_len=cfg.seq_len)

    # --- STEP 5: Create PyTorch dataloaders ---
    dl_kwargs = dict(
        shuffle=False,  # windows are already deterministic/randomized
        num_workers=cfg.num_workers, 
        pin_memory=True,
        drop_last=False,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.train_batch_size, **dl_kwargs)
    eval_dl = DataLoader(eval_ds, batch_size=cfg.eval_batch_size, **dl_kwargs)

    return train_dl, eval_dl











    # info: Dict[str, Any] = {
    #     "dataset_name": dataset_name,
    #     "out_root": str(out_root),
    #     "train_mmap_path": str(train_mmap_path),
    #     "val_mmap_path": str(val_mmap_path),
    #     "train_tokens_in_buffer": int(len(train_tokens_buffer)),
    #     "val_tokens_in_buffer": int(len(val_tokens_buffer)),
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

