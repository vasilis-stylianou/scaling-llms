from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import logging


import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler

from scaling_llms.constants import (
    DATASET_FILES,
    HF_CACHE_DIR_NAME,
    METADATA_FILES,
    TOKENIZED_CACHE_DIR_NAME,
)
from scaling_llms.distributed import (
    get_global_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
    barrier_if_distributed,
    is_distributed,
)
from scaling_llms.registries import (
    DatasetArtifactsDir,
    DatasetIdentity,
    DatasetRegistry,
    TokenizedDatasetInfo,
)
from scaling_llms.tracking import Run
from scaling_llms.utils.config import BaseJsonConfig
from scaling_llms.utils.loggers import DataLogger


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


def get_vocab_size(tokenizer_name: str) -> int:
    tokenizer = load_tokenizer(tokenizer_name)
    return tokenizer["vocab_size"]


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
        print(
            f"Found existing token memmap at {token_mmap_path}, skipping tokenization."
        )
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
    eval_split: str | None,
    dataset_config: str | None = None,
    text_field: str = "text",
    cache_dir: Path | None = None,
) -> tuple[list[str], list[str] | None]:

    kwargs = dict(
        path=dataset_name,
        name=dataset_config,
        cache_dir=cache_dir,
    )
    train_ds = load_dataset(split=train_split, **kwargs)  # type: ignore[index]
    train_texts = [x[text_field] for x in train_ds]  # type: ignore[index]

    eval_texts = None
    if eval_split is not None:
        eval_ds = load_dataset(split=eval_split, **kwargs)  # type: ignore[index]
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
        return 2**63 - 1  # effectively infinite (max int64)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Hash sample idx to get a pseudo-random but deterministic window start
        hashed_index = _splitmix64(self.seed ^ int(idx))
        start_index = int(
            hashed_index % (self.max_idx + 1)
        )  # start index must be in [0, max_idx]

        # Return input and target windows as PyTorch tensors
        x = np.asarray(
            self.tokens_memmap_buffer[start_index : start_index + self.seq_len],
            dtype=np.int64,
        )
        y = np.asarray(
            self.tokens_memmap_buffer[start_index + 1 : start_index + 1 + self.seq_len],
            dtype=np.int64,
        )
        return torch.from_numpy(x), torch.from_numpy(y)


class OffsetDataset(Dataset):
    """
    Resume + DDP sharding in one class.
    - offset: consumed_samples (rank-agnostic resume point)
    - rank / world_size: default to 0/1 for single-GPU (no-op)

    Offseting:
    ----------
    Views a dataset starting at a global sample index for resuming training.
    global_sample_idx = offset + i, where i is the index within the Dataset.

    DDP sharding:
    -------------
    Shards an infinite dataset across DDP ranks via strided indexing.

    Each rank *r* (of *W* total) sees indices ``r, W+r, 2W+r, …``.

    Offseting + DDP sharding:
    -------------------------
    The final global index is ``offset + rank + i * world_size``.
    """

    def __init__(
        self,
        dataset: Dataset,
        offset_start: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.offset = int(offset_start)
        self.rank = int(rank)
        self.world_size = int(world_size)

        if self.offset < 0 or self.offset > len(self.dataset):
            raise ValueError("Invalid start")

    def __len__(self) -> int:
        return 2**63 - 1  # effectively infinite

    def __getitem__(self, i: int):
        global_idx = self.offset + self.rank + i * self.world_size
        return self.dataset[global_idx]


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
        y = np.asarray(
            self.tokens_memmap_buffer[i + 1 : i + 1 + self.seq_len], dtype=np.int64
        )
        return torch.from_numpy(x), torch.from_numpy(y)


class DistributedEvalSampler(Sampler[int]):
    """
    Shards a fixed eval dataset across DDP ranks using strided indexing.
    No padding, no dropped samples — every real index appears on exactly one rank.

    Rank r of world size W sees indices r, W+r, 2W+r, … up to the dataset length.

    e.g. with 3 ranks and 10 samples, the indices would be:
    - Rank 0: 0, 3, 6, 9    
    - Rank 1: 1, 4, 7
    - Rank 2: 2, 5, 8
    """
    def __init__(self, dataset_len: int, rank: int, world_size: int) -> None:
        self.dataset_len = dataset_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        yield from range(self.rank, self.dataset_len, self.world_size)

    def __len__(self) -> int:
        if self.rank >= self.dataset_len:
            return 0
        return (self.dataset_len - self.rank + self.world_size - 1) // self.world_size


# ============================================================
# TOKENIZED DATASET FACTORY
# ============================================================
def make_tokenized_dataset(
    dataset_id: DatasetIdentity,
    local_train_mmap_path: Path,
    local_hf_cache_dir: Path,
    local_eval_mmap_path: Path | None = None,
    dataset_registry: DatasetRegistry | None = None,
) -> tuple[DatasetArtifactsDir, TokenizedDatasetInfo]:
    """
    TODO
    """
    # Load tokenizer
    tokenizer = load_tokenizer(dataset_id.tokenizer_name)
    encode_fn, eos_id, vocab_size = (
        tokenizer["encode"],
        tokenizer["eot_token"],
        tokenizer["vocab_size"],
    )

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
    total_train_tokens = build_memmap_tokens(
        token_mmap_path=local_train_mmap_path,
        texts=train_texts,
        encode_fn=encode_fn,
        eos_id=eos_id,
        dtype=dtype,
        append_eos=True,
    )

    total_eval_tokens = None
    if eval_texts is not None and local_eval_mmap_path is not None:
        total_eval_tokens = build_memmap_tokens(
            token_mmap_path=local_eval_mmap_path,
            texts=eval_texts,
            encode_fn=encode_fn,
            eos_id=eos_id,
            dtype=dtype,
            append_eos=True,
        )

    # STEP 3: Build dataset info
    dataset_info = TokenizedDatasetInfo(
        vocab_size=vocab_size,
        eos_id=eos_id,
        dtype=dtype.name,
        total_train_tokens=total_train_tokens,
        total_eval_tokens=total_eval_tokens,
    )

    # STEP 4: Register dataset in Data Registry (if registry provided)
    if dataset_registry is not None:
        artifacts_dir = dataset_registry.register_dataset(
            src_path_train_bin=local_train_mmap_path,
            src_path_eval_bin=local_eval_mmap_path,
            identity=dataset_id,
            dataset_info=dataset_info,
            vocab_size=vocab_size,
            total_train_tokens=total_train_tokens,
            total_eval_tokens=total_eval_tokens,
        )
    else:
        artifacts_dir = DatasetArtifactsDir(root=local_train_mmap_path.parent)

    return artifacts_dir, dataset_info


# ============================================================
# DATALOADER FACTORY
# ============================================================
@dataclass(frozen=True)
class DataLoaderConfig(BaseJsonConfig):
    seq_len: int
    train_batch_size: int           # micro-batch per GPU (hardware tuning knob)
    train_global_batch_size: int    # effective batch size across all GPUs × accum_steps (scientific invariant)
    eval_batch_size: int | None
    start_sample_idx: int
    seed: int

    # (Optional) DataLoader performance
    num_workers: int = 1
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = 2   # None = disabled (required when num_workers=0)

    def __post_init__(self) -> None:
        if self.start_sample_idx < 0:
            raise ValueError(
                f"start_sample_idx must be >= 0, got {self.start_sample_idx}"
            )
        if self.train_batch_size <= 0:
            raise ValueError(
                f"train_batch_size must be > 0, got {self.train_batch_size}"
            )
        if self.train_global_batch_size <= 0:
            raise ValueError(
                f"train_global_batch_size must be > 0, got {self.train_global_batch_size}"
            )
        if self.train_global_batch_size % self.train_batch_size != 0:
            raise ValueError(
                f"train_global_batch_size ({self.train_global_batch_size}) must be "
                f"divisible by train_batch_size ({self.train_batch_size})"
            )
        if self.num_workers == 0:
            if self.prefetch_factor is not None:
                raise ValueError(
                    "prefetch_factor must be None when num_workers=0 "
                    f"(got prefetch_factor={self.prefetch_factor})"
                )
            if self.persistent_workers:
                raise ValueError(
                    "persistent_workers must be False when num_workers=0"
                )

    def derive_accum_steps(self, world_size: int) -> int:
        """
        Compute gradient accumulation steps needed to hit train_global_batch_size
        on the current topology: accum_steps = global / (micro × world_size).
        Raises if the target global batch is not achievable (not divisible).
        """
        if world_size <= 0:
            raise ValueError(f"world_size must be > 0, got {world_size}")
        denom = self.train_batch_size * world_size
        if self.train_global_batch_size % denom != 0:
            raise ValueError(
                f"Cannot hit train_global_batch_size={self.train_global_batch_size} with "
                f"train_batch_size={self.train_batch_size} × world_size={world_size} = {denom}. "
                f"Adjust train_batch_size, world_size, or train_global_batch_size so that "
                f"global is divisible by (micro × world_size)."
            )
        return self.train_global_batch_size // denom

    def get_performance_kwargs(self) -> dict:
        """
        Returns kwargs safe to pass directly to torch DataLoader.
        Guards prefetch_factor and pin_memory against num_workers=0.
        """
        uses_workers = self.num_workers > 0
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and uses_workers,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers and uses_workers,
            prefetch_factor=self.prefetch_factor if uses_workers else None,
        )


def make_dataloaders(
    # Token Mem-Map Info
    local_train_mmap_path: Path,
    dtype: np.dtype,
    # Dataloader Config
    dataloader_config: DataLoaderConfig,
    # Optional eval
    local_eval_mmap_path: Path | None = None,
) -> dict[str, Any]:
    # DDP info for deterministic sampling and distributed eval;
    # defaults to single-GPU if not distributed (rank 0, world size 1)
    rank = get_global_rank()
    world_size = get_world_size()

    # STEP 1: Load token buffers using memory-mapping for efficient random access without loading all tokens into RAM
    train_tokens_buffer = load_memmap_tokens(local_train_mmap_path, dtype=dtype)

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
        # DDP sharding (no-op for single GPU)
        rank=rank,
        world_size=world_size,
    )

    # STEP 3: Create PyTorch dataloaders
    dl_kwargs = dataloader_config.get_performance_kwargs()
    dl_kwargs["shuffle"] = False
    # NOTE: shuffle must stay False when sampler is provided
    # and windows are already deterministic/randomized

    train_dl = DataLoader(
        train_ds,
        batch_size=dataloader_config.train_batch_size,
        **dl_kwargs,
    )

    ## b) Eval DS: sequential non-overlapping chunks (if eval data and batch size exist)
    eval_dl = None
    if local_eval_mmap_path is not None and dataloader_config.eval_batch_size is not None:
        eval_tokens_buffer = load_memmap_tokens(local_eval_mmap_path, dtype=dtype)
        eval_ds = SequentialTokenChunks(
            eval_tokens_buffer,
            seq_len=dataloader_config.seq_len,
        )

        eval_sampler = None
        if is_distributed():
            eval_sampler = DistributedEvalSampler(len(eval_ds), rank, world_size)

        eval_dl = DataLoader(
            eval_ds,
            batch_size=dataloader_config.eval_batch_size,
            sampler=eval_sampler,
            **dl_kwargs,
        )

    return {
        "train": train_dl,
        "eval": eval_dl,
    }


# ============================================================
# HIGH-LEVEL DATALOADER FACTORY WITH REGISTRY INTEGRATION
# ============================================================
@dataclass(frozen=True)
class LocalCachePaths:
    cache_dir: Path | None = None
    tmp_dir: TemporaryDirectory | None = field(init=False)
    hf_cache_dir: Path = field(init=False)
    tokenized_cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        cache_dir, tmp_dir = self._prepare_cache_dir(self.cache_dir)
        object.__setattr__(self, "cache_dir", cache_dir)
        object.__setattr__(self, "tmp_dir", tmp_dir)
        object.__setattr__(
            self, "hf_cache_dir", Path(self.cache_dir) / HF_CACHE_DIR_NAME
        )
        object.__setattr__(
            self, "tokenized_cache_dir", Path(self.cache_dir) / TOKENIZED_CACHE_DIR_NAME
        )

    def dataset_dir(self, dataset_id: DatasetIdentity) -> Path:
        return self.tokenized_cache_dir / dataset_id.slug()

    def train_mmap_path(self, dataset_id: DatasetIdentity) -> Path:
        return self.dataset_dir(dataset_id) / DATASET_FILES.train_tokens

    def eval_mmap_path(self, dataset_id: DatasetIdentity) -> Path | None:
        if dataset_id.eval_split is None:
            return None
        return self.dataset_dir(dataset_id) / DATASET_FILES.eval_tokens

    def cleanup_if_tmp_dir(self) -> None:
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()

    def _prepare_cache_dir(
        self, cache_dir: Path | None
    ) -> tuple[Path, TemporaryDirectory | None]:
        if cache_dir is not None:
            return cache_dir, None
        tmp_dir = TemporaryDirectory(prefix="scaling-llms-data-")
        return Path(tmp_dir.name), tmp_dir


def get_dataloaders(
    dataset_id: DatasetIdentity,
    dataset_registry: DatasetRegistry,
    dataloader_config: DataLoaderConfig,
    cache_dir: Path | None = None,
    dataset_info: TokenizedDatasetInfo | None = None,
    run: Run | None = None,
) -> dict[str, Any]:

    logger = DataLogger(
        name="DataLoader",
        file_name=str(METADATA_FILES.data_log) if run is not None else None,
        log_dir=run.metadata_dir if run is not None else None,
        level=logging.INFO,
    )

    logger.log_dataset_id(dataset_id)
    logger.log_dataloader_config(dataloader_config)

    # STEP 1: Prepare tmp paths for token memmaps
    cache_paths = LocalCachePaths(cache_dir)
    train_mmap_cache_path = cache_paths.train_mmap_path(dataset_id)
    eval_mmap_cache_path = cache_paths.eval_mmap_path(dataset_id)

    # STEP 2: Create tokenized dataset and register in Data Registry if not already present
    # In DDP, only rank 0 prepares data to avoid race conditions on memmap writes.
    _is_main = is_main_process()
    if _is_main:
        if dataset_registry.dataset_exists(dataset_id):
            logger.log_tokenization(
                "Token memmaps already exist in Data Registry, proceeding to create dataloaders..."
            )
            artifacts_dir = dataset_registry.get_dataset_artifacts(
                dataset_id,
                raise_if_not_found=True,
                pull=True,  # ensure we have the latest artifacts locally
            )
        else:
            logger.log_tokenization(
                "Token memmaps not found in Data Registry, proceeding with local preparation and upload..."
            )
            artifacts_dir, _ = make_tokenized_dataset(
                dataset_id=dataset_id,
                local_train_mmap_path=train_mmap_cache_path,
                local_hf_cache_dir=cache_paths.hf_cache_dir,
                local_eval_mmap_path=eval_mmap_cache_path,
                dataset_registry=dataset_registry,
            )
            cache_paths.cleanup_if_tmp_dir()

    barrier_if_distributed()  # wait for rank 0 to finish data preparation (no-op single GPU)

    if is_distributed() and (not _is_main):
        artifacts_dir = dataset_registry.get_dataset_artifacts(
            dataset_id, 
            raise_if_not_found=True, 
            pull=False  
            # avoid redundant pull in non-main ranks; main rank already pulled during preparation
        )

    # Load dataset info from Data Registry
    dataset_info = dataset_info or dataset_registry.get_dataset_info(dataset_id)
    dtype = np.dtype(dataset_info.dtype)
    logger.log_dataset_info(dataset_info)

    # STEP 4: Create dataloaders
    logger.log_dataloader_creation(
        "Creating deterministic training dataset with random windows "
        "and sequential evaluation dataset with non-overlapping chunks..."
    )
    eval_bin = artifacts_dir.eval_bin if artifacts_dir.eval_bin.exists() else None
    dls = make_dataloaders(
        local_train_mmap_path=artifacts_dir.train_bin,
        dtype=dtype,
        dataloader_config=dataloader_config,
        local_eval_mmap_path=eval_bin,
    )

    eval_info = f" and {len(dls['eval'])} evaluation batches" if dls["eval"] is not None else ""
    logger.log_dataloader_creation(
        f"Prepared dataloaders with {len(dls['train'])} training batches{eval_info}."
    )

    # STEP 5: Prepare output info
    output_info = dict(**asdict(dataset_info), **asdict(dataloader_config))

    return {
        "train": dls["train"],
        "eval": dls["eval"],
        "info": output_info,
    }
