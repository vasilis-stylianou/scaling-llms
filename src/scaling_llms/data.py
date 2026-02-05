from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from datasets import load_dataset
import itertools
import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# DATA SOURCES
# -----------------------------
@dataclass(frozen=True)
class DataSources:
    """Available data sources for DataManager."""
    tiny_shakespeare: str = "tiny_shakespeare"
    wikitext103: str = "wikitext103"
    openwebtext: str = "openwebtext"

# Singleton instance
DATA_SOURCES = DataSources()


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def infinite_dataloader(loader):
    """Yield batches from `loader` indefinitely.

    Useful for training loops that want an endless stream of batches.
    """
    while True:
        for batch in loader:
            yield batch


def train_val_split_1d(data: torch.Tensor, val_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    assert data.ndim == 1
    assert 0.0 < val_frac < 1.0
    n = data.numel()
    n_val = int(n * val_frac)
    train = data[:-n_val]
    val = data[-n_val:]
    return train, val


def load_dataset_bytes(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str = "train",
    max_samples: int | None = None,
    text_separator: str = "\n",
) -> torch.Tensor:
    """Load a HuggingFace dataset and return as token IDs (0-255 bytes).
    
    Requires: pip install datasets
    
    Args:
        dataset_name: name of the HuggingFace dataset
        dataset_config: optional config name (e.g., "wikitext-103-v1")
        split: dataset split to load ("train", "validation", "test")
        max_samples: maximum number of documents to load (None = all)
        text_separator: string to join text documents with
    
    Returns:
        1D tensor of token IDs (byte values 0-255)
    """
    
    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # Apply max_samples limit if specified
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    # Concatenate all text and encode as bytes
    text = text_separator.join(dataset["text"])
    data = text.encode("utf-8")
    x = torch.frombuffer(data, dtype=torch.uint8).clone().to(torch.int64)
    
    return x


def load_tiny_shakespeare_direct(data_dir: str = "./data") -> torch.Tensor:
    """Download and load Tiny Shakespeare from raw URL.
    
    Uses direct download since HF dataset uses deprecated loading script.
    
    Args:
        data_dir: directory to store the downloaded file
    
    Returns:
        1D tensor of token IDs (byte values 0-255)
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    
    # Download if not exists
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    
    # Load and encode
    with open(path, "rb") as f:
        data = f.read()
    x = torch.frombuffer(data, dtype=torch.uint8).clone().to(torch.int64)
    
    return x


# -----------------------------
# RANDOM WINDOW LM DATASET
# -----------------------------
class RandomWindowLM(Dataset):
    def __init__(
        self, 
        tokens_1d: torch.Tensor, 
        seq_len: int, 
        generator: torch.Generator | None = None
    ):
        assert tokens_1d.ndim == 1
        assert tokens_1d.numel() > seq_len + 1
        self.tokens = tokens_1d
        self.seq_len = seq_len
        # Optional per-dataset generator to make sampling reproducible
        self.generator = generator

    def __len__(self) -> int:
        # arbitrary: number of possible windows (not used much since we sample randomly)
        return self.tokens.numel() - (self.seq_len + 1)

    def __getitem__(self, idx: int):
        # ignore idx, sample random window
        max_start = self.tokens.numel() - (self.seq_len + 1)
        if self.generator is not None:
            start = torch.randint(0, max_start, (1,), generator=self.generator).item()
        else:
            start = torch.randint(0, max_start, (1,)).item()
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = chunk[:-1]  # [T]
        y = chunk[1:]   # [T]
        return x, y


# -----------------------------
# DATA MANAGER
# -----------------------------
class DataManager:
    """Factory for dataset/data-loader creation across multiple sources.

    Usage:
      dm = DataManager(data_dir="./data", seed=1234)
      train_iter, val_iter = dm.get_loaders(DATA_SOURCES.tiny_shakespeare, seq_len=512, batch_size=8, as_iterable=True, resume_step=100)

    The manager maintains a registry of load functions. Each source has a load function
    that returns a 1D token tensor. Common DataLoader creation logic is handled internally.
    """

    def __init__(self, data_dir: str = "./data", seed: int = 1337):
        self.data_dir = data_dir
        self.seed = int(seed)
        # registry maps source name -> (load_fn, kwargs)
        self._registry = {
            DATA_SOURCES.tiny_shakespeare: {
                # Uses direct download, not HF datasets
            },
            DATA_SOURCES.wikitext103: {
                "dataset_name": "wikitext",
                "dataset_config": "wikitext-103-v1",
                "text_separator": "\n",
            },
            DATA_SOURCES.openwebtext: {
                "dataset_name": "Skylion007/openwebtext",  # Parquet-based version
                "text_separator": "\n\n",
            },
        }

    # --- API ---
    def list_data_sources(self) -> list[str]:
        """List all registered data sources."""
        return list(self._registry.keys())

    def get_loaders(
        self,
        source: str,
        *,
        seq_len: int = 512,
        batch_size: int = 16,
        val_frac: float = 0.1,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        as_iterable: bool = False,
        resume_step: int = 0,
        seed: int | None = None,
        # Source-specific kwargs
        max_samples: int | None = None,  # for openwebtext
        val_split: str = "validation",   # for wikitext103
    ) -> tuple[Iterator, Iterator] | tuple[DataLoader, DataLoader]:
        """Create loaders for `source`.

        - `as_iterable=True` returns infinite iterators for train/val.
        - `resume_step` advances the train iterator by `resume_step` batches.
        - `seed` overrides the manager seed for deterministic sampling.
        - `max_samples` limits the number of documents (openwebtext only).
        - `val_split` specifies validation split (wikitext103: "validation" or "test").
        """
        if source not in self._registry:
            raise ValueError(f"Unknown data source: {source}")

        config = self._registry[source]
        use_seed = self.seed if seed is None else int(seed)

        # Load dataset with source-specific configuration
        load_kwargs = config.copy()
        
        # Handle source-specific parameters
        if source == DATA_SOURCES.tiny_shakespeare:
            # Use direct download (HF dataset uses deprecated loading script)
            tokens = load_tiny_shakespeare_direct(data_dir=self.data_dir)
            train_loader, val_loader = self._create_loaders(
                tokens, val_frac, seq_len, batch_size,
                num_workers, pin_memory, persistent_workers, use_seed
            )
        elif source == DATA_SOURCES.wikitext103:
            # WikiText-103 has pre-split train/validation/test sets
            train_tokens = load_dataset_bytes(**load_kwargs, split="train")
            val_tokens = load_dataset_bytes(**load_kwargs, split=val_split)
            # For wikitext, we already have separate train/val so no split needed
            torch.manual_seed(use_seed)
            gen = torch.Generator()
            gen.manual_seed(use_seed)
            
            train_ds = RandomWindowLM(train_tokens, seq_len=seq_len, generator=gen)
            val_ds = RandomWindowLM(val_tokens, seq_len=seq_len, generator=gen)
            
            dl_kwargs = dict(
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(persistent_workers and num_workers > 0),
            )
            train_loader = DataLoader(train_ds, **dl_kwargs)
            val_loader = DataLoader(val_ds, **dl_kwargs)
        elif source == DATA_SOURCES.openwebtext:
            # Skylion007/openwebtext is large, apply max_samples limit
            tokens = load_dataset_bytes(**load_kwargs, split="train", max_samples=max_samples or 10000)
            train_loader, val_loader = self._create_loaders(
                tokens, val_frac, seq_len, batch_size,
                num_workers, pin_memory, persistent_workers, use_seed
            )
        else:
            # Generic: tiny_shakespeare and custom sources
            tokens = load_dataset_bytes(**load_kwargs, split="train")
            train_loader, val_loader = self._create_loaders(
                tokens, val_frac, seq_len, batch_size,
                num_workers, pin_memory, persistent_workers, use_seed
            )

        if not as_iterable:
            return train_loader, val_loader

        train_iter = infinite_dataloader(train_loader)
        val_iter = infinite_dataloader(val_loader)

        # Resume by advancing the iterator efficiently
        if resume_step and resume_step > 0:
            # efficient skip: consume `resume_step` items without storing
            deque(itertools.islice(train_iter, resume_step), maxlen=0)

        return train_iter, val_iter

    # --- PRIVATE HELPERS ---
    def _create_loaders(
        self,
        tokens: torch.Tensor,
        val_frac: float,
        seq_len: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Common logic to create train/val DataLoaders from token tensor."""
        torch.manual_seed(seed)
        
        # Split into train/val
        train_tokens, val_tokens = train_val_split_1d(tokens, val_frac=val_frac)
        
        # Create generator for reproducible sampling
        gen = torch.Generator()
        gen.manual_seed(seed)
        
        # Create datasets
        train_ds = RandomWindowLM(train_tokens, seq_len=seq_len, generator=gen)
        val_ds = RandomWindowLM(val_tokens, seq_len=seq_len, generator=gen)
        
        # Create loaders
        dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
        )
        train_loader = DataLoader(train_ds, **dl_kwargs)
        val_loader = DataLoader(val_ds, **dl_kwargs)
        
        return train_loader, val_loader