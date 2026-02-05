# Tracking — Key Design Points

## `RunManager`

### Purpose
- Owns and exposes the output directory of a training run
- Centralizes all filesystem I/O (configs, metrics, checkpoints, TensorBoard)
- Keeps `Trainer` free of path-handling and logging logic

---

### Run Directory Structure
- Each run lives in `log_dir/run_<N>/`
- Stable, predictable layout:
  - `metadata/` → run descriptors (config, env, git, notes)
  - `metrics/`  → append-only JSONL streams (train / network / system / eval)
  - `checkpoints/` → resumable training state
  - `tb/` → TensorBoard event files

---

### Metadata
- `trainer_configs.json` logged **once** (resolved `TrainerConfig`)
- Optional files:
  - `env.json` (torch, cuda, device info)
  - `git.json` (commit hash, dirty flag)
  - `notes.txt` (human notes)
- Treated as immutable run descriptors

---

### Metrics
- Logged as JSONL (append-only), one file per category
- Metric categories:
  - `train`   → loss, nll, ppl, lr, tokens_seen_total
  - `network` → grad_norm, param_norm, grad_to_param_ratio, grad_zero_frac
  - `system`  → step_ms, tokens_per_sec, peak_alloc_gb
  - `eval`    → eval_loss, perplexity, validation metric
- Each category has its **own logging frequency**
- Metrics are computed **only if they will be logged**

---

### Checkpoints
- Store only resumable state:
  - model / optimizer / scaler / scheduler state
  - step counters and RNG state
- No full config duplication (config lives in `metadata/`)
- Common files:
  - `last.pt` (latest state)
  - `best.pt` (best by validation metric)
  - `step_<N>.pt` (periodic snapshots)
  - `crash_YYYYMMDD_HHMMSS.pt` (emergency snapshot)

---

### Public API
```python
log_metrics(category, step, metrics)
log_tb(category, step, metrics)
log_metadata(name, obj, format="json")
````

* `category` routes metrics to the correct logger
* JSONL and TensorBoard are independent logging services
* TensorBoard logging is optional and gated by config

---

### Internal Design

* Per-service tracker registries:

```python
_jsonl_tracker_dict: dict[MetricCategory, JsonlTracker]
_tb_tracker_dict:    dict[MetricCategory, TensorBoardTracker]
```

* Each metric category maps to exactly one tracker per service
* Tracker implementations are swappable and isolated

---

### Trainer Integration

* Trainer never touches filesystem paths directly
* Trainer responsibilities:

  1. Decide *what* to log and *when* (based on step and frequency)
  2. Compute the relevant metrics
  3. Delegate logging to RunManager

Example:

```python
from scaling_llms.tracking.constants import METRICS

run.log_metrics(METRICS.train, step, metrics)
run.log_tb(METRICS.train, step, metrics)
```

---

## Trackers

The tracker classes handle all metric and data I/O. Each tracker is responsible for serializing metrics to a specific format/backend.

### `BaseTracker` (abstract)
- Defines the interface all trackers must implement
- Methods: `log_metrics()`, `write()`, `close()`

### `StepTracker`
- Base implementation for step-oriented trackers
- Handles directory creation and `enabled` flag
- Loops through metrics dict and calls `write()` for each entry

### `JsonlTracker`
- Extends `StepTracker`, writes metrics to JSONL files
- One file per category (e.g., `train.jsonl`)
- Each line is: `{"step": int, "metric": str, "value": float}`
- Append-only, easy to stream and parse

### `TensorBoardTracker`
- Extends `StepTracker`, writes to TensorBoard event files
- Requires torch and `torch.utils.tensorboard`
- Uses `SummaryWriter.add_scalar()` for scalar metrics
- Can be disabled gracefully if TensorBoard is unavailable

### `TrackerConfig`
- Dataclass specifying how to instantiate a tracker
- Fields: `cls`, `enabled`, `log_dir`, `name`, `kwargs`
- Used to create trackers with consistent configuration

### `TrackerDict`
- Manages multiple trackers (one per metric category)
- Implements `Mapping` interface for dict-like access
- Routes `log_metrics()` calls to the correct tracker
- Centralized `close()` to clean up all trackers

### `JsonlTrackerReader`
- Reads JSONL logs produced by `JsonlTracker`
- Exposes logs as pandas DataFrames
- Caches loaded data for performance
- Methods: `__getitem__()`, `get()`, `keys()`, `reload()`

---

### Design Principles

* Clear separation of concerns (training vs I/O)
* Append-only, machine-readable metrics
* Deterministic resume via checkpoints
* Minimal, explicit APIs
* Script- and diff-friendly artifacts

