# Training Run Design Doc

## Step 1: Create a Run
- **Input:** `exp_name`, `run_name` (optional: `resume=True`)
- **Main:** `run = registry.start_run(exp_name, run_name)`
- **Output:** `run` — an instance of `RunManager`
- **Artifacts:**
  - Creates the run directory and records run metadata in the registry DB.
  - Directory structure:
    - `experiment_name/run_dir/`
      - `metrics/`
      - `tb/`
      - `checkpoints/`
      - `metadata/`

## Step 2: Load Data
- **Input:** `DataConfig`
- **Main:** `get_dataloaders(cfg)`
- **Output:** `train_dl`, `eval_dl`
- **Artifacts:** token buffers saved as `.bin` memmaps.
  - Relative paths (from dataset config):
    - `<dataset_name>/<dataset_config>/train.bin`
    - `<dataset_name>/<dataset_config>/val.bin`
  - **Permanent storage:** Google Drive data root.
  - **Local cache:**
    - Raw texts: `hf_cache_dir` (HuggingFace cache)
    - Tokenized memmaps: `tokenized_cache_dir`
  - **Cache cap:** `ensure_local_dataset_cache_cap()` enforces `MAX_CACHE_GB` per cache dir.
- **Flow:**
  1. If Drive memmaps exist, use Drive as source of truth.
  2. If local memmaps are missing, copy from Drive to local cache.
  3. If Drive memmaps are missing, build locally, then copy to Drive.
  4. Dataloaders always read from local memmaps for speed.
- **Metadata:** serialize `DataConfig` and store it in `run.metadata`.
  - `run.log_metadata(data_cfg, RUN_FILES.data_config, format="json")`

## Step 3: Initialize Model
- **Input:** `GPTConfig`
- **Main:** `model = GPTModel(cfg)`
- **Output:** `model`
- **Metadata:** serialize `GPTConfig` and store it in `run.metadata`.
  - `run.log_metadata(model_cfg, RUN_FILES.model_config, format="json")`

## Step 4: Initialize Trainer
- **Input:** `TrainerConfig`
- **Main:** `trainer = Trainer(cfg, model, train_dl, eval_dl, run=run)`
- **Output:** `trainer`
- **Metadata:** serialize `TrainerConfig` and store it in `run.metadata`.
  - `run.log_metadata(trainer_cfg, RUN_FILES.trainer_config, format="json")`

## Step 5: Train
- **Main:** `trainer.train()`
- **Artifacts:**
  - **Metrics (JSONL):**
    - `metrics/train.jsonl`
    - `metrics/eval.jsonl`
    - `metrics/network.jsonl`
    - `metrics/system.jsonl`
  - **TensorBoard (optional):**
    - `tb/` event files (train/eval/network/system)
  - **Checkpoints:**
    - `checkpoints/` serialized state (model, optimizer, scaler, lr scheduler, trainer)
