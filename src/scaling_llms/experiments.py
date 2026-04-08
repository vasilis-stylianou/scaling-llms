# experiments.py

from __future__ import annotations

import json
import pandas as pd
from typing import Any

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from scaling_llms.checkpointing import instantiate_model_from_run
from scaling_llms.constants import CKPT_FILES, METADATA_FILES
from scaling_llms.data import DataLoaderConfig, get_dataloaders, get_vocab_size
from scaling_llms.distributed import barrier_if_distributed, is_distributed, is_main_process
from scaling_llms.models import GPTConfig, GPTModel
from scaling_llms.registries import (
    DatasetIdentity,
    DatasetRegistry,
    RunIdentity,
    RunRegistry,
)
from scaling_llms.tracking.run import Run
from scaling_llms.trainer import Trainer, TrainerConfig
from scaling_llms.utils.training import make_lr_scheduler, set_determinism

# =============================================================================
# Shared functions
# =============================================================================
def build_raw_model(
    *,
    seq_len: int,
    vocab_size: int,
    gpt_hparams: dict[str, Any],
) -> GPTModel:
    model_cfg = GPTConfig(
        seq_len=seq_len,
        vocab_size=vocab_size,
        **gpt_hparams,
    )
    return GPTModel(model_cfg)


def build_trainer(
    cfg: TrainerConfig,
    raw_model: torch.nn.Module,
    train_dl=None,
    eval_dl=None,
    run=None,
) -> Trainer:
    """
    Owns the full setup order:
      1. move raw_model to device
      2. wrap with DDP if is distributed
      3. compile if use_compile
      4. create optimizer, scaler, scheduler
      5. construct Trainer
    """
    # Ensure determinism before any GPU work 
    # (e.g. in DDP all models must have same initial weights)
    set_determinism(cfg.seed) 
    
    # 1. Move to device
    raw_model.to(cfg.device)

    # 2. DDP
    model = raw_model
    if is_distributed():
        model = DDP(model, device_ids=[cfg.local_rank])

    # 3. Compile
    if cfg.use_compile:
        model = torch.compile(model)

    # 4. Training objects
    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.precision == "fp16") and cfg.device.startswith("cuda")
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    lr_scheduler = make_lr_scheduler(optimizer, cfg)

    # 5. Trainer
    return Trainer(
        cfg=cfg,
        model=model,
        raw_model=raw_model,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        train_dl=train_dl,
        eval_dl=eval_dl,
        run=run,
    )


def load_training_state(
    ckpt: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    lr_scheduler: Any,
    device: str,
) -> dict[str, Any]:
    """
    Restore optimizer/scaler/lr_scheduler/trainer state into already-constructed
    training objects. Assumes ckpt was loaded with map_location='cpu'.
    """

    # Optimizer — restore state, then move tensors to device manually
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # Scaler
    if (
        scaler is not None
        and scaler.is_enabled()
        and ckpt.get("scaler") is not None
    ):
        scaler.load_state_dict(ckpt["scaler"])

    # LR Scheduler
    if lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    return ckpt.get("trainer", {})


def load_trainer_from_checkpoint(
    ckpt_run: Run,
    ckpt_filename: str,
    reset_state: bool = False,
    active_run: Run | None = None,
) -> Trainer:
    """
    Load a checkpoint from disk and restore model weights, optimizer state, etc.

    Returns a Trainer with model moved to device and ready for training or evaluation.

    If reset_state=True, only model weights are loaded; optimizer/scaler/scheduler 
    state is ignored and left at initial values.
    """
    cfg = TrainerConfig.from_json(
        ckpt_run.artifacts_dir.metadata_path(METADATA_FILES.trainer_config)
    )

    # 1. Instantiate raw model on CPU
    raw_model = instantiate_model_from_run(ckpt_run)

    # 2. Read checkpoint once on CPU
    ckpt_path = ckpt_run.artifacts_dir.checkpoint_path(ckpt_filename)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 3. Load model weights into raw CPU model
    raw_model.load_state_dict(ckpt["model"])
    
    # 4. Build trainer; this moves model to device and wraps DDP/compile
    trainer = build_trainer(
        cfg=cfg,
        raw_model=raw_model,
        run=active_run,
    )

    # 5) Restore optimizer / scaler / scheduler / trainer state from checkpoint
    # and move to device.
    if not reset_state:
        trainer_state = load_training_state(
            ckpt,
            optimizer=trainer.optimizer,
            scaler=trainer.scaler,
            lr_scheduler=trainer.lr_scheduler,
            device=cfg.device,
        )
        trainer.load_state_dict(trainer_state)

    return trainer


def init_trainer(
    *,
    dataset_registry: DatasetRegistry,
    dataset_id: DatasetIdentity,
    dl_cfg: DataLoaderConfig,
    trainer_cfg: TrainerConfig,
    gpt_hparams: dict[str, Any],
    run: Run | None = None,
) -> Trainer:
    # NOTE: in DDP only rank 0 prepares 
    dl_dict = get_dataloaders(
        dataset_id=dataset_id,
        dataset_registry=dataset_registry,
        dataloader_config=dl_cfg,
        run=run, # if not None, used for printing logs 
    )
    if run is not None:
        run.log_metadata(dataset_id, METADATA_FILES.dataset_id, format="json")
        run.log_metadata(dl_cfg, METADATA_FILES.dataloader_config, format="json")

    raw_model = build_raw_model(
        seq_len=dl_cfg.seq_len,
        vocab_size=dl_dict["info"]["vocab_size"],
        gpt_hparams=gpt_hparams,
    )

    return build_trainer(
        cfg=trainer_cfg,
        raw_model=raw_model,
        train_dl=dl_dict["train"],
        eval_dl=dl_dict["eval"],
        run=run,
    )


def build_trainer_from_checkpoint(
    *,
    ckpt_run: Run,
    dataset_registry: DatasetRegistry,
    ckpt_filename: str,
    reset_state: bool,
    active_run: Run | None = None,
) -> Trainer:
    # Load trainer from checkpoint, which also moves model to device and wraps DDP/compile
    trainer = load_trainer_from_checkpoint(ckpt_run, ckpt_filename, reset_state=reset_state, active_run=active_run)

    # Build dataloaders and attach to trainer
    dataset_id = DatasetIdentity.from_json(
        path=ckpt_run.artifacts_dir.metadata_path(METADATA_FILES.dataset_id)
    )
    dl_cfg = DataLoaderConfig.from_json(
        path=ckpt_run.artifacts_dir.metadata_path(METADATA_FILES.dataloader_config),
        overwrite_data={"start_sample_idx": trainer.consumed_samples},
    )
    dl_dict = get_dataloaders(
        dataset_id=dataset_id,
        dataset_registry=dataset_registry,
        dataloader_config=dl_cfg,
        run=active_run,
    )
    trainer.attach_dataloaders(dl_dict["train"], dl_dict["eval"])

    return trainer


def validate_checkpoint_compatibility(
    *,
    ckpt_run: Run,
    target_seq_len: int,
    target_vocab_size: int,
) -> None:
    old_dl_kwargs = json.loads(
        ckpt_run.artifacts_dir.metadata_path(METADATA_FILES.dataloader_config).read_text()
    )
    if old_dl_kwargs["seq_len"] != target_seq_len:
        raise ValueError(
            f"Sequence length mismatch between old run ({old_dl_kwargs['seq_len']}) "
            f"and new run ({target_seq_len}). "
            "Checkpoint loading may fail."
        )

    old_model_kwargs = json.loads(
        ckpt_run.artifacts_dir.metadata_path(METADATA_FILES.model_config).read_text()
    )
    if old_model_kwargs["vocab_size"] != target_vocab_size:
        raise ValueError(
            f"Vocab size mismatch between old run ({old_model_kwargs['vocab_size']}) "
            f"and new run ({target_vocab_size}). "
            "Checkpoint loading may fail."
        )


def build_trainer_from_checkpoint_transfer(
    *,
    ckpt_run: Run,
    ckpt_filename: str,
    dataset_registry: DatasetRegistry,
    dataset_kwargs: dict[str, Any],
    dataloader_kwargs: dict[str, Any],
    active_run: Run | None = None,
) -> Trainer:
    
    # Load model weights only and reset training state
    trainer = load_trainer_from_checkpoint(
        ckpt_run=ckpt_run,
        ckpt_filename=ckpt_filename,
        reset_state=True, # weights only; skip loading training state
        active_run=active_run,
    )

    # Build dataloaders and attach to trainer
    dataset_id = DatasetIdentity(**dataset_kwargs)
    dl_cfg = DataLoaderConfig(**dataloader_kwargs)
    dl_dict = get_dataloaders(
        dataset_id=dataset_id,
        dataset_registry=dataset_registry,
        dataloader_config=dl_cfg,
        run=active_run,
    )
    trainer.attach_dataloaders(dl_dict["train"], dl_dict["eval"])

    if active_run is not None:
        active_run.log_metadata(dataset_id, METADATA_FILES.dataset_id, format="json")
        active_run.log_metadata(dl_cfg, METADATA_FILES.dataloader_config, format="json")

    return trainer

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
class RunMethod(StrEnum):
    START = "start"
    RESUME = "resume"
    START_FROM_CHECKPOINT = "start_from_checkpoint"


@dataclass(slots=True)
class RunConfig:
    method: RunMethod
    run_name: str

    # start
    dataset_kwargs: dict[str, Any] | None = None
    dataloader_kwargs: dict[str, Any] | None = None
    gpt_hparams: dict[str, Any] | None = None
    trainer_kwargs: dict[str, Any] | None = None
    overwrite: bool = False
    ignore_if_run_exists: bool = False

    # resume / checkpoint
    ckpt_filename: str = CKPT_FILES.best_ckpt
    ckpt_exp_name: str | None = None
    ckpt_run_name: str | None = None

    # shared
    max_steps: int | None = None

    def __post_init__(self):
        self._validate()
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        if not isinstance(data, dict):
            raise ValueError("run config must be a dict")

        def _req(key: str):
            if key not in data:
                raise ValueError(f"missing required key: {key}")
            return data[key]

        def _opt_dict(key: str) -> dict[str, Any] | None:
            val = data.get(key)
            if val is None:
                return None
            if not isinstance(val, dict):
                raise ValueError(f"{key} must be a dict when provided")
            return val

        def _opt_str(key: str) -> str | None:
            val = data.get(key)
            if val is None:
                return None
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{key} must be a non-empty string when provided")
            return val.strip()

        def _bool(key: str, default: bool = False) -> bool:
            val = data.get(key, default)
            if not isinstance(val, bool):
                raise ValueError(f"{key} must be a boolean")
            return val

        def _pos_int_or_none(key: str) -> int | None:
            val = data.get(key)
            if val is None:
                return None
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{key} must be a positive int or None")
            return val

        # method
        try:
            method = RunMethod(_req("method"))
        except Exception as exc:
            allowed = ", ".join(m.value for m in RunMethod)
            raise ValueError(f"method must be one of: {allowed}") from exc

        # run_name
        run_name = _req("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError("run_name must be a non-empty string")

        ckpt_filename = data.get("ckpt_filename", CKPT_FILES.best_ckpt)
        if ckpt_filename is not None:
            if not isinstance(ckpt_filename, str) or not ckpt_filename.strip():
                raise ValueError("ckpt_filename must be a non-empty string")

        return cls(
            method=method,
            run_name=run_name.strip(),
            dataset_kwargs=_opt_dict("dataset_kwargs"),
            dataloader_kwargs=_opt_dict("dataloader_kwargs"),
            gpt_hparams=_opt_dict("gpt_hparams"),
            trainer_kwargs=_opt_dict("trainer_kwargs"),
            overwrite=_bool("overwrite", False),
            ignore_if_run_exists=_bool("ignore_if_run_exists", False),
            ckpt_filename=ckpt_filename,
            ckpt_exp_name=_opt_str("ckpt_exp_name"),
            ckpt_run_name=_opt_str("ckpt_run_name"),
            max_steps=_pos_int_or_none("max_steps"),
        )


    def _validate(self) -> None:
        required_by_method = {
            RunMethod.START: (
                "dataset_kwargs",
                "dataloader_kwargs",
                "gpt_hparams",
                "trainer_kwargs",
            ),
            RunMethod.RESUME: (
                "ckpt_filename",
            ),
            RunMethod.START_FROM_CHECKPOINT: (
                "dataset_kwargs",
                "dataloader_kwargs",
                "ckpt_exp_name",
                "ckpt_run_name",
                "ckpt_filename",
            ),
        }

        if self.method not in required_by_method:
            raise ValueError(f"Unsupported method: {self.method}")

        missing = [
            field_
            for field_ in required_by_method[self.method]
            if getattr(self, field_) is None
        ]

        if missing:
            raise ValueError(
                f"{self.method.value} requires: {', '.join(missing)}"
            )


class ExperimentRunner:

    def __init__(
        self,
        *,
        exp_name: str,
        run_registry: RunRegistry,
        dataset_registry: DatasetRegistry,
    ) -> None:
        self.exp_name = exp_name
        self.run_registry = run_registry
        self.dataset_registry = dataset_registry

    # --- PRIVATE METHODS ---
    @contextmanager
    def _managed_run(self, identity: RunIdentity, **kwargs):
        """
        On main process: enters run_registry.managed_run and yields the Run.
        On worker processes: yields None with no context overhead.
        """
        if is_main_process():
            with self.run_registry.managed_run(identity, **kwargs) as run:
                yield run
        else:
            yield None

    def _train(
        self, 
        identity: RunIdentity, 
        trainer: Trainer, 
        max_steps: int | None,
    ) -> None:
        _is_main = is_main_process()
        if _is_main:
            self.run_registry.set_device_name(identity, trainer.cfg.device_name)

            other_data = trainer.get_runtime_info()
            other_data.update({f"start_{k}": v for k, v in trainer.state_dict().items()})
            other_data["max_steps"] = max_steps
            other_data["num_steps"] = trainer.cfg.num_steps
            other_data["accum_steps"] = trainer.cfg.accum_steps
            other_data["train_micro_batch_size"] = trainer.train_dl.batch_size
            other_data["train_global_batch_size"] = (
                trainer.cfg.accum_steps 
                * trainer.train_dl.batch_size
                * other_data["world_size"]
            )

        barrier_if_distributed()
        trainer.train(max_steps=max_steps)
        barrier_if_distributed()

        if _is_main:
            other_data.update({f"end_{k}": v for k, v in trainer.state_dict().items()})
            self.run_registry.set_other_data(identity, other_data)

    # --- PUBLIC METHODS ---
    def start(
        self,
        run_name: str,
        *,
        dataset_kwargs: dict[str, Any],
        dataloader_kwargs: dict[str, Any],
        gpt_hparams: dict[str, Any],
        trainer_kwargs: dict[str, Any],
        max_steps: int | None = None,
        overwrite: bool = False,
        ignore_if_run_exists: bool = False,
    ) -> Trainer | None:

        identity = RunIdentity(self.exp_name, run_name)
        if ignore_if_run_exists and self.run_registry.run_exists(identity):
            if is_main_process():
                print(
                    f"Run {self.exp_name}/{run_name} already exists, "
                    "but ignore_if_run_exists=True, so ignoring and proceeding."
                )
            return None

        dataset_id = DatasetIdentity(**dataset_kwargs)
        dl_cfg = DataLoaderConfig(**dataloader_kwargs)
        trainer_cfg = TrainerConfig(**trainer_kwargs)

        with self._managed_run(
            identity,
            resume=False,
            overwrite=overwrite,
        ) as run:
            trainer = init_trainer(
                dataset_registry=self.dataset_registry,
                dataset_id=dataset_id,
                dl_cfg=dl_cfg,
                trainer_cfg=trainer_cfg,
                gpt_hparams=gpt_hparams,
                run=run, # for rank >0 pass None since only main process creates the run
            )
            
            self._train(identity, trainer, max_steps)

        return trainer

    def resume(
        self,
        run_name: str,
        *,
        ckpt_filename: str = CKPT_FILES.best_ckpt,
        max_steps: int | None = None,
    ) -> Trainer:
        identity = RunIdentity(self.exp_name, run_name)
        with self._managed_run(
            identity,
            resume=True,
            overwrite=False,
        ) as active_run:
            ckpt_run = (
                active_run
                if is_main_process()
                else self.run_registry.get_run(identity, pull=False)
                # NOTE: non-main processes skip pulling artifacts 
                # since main process will handle all artifact syncing
            )
            trainer = build_trainer_from_checkpoint(
                ckpt_run=ckpt_run,
                dataset_registry=self.dataset_registry,
                ckpt_filename=ckpt_filename,
                reset_state=False,
                active_run=active_run,
            )
            self._train(identity, trainer, max_steps)

        return trainer

    def start_from_checkpoint(
        self,
        run_name: str,
        *,
        dataset_kwargs: dict[str, Any],
        dataloader_kwargs: dict[str, Any],
        ckpt_exp_name: str,
        ckpt_run_name: str,
        ckpt_filename: str,
        max_steps: int | None = None,
        overwrite: bool = False,
        ignore_if_run_exists: bool = False,
    ) -> Trainer | None:
        _is_main = is_main_process()

        # Validate run existence 
        identity = RunIdentity(self.exp_name, run_name)
        if ignore_if_run_exists and self.run_registry.run_exists(identity):
            if _is_main:
                print(
                    f"Run {self.exp_name}/{run_name} already exists, "
                    "but ignore_if_run_exists=True, so ignoring and proceeding."
                )
            return None
        
        # Validate checkpoint compatibility 
        ckpt_run = self.run_registry.get_run(
            RunIdentity(ckpt_exp_name, ckpt_run_name), 
            pull=_is_main
            # NOTE: non-main processes skip pulling artifacts 
            # since main process will handle all artifact syncing
        ) 
        if _is_main:
            validate_checkpoint_compatibility(
                ckpt_run=ckpt_run,
                target_seq_len=dataloader_kwargs["seq_len"],
                target_vocab_size=get_vocab_size(dataset_kwargs["tokenizer_name"]),
            )
        
        barrier_if_distributed()
        
        # Load model weights from checkpoint, init trainer, and train. 
        # Only main process manages the run; other processes just init trainer and train.
        with self._managed_run(
            identity,
            resume=False,
            overwrite=overwrite,
        ) as active_run:
            if active_run is not None: # active_run is None on non-main processes
                active_run.log_metadata(
                    {
                        "initialized_from": {
                            "exp": ckpt_exp_name,
                            "run": ckpt_run_name,
                            "ckpt": ckpt_filename,
                        }
                    },
                    "source_checkpoint.json",
                    format="json",
                )

            trainer = build_trainer_from_checkpoint_transfer(
                ckpt_run=ckpt_run,
                ckpt_filename=ckpt_filename,
                dataset_registry=self.dataset_registry,
                dataset_kwargs=dataset_kwargs,
                dataloader_kwargs=dataloader_kwargs,
                active_run=active_run,
            )
            self._train(identity, trainer, max_steps)

        return trainer

    # --- MAIN ENTRYPOINT ---
    def run(self, config: RunConfig) -> Trainer | None:

        if config.method == RunMethod.START:
            return self.start(
                run_name=config.run_name,
                dataset_kwargs=config.dataset_kwargs,
                dataloader_kwargs=config.dataloader_kwargs,
                gpt_hparams=config.gpt_hparams,
                trainer_kwargs=config.trainer_kwargs,
                max_steps=config.max_steps,
                overwrite=config.overwrite,
                ignore_if_run_exists=config.ignore_if_run_exists,
            )

        if config.method == RunMethod.RESUME:
            return self.resume(
                run_name=config.run_name,
                ckpt_filename=config.ckpt_filename,
                max_steps=config.max_steps,
            )

        if config.method == RunMethod.START_FROM_CHECKPOINT:
            return self.start_from_checkpoint(
                run_name=config.run_name,
                dataset_kwargs=config.dataset_kwargs,
                dataloader_kwargs=config.dataloader_kwargs,
                ckpt_exp_name=config.ckpt_exp_name,
                ckpt_run_name=config.ckpt_run_name,
                ckpt_filename=config.ckpt_filename,
                max_steps=config.max_steps,
                overwrite=config.overwrite,
                ignore_if_run_exists=config.ignore_if_run_exists,
            )

        raise ValueError(f"Unsupported method: {config.method}")


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    runs: list[RunConfig]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")

        if not isinstance(self.runs, list) or not self.runs:
            raise ValueError("runs must be a non-empty list")

        validated_runs: list[RunConfig] = []
        for idx, run_cfg in enumerate(self.runs):
            if isinstance(run_cfg, dict):
                try:
                    run_cfg = RunConfig.from_dict(run_cfg)
                except Exception as exc:
                    raise ValueError(f"Invalid RUNS[{idx}]: {exc}") from exc
            elif not isinstance(run_cfg, RunConfig):
                raise ValueError(f"RUNS[{idx}] must be a RunConfig or dict")

            validated_runs.append(run_cfg)

        self.experiment_name = self.experiment_name.strip()
        self.runs = validated_runs

    @classmethod
    def from_module(cls, config_module: Any) -> "ExperimentConfig":
        if not hasattr(config_module, "EXPERIMENT_NAME"):
            raise ValueError("Config module must define EXPERIMENT_NAME")
        if not hasattr(config_module, "RUNS"):
            raise ValueError("Config module must define RUNS")

        return cls(
            experiment_name=config_module.EXPERIMENT_NAME,
            runs=config_module.RUNS,
        )

    
    @classmethod
    def load_from_file(cls, file_path: str) -> "ExperimentConfig":
        import importlib.util
        from pathlib import Path

        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Experiment config file not found: {path}")

        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load experiment config from: {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return cls.from_module(module)
    

# -------------------------------------------------------------------------------------
# TODO: 
# -------------------------------------------------------------------------------------
class ExperimentManager:
    def __init__(self, *, run_registry: RunRegistry) -> None:
        self.run_registry = run_registry

    def delete_experiment(self, exp_name: str, confirm: bool = True) -> None:
        if confirm:
            response = input(
                f"Are you sure you want to delete experiment '{exp_name}' and all its runs? "
                "Type 'y' or 'yes' to confirm: "
            )
            if response.strip().lower() not in ("y", "yes"):
                print("Deletion cancelled.")
                return

        runs_df = self.get_experiment_runs(exp_name)
        for run_name in runs_df["run_name"].tolist():
            self.run_registry.delete_run(
                RunIdentity(exp_name, run_name),
                confirm=False,
            )

    def delete_run(self, exp_name: str, run_name: str, confirm: bool = True) -> None:
        self.run_registry.delete_run(
            RunIdentity(exp_name, run_name),
            confirm=confirm,
        )

    def get_experiment_runs(self, exp_name: str) -> pd.DataFrame:
        return self.run_registry.get_runs_as_df(exp_name)

    def list_experiments(self) -> list[str]:
        runs_df = self.run_registry.get_runs_as_df()
        if "experiment_name" not in runs_df.columns:
            return []
        return sorted(runs_df["experiment_name"].dropna().unique().tolist())

    # def delete_experiment(self, experiment_name: str) -> None:
    #     self.execute(
    #         f"DELETE FROM {self.table_name} WHERE experiment_name=:experiment_name",
    #         {"experiment_name": experiment_name},
    #     )


    # def get_experiment_dir(self, experiment_name: str) -> Path:
    #     # TODO: validation?
    #     return self.artifacts.root / experiment_name


    # def get_git_commit(self, identity: RunIdentity) -> str | None:
    #     return self.metadata.get_git_commit(identity)

    # def set_status(self, identity: RunIdentity, status: RunStatus) -> None:
    #     self._set_status_value(identity, status.value)

    # def _set_status_value(self, identity: RunIdentity, status_value: str) -> None:
    #     self.metadata.set_status_value(identity, status_value)

    
    # def delete_experiment(self, experiment_name: str, confirm: bool = True) -> None:
    #     self.get_experiment_dir(experiment_name)

    #     if confirm:
    #         response = input(
    #             f"Are you sure you want to delete experiment '{experiment_name}'? "
    #             "Type 'y' or 'yes' to confirm: "
    #         )
    #         if response.strip().lower() not in ("y", "yes"):
    #             print("Deletion cancelled.")
    #             return

    #     self.artifacts.delete_experiment_dir(experiment_name)
    #     self.metadata.delete_experiment(experiment_name)
