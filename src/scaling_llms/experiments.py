# experiments.py

from __future__ import annotations

import json
import pandas as pd
from typing import Any


from dataclasses import dataclass
from enum import StrEnum

from scaling_llms.constants import (
    CKPT_FILES,
    METADATA_FILES,
)
from scaling_llms.data import DataLoaderConfig, get_dataloaders
from scaling_llms.models import GPTConfig, GPTModel
from scaling_llms.registries import (
    DatasetIdentity,
    DatasetRegistry,
    RunIdentity,
    RunRegistry,
)
from scaling_llms.trainer import Trainer, TrainerConfig


# =============================================================================
# Shared functions
# =============================================================================
def build_model(
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


def init_trainer(
    *,
    run: Any,
    dataset_registry: DatasetRegistry,
    dataset_id: DatasetIdentity,
    dl_cfg: DataLoaderConfig,
    trainer_cfg: TrainerConfig,
    gpt_hparams: dict[str, Any],
) -> Trainer:
    dl_dict = get_dataloaders(
        dataset_id=dataset_id,
        dataset_registry=dataset_registry,
        dataloader_config=dl_cfg,
        run=run,
    )

    run.log_metadata(dataset_id, METADATA_FILES.dataset_id, format="json")
    run.log_metadata(dl_cfg, METADATA_FILES.dataloader_config, format="json")

    model = build_model(
        seq_len=dl_cfg.seq_len,
        vocab_size=dl_dict["info"]["vocab_size"],
        gpt_hparams=gpt_hparams,
    )

    return Trainer(
        cfg=trainer_cfg,
        model=model,
        train_dl=dl_dict["train"],
        eval_dl=dl_dict["eval"],
        run=run,
    )


def trainer_from_checkpoint(
    *,
    run: Any,
    dataset_registry: DatasetRegistry,
    ckpt_filename: str,
    reset_state: bool,
) -> Trainer:
    trainer = Trainer.from_checkpoint(
        run,
        ckpt_name=ckpt_filename,
        reset_state=reset_state,
    )

    dataset_id = DatasetIdentity.from_json(
        path=run.artifacts_dir.metadata_path(METADATA_FILES.dataset_id)
    )
    dl_cfg = DataLoaderConfig.from_json(
        path=run.artifacts_dir.metadata_path(METADATA_FILES.dataloader_config),
        overwrite_data={"start_sample_idx": trainer.step_idx},
    )

    dl_dict = get_dataloaders(
        dataset_id=dataset_id,
        dataset_registry=dataset_registry,
        dataloader_config=dl_cfg,
        run=run,
    )

    trainer.attach_dataloaders(dl_dict["train"], dl_dict["eval"])
    return trainer


def validate_checkpoint_compatibility(
    *,
    source_run: Any,
    target_dataloader_kwargs: dict[str, Any],
    target_dl_info: dict[str, Any],
) -> None:
    old_dl_kwargs = json.loads(
        source_run.artifacts_dir.metadata_path(METADATA_FILES.dataloader_config).read_text()
    )
    if old_dl_kwargs["seq_len"] != target_dataloader_kwargs["seq_len"]:
        raise ValueError(
            f"Sequence length mismatch between old run ({old_dl_kwargs['seq_len']}) "
            f"and new run ({target_dataloader_kwargs['seq_len']}). "
            "Checkpoint loading may fail."
        )

    old_model_kwargs = json.loads(
        source_run.artifacts_dir.metadata_path(METADATA_FILES.model_config).read_text()
    )
    if old_model_kwargs["vocab_size"] != target_dl_info["vocab_size"]:
        raise ValueError(
            f"Vocab size mismatch between old run ({old_model_kwargs['vocab_size']}) "
            f"and new run ({target_dl_info['vocab_size']}). "
            "Checkpoint loading may fail."
        )


def build_transfer_dataloaders(
    *,
    run: Any,
    dataset_registry: DatasetRegistry,
    dataset_kwargs: dict[str, Any],
    dataloader_kwargs: dict[str, Any],
) -> tuple[DatasetIdentity, DataLoaderConfig, dict[str, Any]]:
    
    dataset_id = DatasetIdentity(**dataset_kwargs)
    dl_cfg = DataLoaderConfig(**dataloader_kwargs)

    run.log_metadata(dataset_id, METADATA_FILES.dataset_id, format="json")
    run.log_metadata(dl_cfg, METADATA_FILES.dataloader_config, format="json")

    dl_dict = get_dataloaders(
        dataset_id=dataset_id,
        dataset_registry=dataset_registry,
        dataloader_config=dl_cfg,
        run=run,
    )
    return dataset_id, dl_cfg, dl_dict


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
            print(
                f"Run {self.exp_name}/{run_name} already exists, "
                "but ignore_if_run_exists=True, so ignoring and proceeding."
            )
            return None

        dataset_id = DatasetIdentity(**dataset_kwargs)
        dl_cfg = DataLoaderConfig(**dataloader_kwargs)
        trainer_cfg = TrainerConfig(**trainer_kwargs)

        with self.run_registry.managed_run(
            identity,
            resume=False,
            overwrite=overwrite,
        ) as run:
            trainer = init_trainer(
                run=run,
                dataset_registry=self.dataset_registry,
                dataset_id=dataset_id,
                dl_cfg=dl_cfg,
                trainer_cfg=trainer_cfg,
                gpt_hparams=gpt_hparams,
            )
            self.run_registry.set_device_name(identity, trainer.cfg.device_name)
            trainer.train(max_steps=max_steps)
            return trainer

    def resume(
        self,
        run_name: str,
        *,
        ckpt_filename: str = CKPT_FILES.best_ckpt,
        max_steps: int | None = None,
    ) -> Trainer:
        identity = RunIdentity(self.exp_name, run_name)
        with self.run_registry.managed_run(
            identity,
            resume=True,
            overwrite=False,
        ) as run:
            trainer = trainer_from_checkpoint(
                run=run,
                dataset_registry=self.dataset_registry,
                ckpt_filename=ckpt_filename,
                reset_state=False,
            )
            self.run_registry.set_device_name(identity, trainer.cfg.device_name)
            trainer.train(max_steps=max_steps)
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
    ) -> Trainer:
        identity = RunIdentity(self.exp_name, run_name)
        if ignore_if_run_exists and self.run_registry.run_exists(identity):
            print(
                f"Run {self.exp_name}/{run_name} already exists, "
                "but ignore_if_run_exists=True, so ignoring and proceeding."
            )
            return None
        with self.run_registry.managed_run(
            identity,
            resume=False,
            overwrite=overwrite,
        ) as new_run:
            new_run.log_metadata(
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

            _, _, dl_dict = build_transfer_dataloaders(
                run=new_run,
                dataset_registry=self.dataset_registry,
                dataset_kwargs=dataset_kwargs,
                dataloader_kwargs=dataloader_kwargs,
            )

            with self.run_registry.managed_run(
                RunIdentity(ckpt_exp_name, ckpt_run_name),
                resume=True,
                overwrite=False,
            ) as old_run:
                validate_checkpoint_compatibility(
                    source_run=old_run,
                    target_dataloader_kwargs=dataloader_kwargs,
                    target_dl_info=dl_dict["info"],
                )

                trainer = Trainer.from_checkpoint(
                    old_run,
                    ckpt_name=ckpt_filename,
                    train_dl=dl_dict["train"],
                    eval_dl=dl_dict["eval"],
                    reset_state=True,
                )

            trainer.attach_run(new_run)
            self.run_registry.set_device_name(identity, trainer.cfg.device_name)
            trainer.train(max_steps=max_steps)
            return trainer
        
    def run(self, config: RunConfig) -> Trainer | None:

        if config.method == RunMethod.START:
            return self.start(
                run_name=config.run_name,
                dataset_kwargs=config.dataset_kwargs or {},
                dataloader_kwargs=config.dataloader_kwargs or {},
                gpt_hparams=config.gpt_hparams or {},
                trainer_kwargs=config.trainer_kwargs or {},
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
                dataset_kwargs=config.dataset_kwargs or {},
                dataloader_kwargs=config.dataloader_kwargs or {},
                ckpt_exp_name=config.ckpt_exp_name or "",
                ckpt_run_name=config.ckpt_run_name or "",
                ckpt_filename=config.ckpt_filename,
                max_steps=config.max_steps,
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
    def load_from_module_path(cls, module_path: str) -> "ExperimentConfig":
        import importlib
        import importlib.util
        from pathlib import Path

        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            module_file = Path.cwd() / (module_path.replace(".", "/") + ".py")
            if not module_file.exists():
                raise

            spec = importlib.util.spec_from_file_location(module_path, module_file)
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Could not load config module from path: {module_file}"
                ) from exc

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
