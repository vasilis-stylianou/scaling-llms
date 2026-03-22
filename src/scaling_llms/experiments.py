# experiments.py

from __future__ import annotations

import json
import pandas as pd
from typing import Any

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


# =============================================================================
# Single-registry runner
# =============================================================================
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
    ) -> Trainer:
        identity = RunIdentity(self.exp_name, run_name)
        with self.run_registry.managed_run(
            identity,
            resume=False,
            overwrite=False,
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
        

# -------------------------------------------------------------------------------------
# TODO: EXPERIMENT METHODS WILL BE MIGRATED TO experiments.py
# -------------------------------------------------------------------------------------

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
