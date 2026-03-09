from __future__ import annotations

import json
from typing import Any

from scaling_llms.constants import (
    CKPT_FILES,
    LOCAL_DATA_DIR,
    LOCAL_DEV_DATA_DIR,
    METADATA_FILES,
    PROJECT_NAME,
    PROJECT_DEV_NAME,
)
from scaling_llms.data import DataLoaderConfig, get_dataloaders
from scaling_llms.models import GPTConfig, GPTModel
from scaling_llms.registries.datasets.identity import DatasetIdentity
from scaling_llms.registries.runs.identity import RunIdentity
from scaling_llms.storage.google_drive import (
    make_gdrive_data_registry,
    make_gdrive_run_registry,
)
from scaling_llms.trainer import Trainer, TrainerConfig


class ExperimentRunner:
    """
    High-level interface for managing training experiments, runs, and checkpoints.

    Main Methods:
    - start: Start a new training run with specified configs.
    - resume: Resume training from the latest checkpoint.
    - start_from_checkpoint: Start a new run initialized from an existing checkpoint.
    - delete_experiment: Delete an entire experiment and all its runs.
    - delete_run: Delete a specific run within an experiment.

    """

    def __init__(self, exp_name: str, is_dev: bool = True) -> None:
        self.exp_name = exp_name
        self.is_dev = is_dev
        self.project_subdir = PROJECT_DEV_NAME if self.is_dev else PROJECT_NAME
        self.local_data_dir = LOCAL_DEV_DATA_DIR if self.is_dev else LOCAL_DATA_DIR
        self.run_registry = make_gdrive_run_registry(project_subdir=self.project_subdir)
        self.data_registry = make_gdrive_data_registry(project_subdir=self.project_subdir)

    # --- API ---
    def start(
        self,
        run_name: str,
        dataset_kwargs: dict[str, Any],
        dataloader_kwargs: dict[str, Any],
        gpt_hparams: dict[str, Any],
        trainer_kwargs: dict[str, Any],
        max_steps: int | None = None,
        overwrite: bool = False,
        ignore_if_run_exists: bool = False,
    ) -> Trainer:
        if ignore_if_run_exists and self.run_registry.run_exists(RunIdentity(self.exp_name, run_name)):
            print(f"Run {self.exp_name}/{run_name} already exists, but ignore_if_run_exists=True, so ignoring and proceeding.")
            return None  

        dataset_id = DatasetIdentity(**dataset_kwargs)
        dl_cfg = DataLoaderConfig(**dataloader_kwargs)
        trainer_cfg = TrainerConfig(**trainer_kwargs)

        with self.run_registry.managed_run(
            RunIdentity(self.exp_name, run_name),
            resume=False, 
            overwrite=overwrite, 
        ) as run:
            trainer = self._init_trainer(
                run=run,
                dataset_id=dataset_id,
                dl_cfg=dl_cfg,
                trainer_cfg=trainer_cfg,
                gpt_hparams=gpt_hparams,
            )
            trainer.train(max_steps=max_steps)
            return trainer

    def resume(
        self,
        run_name: str,
        ckpt_filename: str = CKPT_FILES.best_ckpt,
        max_steps: int | None = None,
    ) -> Trainer:
        with self.run_registry.managed_run(
            RunIdentity(self.exp_name, run_name), resume=True, overwrite=False
        ) as run:
            trainer = self._trainer_from_checkpoint(
                run=run, ckpt_filename=ckpt_filename
            )
            trainer.train(max_steps=max_steps)
            return trainer

    def start_from_checkpoint(
        self,
        run_name: str,
        dataset_kwargs: dict[str, Any],
        dataloader_kwargs: dict[str, Any],
        ckpt_exp_name: str,
        ckpt_run_name: str,
        ckpt_filename: str,
        max_steps: int | None = None,
    ) -> Trainer:
        with self.run_registry.managed_run(
            RunIdentity(self.exp_name, run_name), resume=False, overwrite=False
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
            dataset_id = DatasetIdentity(**dataset_kwargs)
            dl_cfg = DataLoaderConfig(**dataloader_kwargs)

            new_run.log_metadata(dataset_id, METADATA_FILES.dataset_id, format="json")
            new_run.log_metadata(dl_cfg, METADATA_FILES.dataloader_config, format="json")
            dl_dict = get_dataloaders(
                dataset_id=dataset_id,
                data_registry=self.data_registry,
                dataloader_config=dl_cfg,
                local_data_dir=self.local_data_dir,
                run=new_run,
            )

            # Open old run just long enough to load ckpt into a Trainer object
            with self.run_registry.managed_run(
                RunIdentity(ckpt_exp_name, ckpt_run_name),
                resume=True,
                overwrite=False,
            ) as old_run:
                # Validate sequence length matches between old and new runs
                old_dl_kwargs = json.loads(
                    old_run.artifacts.metadata_path(METADATA_FILES.dataloader_config).read_text()
                )  
                
                if old_dl_kwargs["seq_len"] != dataloader_kwargs["seq_len"]:
                    self.delete_run(
                        run_name=run_name,
                        confirm=False
                    )  # Clean up new run since it won't be usable
                    raise ValueError(
                        f"Sequence length mismatch between old run ({old_dl_kwargs['seq_len']}) and new run ({dataloader_kwargs['seq_len']}). "
                        "Checkpoint loading may fail."
                    )

                # Validate vocab size matches between old and new runs
                old_model_kwargs = json.loads(
                    old_run.artifacts.metadata_path(METADATA_FILES.model_config).read_text()
                )
                if old_model_kwargs["vocab_size"] != dl_dict["info"]["vocab_size"]:
                    self.delete_run(
                        run_name=run_name,
                        confirm=False
                    )  # Clean up new run since it won't be usable
                    raise ValueError(
                        f"Vocab size mismatch between old run ({old_model_kwargs['vocab_size']}) and new run ({dl_dict['info']['vocab_size']}). "
                        "Checkpoint loading may fail."
                    )

                # Load trainer from checkpoint
                trainer = Trainer.from_checkpoint(
                    old_run,
                    ckpt_name=ckpt_filename,
                    train_dl=dl_dict["train"],
                    eval_dl=dl_dict["eval"],
                    reset_state=True,  # Reset state (e.g. optimizer, lr scheduler,step_idx)
                )

            trainer.attach_run(new_run)
            trainer.train(max_steps=max_steps)

            return trainer

    def delete_experiment(self, confirm=True) -> None:
        self.run_registry.delete_experiment(self.exp_name, confirm=confirm)

    def delete_run(self, run_name: str, confirm=True) -> None:
        self.run_registry.delete_run(RunIdentity(self.exp_name, run_name), confirm=confirm)

    # --- Internal methods ---
    def _build_model(
        self, seq_len: int, vocab_size: int, gpt_hparams: dict[str, Any]
    ) -> GPTModel:
        model_cfg = GPTConfig(
            seq_len=seq_len,
            vocab_size=vocab_size,
            **gpt_hparams,
        )
        return GPTModel(model_cfg)

    def _init_trainer(
        self,
        *,
        run: Any,
        dataset_id: DatasetIdentity,
        dl_cfg: DataLoaderConfig,
        trainer_cfg: TrainerConfig,
        gpt_hparams: dict[str, Any],
    ) -> Trainer:
        dl_dict = get_dataloaders(
            dataset_id=dataset_id,
            data_registry=self.data_registry,
            dataloader_config=dl_cfg,
            local_data_dir=self.local_data_dir,
            run=run,
        )
        run.log_metadata(dataset_id, METADATA_FILES.dataset_id, format="json")
        run.log_metadata(dl_cfg, METADATA_FILES.dataloader_config, format="json")

        model = self._build_model(
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

    def _trainer_from_checkpoint(self, run: Any, ckpt_filename: str) -> Trainer:
        # Load trainer state from checkpoint
        trainer = Trainer.from_checkpoint(
            run,
            ckpt_name=ckpt_filename,
            reset_state=False,  # Load full trainer state by default when resuming
        )

        # Reload dataset identity and dataloader config from run metadata,
        # offsetting start_sample_idx by the trainer's current step for correct resumption.
        dataset_id = DatasetIdentity.from_json(
            path=run.artifacts.metadata_path(METADATA_FILES.dataset_id)
        )
        dl_cfg = DataLoaderConfig.from_json(
            path=run.artifacts.metadata_path(METADATA_FILES.dataloader_config),
            overwrite_data={"start_sample_idx": trainer.step_idx},
        )

        dl_dict = get_dataloaders(
            dataset_id=dataset_id,
            data_registry=self.data_registry,
            dataloader_config=dl_cfg,
            local_data_dir=self.local_data_dir,
            run=run,
        )

        # Attach dataloaders to trainer
        trainer.attach_dataloaders(dl_dict["train"], dl_dict["eval"])

        return trainer