from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Any, Iterator

from scaling_llms.constants import (
    PROJECT_NAME,
    PROJECT_DEV_NAME,
    RUN_FILES,
    LOCAL_DATA_DIR,
    LOCAL_DEV_DATA_DIR,
)
from scaling_llms.data import DataConfig, get_dataloaders
from scaling_llms.models import GPTConfig, GPTModel
from scaling_llms.registries import GoogleDriveRunRegistry
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

    def __init__(self, exp_name: str, run_name: str, is_dev: bool = True) -> None:
        self.exp_name = exp_name
        self.run_name = run_name
        self.is_dev = is_dev
        self.project_subdir = PROJECT_DEV_NAME if self.is_dev else PROJECT_NAME
        self.local_data_dir = LOCAL_DEV_DATA_DIR if self.is_dev else LOCAL_DATA_DIR
        self.registry = GoogleDriveRunRegistry(project_subdir=self.project_subdir)

    # --- API ---
    def start(
        self,
        data_kwargs: dict[str, Any],
        gpt_hparams: dict[str, Any],
        trainer_kwargs: dict[str, Any],
        max_steps: int | None = None,
        overwrite: bool = False,
    ) -> Trainer:
        data_cfg = DataConfig(local_data_dir=self.local_data_dir, **data_kwargs)
        trainer_cfg = TrainerConfig(**trainer_kwargs)

        with self._managed_run(
            self.exp_name, self.run_name, resume=False, overwrite=overwrite
        ) as run:
            trainer = self._init_trainer(
                run=run,
                data_cfg=data_cfg,
                trainer_cfg=trainer_cfg,
                gpt_hparams=gpt_hparams,
            )
            trainer.train(max_steps=max_steps)
            return trainer

    def resume(
        self,
        ckpt_filename: str = RUN_FILES.best_ckpt,
        max_steps: int | None = None,
    ) -> Trainer:
        with self._managed_run(
            self.exp_name, self.run_name, resume=True, overwrite=False
        ) as run:
            trainer = self._trainer_from_checkpoint(
                run=run, ckpt_filename=ckpt_filename
            )
            trainer.train(max_steps=max_steps)
            return trainer

    def start_from_checkpoint(
        self,
        data_kwargs: dict[str, Any],
        ckpt_exp_name: str,
        ckpt_run_name: str,
        ckpt_filename: str,
        max_steps: int | None = None,
    ) -> Trainer:
        with self._managed_run(
            self.exp_name, self.run_name, resume=False, overwrite=False
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
            data_cfg = DataConfig(local_data_dir=self.local_data_dir, **data_kwargs)
            new_run.log_metadata(data_cfg, RUN_FILES.data_config, format="json")
            dl_dict = get_dataloaders(
                data_cfg, new_run, project_subdir=self.project_subdir
            )

            # Open old run just long enough to load ckpt into a Trainer object
            with self._managed_run(
                ckpt_exp_name,
                ckpt_run_name,
                resume=True,
                overwrite=False,
            ) as old_run:
                # Validate sequence length matches between old and new runs
                old_data_kwargs = json.loads(
                    old_run.get_metadata_path(RUN_FILES.data_config).read_text()
                )  # Ensure data config is present in old run metadata
                print(old_data_kwargs)
                if old_data_kwargs["seq_len"] != data_kwargs["seq_len"]:
                    self.delete_run(
                        confirm=False
                    )  # Clean up new run since it won't be usable
                    raise ValueError(
                        f"Sequence length mismatch between old run ({old_data_kwargs['seq_len']}) and new run ({data_kwargs['seq_len']}). "
                        "Checkpoint loading may fail."
                    )

                # Validate vocab size matches between old and new runs
                old_model_kwargs = json.loads(
                    old_run.get_metadata_path(RUN_FILES.model_config).read_text()
                )
                if old_model_kwargs["vocab_size"] != dl_dict["info"]["vocab_size"]:
                    self.delete_run(
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
                    reset_state=True,  # Reset state (e.g. step_idx) since this is a new run
                )

            trainer.attach_run(new_run)
            trainer.train(max_steps=max_steps)

            return trainer

    def delete_experiment(self, confirm=True) -> None:
        self.registry.delete_experiment(self.exp_name, confirm=confirm)

    def delete_run(self, confirm=True) -> None:
        self.registry.delete_run(self.exp_name, self.run_name, confirm=confirm)

    # --- Internal methods ---
    @contextmanager
    def _managed_run(
        self,
        exp_name,
        run_name,
        resume: bool,
        overwrite: bool,
    ) -> Iterator[Any]:
        run = self.registry.start_run(
            experiment_name=exp_name,
            run_name=run_name,
            resume=resume,
            overwrite=overwrite,
        )
        try:
            yield run
        finally:
            run.close()

    def _build_model(
        self, data_cfg: DataConfig, vocab_size: int, gpt_hparams: dict[str, Any]
    ) -> GPTModel:
        model_cfg = GPTConfig(
            seq_len=data_cfg.seq_len,
            vocab_size=vocab_size,
            **gpt_hparams,
        )
        return GPTModel(model_cfg)

    def _init_trainer(
        self,
        *,
        run: Any,
        data_cfg: DataConfig,
        trainer_cfg: TrainerConfig,
        gpt_hparams: dict[str, Any],
    ) -> Trainer:
        dl_dict = get_dataloaders(data_cfg, run, project_subdir=self.project_subdir)

        run.log_metadata(data_cfg, RUN_FILES.data_config, format="json")

        model = self._build_model(
            data_cfg=data_cfg,
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
        data_cfg = DataConfig.from_json(run.get_metadata_path(RUN_FILES.data_config))
        dl_dict = get_dataloaders(data_cfg, run, project_subdir=self.project_subdir)
        return Trainer.from_checkpoint(
            run,
            ckpt_name=ckpt_filename,
            train_dl=dl_dict["train"],
            eval_dl=dl_dict["eval"],
        )
