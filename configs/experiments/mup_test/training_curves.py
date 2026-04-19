from itertools import product

from constants import (
    BASE_WIDTH,
    CONSTANT_GPT_HPARAMS,
    CONSTANT_TRAINER_ARGS,
    DATASET_KWARGS, 
    DATALOADER_KWARGS, 
    EXPERIMENT_NAME_PREFIX,
    LOG2LRS,
    WIDTHS,
)

EXPERIMENT_NAME = f"{EXPERIMENT_NAME_PREFIX}_training_curves"

RUNS = []
for use_mup, width, log2lr in product([False, True], WIDTHS, LOG2LRS):
    run_name = "_".join([
        "mup" if use_mup else "sp",
        f"width={width}",
        f"log2lr={log2lr}",
    ])
    gpt_hparams=dict(
        n_embd=width,
        mup_base_width=BASE_WIDTH if use_mup else None,
        **CONSTANT_GPT_HPARAMS,
    )
    trainer_kwargs = dict(**CONSTANT_TRAINER_ARGS, lr=2**log2lr)
    RUNS.append({
        "method": "start",
        "run_name": run_name,
        "overwrite": True,
        "dataset_kwargs": DATASET_KWARGS,
        "dataloader_kwargs": DATALOADER_KWARGS,
        "trainer_kwargs": trainer_kwargs,
        "gpt_hparams": gpt_hparams
    })