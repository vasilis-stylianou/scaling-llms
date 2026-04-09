from scaling_llms.experiment_configs.wikitext_103_ablations.constants import (
    CONSTANT_GPT_HPARAMS,
    CONSTANT_OPTIMIZATION_KWARGS,
    DATASET_KWARGS,
    DATALOADER_KWARGS,
    EXPERIMENT_NAME_PREFIX
)

# -------------------------
# TRAINER CONFIGS
# -------------------------
_2D_SCREENING_TRAINER_KWARGS = dict(
    num_steps=455,
    weight_decay=0.1,
    accum_steps=8,
    grad_clip_norm=1.0,
    iter_mode="infinite",
    lr_schedule="cosine",
    warmup_steps=20,
    min_lr_ratio=0.1,
    enable_tb=True,
    eval_log_freq=20,
    net_log_freq=20,
    sys_log_freq=40,
    ckpt_log_freq=40,
    enable_cuda_timer=True,
    **CONSTANT_OPTIMIZATION_KWARGS
)

# -------------------------
# MODEL CONFIGS
# -------------------------
_ROTARY_RMS_GELU_HPARAMS = dict(
    mlp_type="standard_gelu",
    pos_encoding_type="rotary",
    norm_type="rmsnorm",
    **CONSTANT_GPT_HPARAMS
)

_ROTARY_RMS_SWIGLU_HPARAMS = dict(
    mlp_type="swiglu",
    pos_encoding_type="rotary",
    norm_type="rmsnorm",
    **CONSTANT_GPT_HPARAMS
)

_ROTARY_LN_SWIGLU_HPARAMS = dict(
    mlp_type="swiglu",
    pos_encoding_type="rotary",
    norm_type="layernorm",
    **CONSTANT_GPT_HPARAMS
)

_2D_RUNS = [
    ("rotary_rms_gelu", _ROTARY_RMS_GELU_HPARAMS),
    ("rotary_rms_swiglu", _ROTARY_RMS_SWIGLU_HPARAMS),
    ("rotary_ln_swiglu", _ROTARY_LN_SWIGLU_HPARAMS),
]


EXPERIMENT_NAME = f"{EXPERIMENT_NAME_PREFIX}_2d_screening"
RUNS = [
    { 
        "method": "start",
        "run_name": run_name,
        "overwrite": True,
        "dataset_kwargs": DATASET_KWARGS,
        "dataloader_kwargs": DATALOADER_KWARGS,
        "trainer_kwargs": _2D_SCREENING_TRAINER_KWARGS,
        "gpt_hparams": hparams
    }
    for run_name, hparams in _2D_RUNS
]