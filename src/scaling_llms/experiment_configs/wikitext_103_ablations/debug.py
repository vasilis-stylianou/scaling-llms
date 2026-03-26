from scaling_llms.experiment_configs.wikitext_103_ablation.constants import (
    CONSTANT_GPT_HPARAMS,
    CONSTANT_OPTIMIZATION_KWARGS,
    DATASET_KWARGS,
    DATALOADER_KWARGS,
    EXPERIMENT_NAME_PREFIX
)

# -------------------------
# TRAINER CONFIGS
# -------------------------
_DEBUG_TRAINER_KWARGS = dict(
    num_steps=50,
    weight_decay=0.0,
    accum_steps=1,
    grad_clip_norm=None,
    iter_mode="single-batch",
    lr_schedule="none",
    warmup_steps=0,
    min_lr_ratio=0.0,
    enable_tb=False,
    eval_log_freq=5,
    net_log_freq=5,
    sys_log_freq=10,
    ckpt_log_freq=0,
    enable_cuda_timer=False,
    **CONSTANT_OPTIMIZATION_KWARGS
)

# -------------------------
# MODEL CONFIGS
# -------------------------
_BASELINE_HPARAMS = dict(
    mlp_type="standard_gelu",
    pos_encoding_type="absolute",
    norm_type="layernorm",
    **CONSTANT_GPT_HPARAMS
)

_RMS_NORM_HPARAMS = dict(
    mlp_type="standard_gelu",
    pos_encoding_type="absolute",
    norm_type="rmsnorm",
    **CONSTANT_GPT_HPARAMS
)

_ROTARY_HPARAMS = dict(
    mlp_type="standard_gelu",
    pos_encoding_type="rotary",
    norm_type="layernorm",
    **CONSTANT_GPT_HPARAMS
)

_SWIGLU_HPARAMS = dict(
    mlp_type="swiglu",
    pos_encoding_type="absolute",
    norm_type="layernorm",
    **CONSTANT_GPT_HPARAMS
)

_DEBUG_RUNS = [
    ("baseline", _BASELINE_HPARAMS),
    ("rms_norm", _RMS_NORM_HPARAMS),
    ("rotary", _ROTARY_HPARAMS),
    ("swiglu", _SWIGLU_HPARAMS)
]


EXPERIMENT_NAME = f"{EXPERIMENT_NAME_PREFIX}_debug"
RUNS = [
    { 
        "method": "start",
        "run_name": run_name,
        "overwrite": True,
        "dataset_kwargs": DATASET_KWARGS,
        "dataloader_kwargs": DATALOADER_KWARGS,
        "trainer_kwargs": _DEBUG_TRAINER_KWARGS,
        "gpt_hparams": hparams
    }
    for run_name, hparams in _DEBUG_RUNS
]