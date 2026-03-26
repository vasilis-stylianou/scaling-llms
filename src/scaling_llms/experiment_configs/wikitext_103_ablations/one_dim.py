# -------------------------
# DATA CONFIGS
# -------------------------
DATASET_KWARGS = dict(
    dataset_name="Salesforce/wikitext",
    dataset_config="wikitext-103-raw-v1",
    train_split="train",
    eval_split="validation",
    tokenizer_name="gpt2_tiktoken",
    text_field = "text"
)

DATALOADER_KWARGS = dict(
    seq_len=1024,
    train_batch_size=16,
    eval_batch_size=16,
    start_sample_idx=0,
    seed=1234
)


# -------------------------
# TRAINER CONFIGS
# -------------------------
CONSTANT_OPTIMIZATION_KWARGS = {
    "lr": 3e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "precision": "bf16",
    "device": "cuda",
}


DEBUG_TRAINER_KWARGS = dict(
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


ONE_DIM_SCREENING_TRAINER_KWARGS = dict(
    num_steps=200,
    weight_decay=0.1,
    accum_steps=16,
    grad_clip_norm=1.0,
    lr_schedule="cosine",
    warmup_steps=10,
    min_lr_ratio=0.1,
    enable_tb=True,
    eval_log_freq=10,
    net_log_freq=10,
    sys_log_freq=20,
    ckpt_log_freq=40,
    enable_cuda_timer=True,
    **CONSTANT_OPTIMIZATION_KWARGS

)


MULTI_DIM_SCREENING_TRAINER_KWARGS = dict(
    num_steps=455,
    weight_decay=0.1,
    accum_steps=16,
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
CONSTANT_GPT_HPARAMS = dict(
    n_embd=768,
    n_layer=12,
    n_head=12,
)

# 1D Screening 
BASELINE_HPARAMS = dict(
    mlp_type="standard_gelu",
    pos_encoding_type="absolute",
    norm_type="layernorm",
    **CONSTANT_GPT_HPARAMS
)

RMS_NORM_HPARAMS = dict(
    mlp_type="standard_gelu",
    pos_encoding_type="absolute",
    norm_type="rmsnorm",
    **CONSTANT_GPT_HPARAMS
)

ROTARY_HPARAMS = dict(
    mlp_type="standard_gelu",
    pos_encoding_type="rotary",
    norm_type="layernorm",
    **CONSTANT_GPT_HPARAMS
)

SWIGLU_HPARAMS = dict(
    mlp_type="swiglu",
    pos_encoding_type="absolute",
    norm_type="layernorm",
    **CONSTANT_GPT_HPARAMS
)

# Multi-Dimensional Screening
# TODO
