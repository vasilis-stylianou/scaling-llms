EXPERIMENT_NAME_PREFIX = "mup_test"

# -------------------------
# DATA CONFIGS
# -------------------------
DATASET_KWARGS = dict(
    dataset_name="Salesforce/wikitext",
    dataset_config="wikitext-103-raw-v1",
    train_split="train",
    eval_split="validation",
    tokenizer_name="gpt2_tiktoken",
    text_field="text"
)

DATALOADER_KWARGS = dict(
    seq_len=1024,
    train_batch_size=64,
    eval_batch_size=None,
    start_sample_idx=0,
    seed=42
)

# -------------------------
# MODEL CONFIGS
# -------------------------
CONSTANT_GPT_HPARAMS = dict(
    # --- Dimensions ---
    # n_embd TO DEFINE IN WIDTH-SWEEP BELOW
    n_layer=6,           # depth fixed throughout
    n_head=4,
    # --- MLP ---
    mlp_type="standard_gelu",
    mlp_bias=False,
    d_ff = None,  # will be set to 4*n_embd in GPTModel if None
    # --- Normalization ---
    norm_type="rmsnorm",
    # --- Positional Encoding ---
    pos_encoding_type="rotary",
    # --- Attention / LM head ---
    attn_bias=False,
    tied_embeddings=False,
    lm_head_bias=False,
   # --- Dropout ---
    attn_pdrop=0.0,
    resid_pdrop=0.0,
)

# -------------------------
# TRAINER CONFIGS
# -------------------------
CONSTANT_TRAINER_ARGS = dict(
    num_steps=200, 
    beta1=0.9,
    beta2=0.95,
    weight_decay=0.0,
    precision="bf16",
    accum_steps=1,
    grad_clip_norm=1.0,
    device="auto",
    use_compile=False,
    local_rank=0,
    seed=42, # same seed for all runs
    iter_mode="infinite",
    lr_schedule=None,
    warmup_steps=0,
    min_lr_ratio=0.0,
    enable_tb=False,
    net_log_freq=20,
    sys_log_freq=-1,
    eval_log_freq=-1,
    ckpt_log_freq=-1,
    keep_last_n=None,
    best_eval_nll_tol=1e-4,
    enable_cuda_timer=False,
)


# -------------------------
# EXPERIMENT GRID VARIABLES
# -------------------------
BASE_WIDTH = 128  # your proxy width
WIDTHS = [BASE_WIDTH, 256, 512, 1024]  # proxy + 3 transfer targets
LOG2LRS = [-12, -10, -9.5, -9, -8.5, -8, -6]