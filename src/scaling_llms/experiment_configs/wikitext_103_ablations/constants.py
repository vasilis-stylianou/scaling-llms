
EXPERIMENT_NAME_PREFIX = "wikitest_103_ablations"

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


# -------------------------
# MODEL CONFIGS
# -------------------------
CONSTANT_GPT_HPARAMS = dict(
    n_embd=768,
    n_layer=12,
    n_head=12,
)