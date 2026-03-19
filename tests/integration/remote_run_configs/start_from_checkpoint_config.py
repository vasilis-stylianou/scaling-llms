EXPERIMENT_NAME = "it_remote_runner_dev"

RUNS = [
    {
        "run_name": "run_transfer",
        "method": "start_from_checkpoint",
        "dataset_kwargs": {
            "dataset_name": "super_glue",
            "dataset_config": "cb",
            "train_split": "train[1%:2%]",
            "eval_split": "test[:1%]",
            "tokenizer_name": "gpt2_tiktoken",
            "text_field": "premise",
        },
        "dataloader_kwargs": {
            "seq_len": 16,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "start_sample_idx": 0,
            "seed": 1,
        },
        "ckpt_exp_name": "it_remote_runner_dev",
        "ckpt_run_name": "run_start",
        "ckpt_filename": "best.pt",
        "max_steps": 1,
    }
]
