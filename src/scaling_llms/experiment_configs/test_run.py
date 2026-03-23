EXPERIMENT_NAME = "test_run_experiments"

RUNS = [
    {
        "method": "start",
        "run_name": "test_run", # generate it randomly for each unit test
        "dataset_kwargs": {
            "dataset_name": "super_glue",
            "dataset_config": "cb",
            "train_split": "train[:1%]",
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
        "gpt_hparams": {
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 1,
        },
        "trainer_kwargs": {
            "num_steps": 1,
            "lr": 3e-4,
            "accum_steps": 1,
            "lr_schedule": "linear",
            "enable_tb": False,
            "net_log_freq": 1,
            "sys_log_freq": 1,
            "eval_log_freq": 1,
            "ckpt_log_freq": 1,
        },
        "max_steps": 2,
    }
]