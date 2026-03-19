EXPERIMENT_NAME = "it_remote_runner_dev"

RUNS = [
    {
        "run_name": "run_start",
        "method": "resume",
        "ckpt_filename": "best.pt",
        "max_steps": 2,
    }
]
