from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Any

from scaling_llms.experiments import NestedExperimentRunner
from scaling_llms.storage.local_disk import make_local_data_registry, make_local_run_registry


_REQUIRED_RUN_KEYS = {
    "run_name",
    "dataset_kwargs",
    "dataloader_kwargs",
    "gpt_hparams",
    "trainer_kwargs",
}


def _validate_config_module(config_module: Any) -> tuple[str, list[dict[str, Any]]]:
    if not hasattr(config_module, "EXPERIMENT_NAME"):
        raise ValueError("Config module must define EXPERIMENT_NAME")
    if not hasattr(config_module, "RUNS"):
        raise ValueError("Config module must define RUNS")

    experiment_name = config_module.EXPERIMENT_NAME
    runs = config_module.RUNS

    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("EXPERIMENT_NAME must be a non-empty string")
    if not isinstance(runs, list) or not runs:
        raise ValueError("RUNS must be a non-empty list")

    normalized_runs: list[dict[str, Any]] = []
    for idx, run_cfg in enumerate(runs):
        if not isinstance(run_cfg, dict):
            raise ValueError(f"RUNS[{idx}] must be a dict")

        missing = _REQUIRED_RUN_KEYS.difference(run_cfg.keys())
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"RUNS[{idx}] missing required keys: {missing_keys}")

        for key in _REQUIRED_RUN_KEYS - {"run_name"}:
            if not isinstance(run_cfg[key], dict):
                raise ValueError(f"RUNS[{idx}]['{key}'] must be a dict")

        if not isinstance(run_cfg["run_name"], str) or not run_cfg["run_name"].strip():
            raise ValueError(f"RUNS[{idx}]['run_name'] must be a non-empty string")

        if "max_steps" in run_cfg and run_cfg["max_steps"] is not None:
            if not isinstance(run_cfg["max_steps"], int) or run_cfg["max_steps"] <= 0:
                raise ValueError(f"RUNS[{idx}]['max_steps'] must be a positive int or None")

        normalized_runs.append(run_cfg)

    return experiment_name, normalized_runs


def _load_config_module(config_module_path: str) -> Any:
    try:
        return importlib.import_module(config_module_path)
    except ModuleNotFoundError as exc:
        module_file = Path.cwd() / (config_module_path.replace(".", "/") + ".py")
        if not module_file.exists():
            raise

        spec = importlib.util.spec_from_file_location(config_module_path, module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load config module from path: {module_file}") from exc

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a group of experiments sequentially")
    parser.add_argument("--config-module", required=True, help="Python module path, e.g. tests.integration.test_config")
    parser.add_argument("--remote-project-root", default="/workspace/remote_registry")
    parser.add_argument("--local-project-root", default="/workspace/local_registry")
    parser.add_argument("--transfer-mode", default="rclone", choices=["shutil", "rclone"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_module = _load_config_module(args.config_module)
    experiment_name, runs = _validate_config_module(config_module)

    remote_project_root = Path(args.remote_project_root)
    local_project_root = Path(args.local_project_root)

    remote_run_registry = make_local_run_registry(project_root=remote_project_root)
    remote_data_registry = make_local_data_registry(project_root=remote_project_root)
    local_run_registry = make_local_run_registry(project_root=local_project_root)

    runner = NestedExperimentRunner(
        exp_name=experiment_name,
        remote_run_registry=remote_run_registry,
        local_run_registry=local_run_registry,
        data_registry=remote_data_registry,
        local_data_dir=local_project_root / "data_registry",
        transfer_mode=args.transfer_mode,
    )

    for run_cfg in runs:
        run_name = run_cfg["run_name"]
        print(f"[group] starting run: {run_name}")
        runner.start(
            run_name=run_name,
            dataset_kwargs=run_cfg["dataset_kwargs"],
            dataloader_kwargs=run_cfg["dataloader_kwargs"],
            gpt_hparams=run_cfg["gpt_hparams"],
            trainer_kwargs=run_cfg["trainer_kwargs"],
            max_steps=run_cfg.get("max_steps"),
        )
        print(f"[group] finished run: {run_name}")


if __name__ == "__main__":
    main()
