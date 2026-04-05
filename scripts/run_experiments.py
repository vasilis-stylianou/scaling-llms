from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import shutil

import yaml
from dotenv import load_dotenv

from scaling_llms.distributed import ddp_cleanup, ddp_setup, is_main_process
from scaling_llms.experiments import ExperimentConfig, ExperimentRunner
from scaling_llms.registries import (
    make_dataset_registry,
    make_run_registry,
    MakeDatasetRegistryConfig,
    MakeRunRegistryConfig,
)
from scaling_llms.utils.loggers import setup_console_logging, BaseLogger


_REQUIRED_TOP_LEVEL_YAML_KEYS = {
    "experiment_config_module",
    "registries",
}


# -------------------------------
# ARGUMENT PARSING
# -------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiments via ExperimentRunner",
    )
    parser.add_argument(
        "config_path",
        help="Path to experiment runtime YAML config",
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        default="nccl", 
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend to use (default: nccl)"
    )
    return parser.parse_args()


# -------------------------------
# CONFIG LOADING AND VALIDATION
# -------------------------------
@dataclass(slots=True)
class LoadedConfig:
    experiment: ExperimentConfig
    run_registry: MakeRunRegistryConfig
    dataset_registry: MakeDatasetRegistryConfig
    cleanup_after_sync: bool = False


def _load_config(config_path: Path) -> LoadedConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    load_dotenv()

    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)

    if data is None:
        raise ValueError("Config file is empty")
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping")

    missing = _REQUIRED_TOP_LEVEL_YAML_KEYS.difference(data.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Missing required config keys: {missing_keys}")

    experiment_config_module = data["experiment_config_module"]
    if (
        not isinstance(experiment_config_module, str)
        or not experiment_config_module.strip()
    ):
        raise ValueError("experiment_config_module must be a non-empty string")

    registries = data["registries"]
    if not isinstance(registries, dict):
        raise ValueError("registries must be a mapping")
    if "runs" not in registries or "datasets" not in registries:
        raise ValueError("registries must contain both 'runs' and 'datasets'")

    database_url_env_name = registries.get("database_url_env_name")
    if not isinstance(database_url_env_name, str) or not database_url_env_name.strip():
        raise ValueError("registries.database_url_env_name must be a non-empty string")

    database_url_env_name = database_url_env_name.strip()

    cleanup_after_sync = data.get("cleanup_after_sync", False)
    if not isinstance(cleanup_after_sync, bool):
        raise ValueError("cleanup_after_sync must be a boolean")

    return LoadedConfig(
        experiment=ExperimentConfig.load_from_module_path(
            experiment_config_module.strip()
        ),
        run_registry=MakeRunRegistryConfig.from_raw(
            name="runs",
            data=registries["runs"],
            env_var_name=database_url_env_name,
        ),
        dataset_registry=MakeDatasetRegistryConfig.from_raw(
            name="datasets",
            data=registries["datasets"],
            env_var_name=database_url_env_name,
        ),
        cleanup_after_sync=cleanup_after_sync,
    )


# -------------------------------
# EXPERIMENT RUNNER
# -------------------------------
def run_experiments(
    exp_cfg: ExperimentConfig,
    run_registry_cfg: MakeRunRegistryConfig,
    dataset_registry_cfg: MakeDatasetRegistryConfig,
    cleanup_after_sync: bool = False,
) -> None:
    setup_console_logging()
    
    _is_main = is_main_process()
    if _is_main:
        logger = BaseLogger("run_experiments")

    run_registry = make_run_registry(run_registry_cfg)
    dataset_registry = make_dataset_registry(dataset_registry_cfg)
    exp_runner = ExperimentRunner(
        exp_name=exp_cfg.experiment_name,
        run_registry=run_registry,
        dataset_registry=dataset_registry,
    )

    for run_cfg in exp_cfg.runs:
        if _is_main:
            logger.info(f"[run] starting: {exp_cfg.experiment_name}/{run_cfg.run_name}")

        trainer = exp_runner.run(config=run_cfg)

        if _is_main:
            logger.info(
                f"[run] finished and synced: {exp_cfg.experiment_name}/{run_cfg.run_name}"
            )

            if (
                cleanup_after_sync 
                and (trainer is not None)
                and (trainer.run is not None) 
            ):
                artifacts_root = trainer.run.artifacts_dir.root
                logger.info(f"[cleanup] deleting local artifacts: {artifacts_root}")
                shutil.rmtree(artifacts_root, ignore_errors=True)


# -------------------------------
# ENTRY POINT
# -------------------------------
def main() -> None:
    args = _parse_args()

    # Initialise DDP process group when launched via torchrun (no-op otherwise)
    ddp_setup(backend=args.backend)

    try:
        # Load config from YAML and validate
        cfg_path = Path(args.config_path).expanduser().resolve()
        cfg = _load_config(cfg_path)

        # Run the experiments
        run_experiments(
            exp_cfg=cfg.experiment,
            run_registry_cfg=cfg.run_registry,
            dataset_registry_cfg=cfg.dataset_registry,
            cleanup_after_sync=cfg.cleanup_after_sync,
        )
    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
