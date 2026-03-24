from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from scaling_llms.experiments import ExperimentConfig, ExperimentRunner
from scaling_llms.registries import (
    make_dataset_registry,
    make_run_registry,
)
from scaling_llms.registries.datasets.registry import MakeDatasetRegistryConfig
from scaling_llms.registries.runs.registry import MakeRunRegistryConfig
from scaling_llms.utils.loggers import setup_console_logging


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
        nargs="?",
        default="configs/experiments/remote_smoke.yaml",
        help="Path to experiment runtime YAML config",
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
    if not isinstance(experiment_config_module, str) or not experiment_config_module.strip():
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
    )


# -------------------------------
# EXPERIMENT RUNNER
# -------------------------------
def run_experiments(
    exp_cfg: ExperimentConfig,
    run_registry_cfg: MakeRunRegistryConfig,
    dataset_registry_cfg: MakeDatasetRegistryConfig,
) -> None:
    setup_console_logging()

    run_registry = make_run_registry(run_registry_cfg)
    dataset_registry = make_dataset_registry(dataset_registry_cfg)
    exp_runner = ExperimentRunner(
        exp_name=exp_cfg.experiment_name,
        run_registry=run_registry,
        dataset_registry=dataset_registry,
    )

    for run_cfg in exp_cfg.runs:
        print(f"[run] starting: {exp_cfg.experiment_name}/{run_cfg.run_name}")
        _ = exp_runner.run(config=run_cfg)
        print(f"[run] finished and synced: {exp_cfg.experiment_name}/{run_cfg.run_name}")
        

# -------------------------------
# ENTRY POINT
# -------------------------------
def main() -> None:
    # Load config from YAML and validate
    args = _parse_args()
    cfg_path = Path(args.config_path).expanduser().resolve()
    cfg = _load_config(cfg_path)

    # Run the experiments
    run_experiments(
        exp_cfg=cfg.experiment,
        run_registry_cfg=cfg.run_registry,
        dataset_registry_cfg=cfg.dataset_registry,
    )


if __name__ == "__main__":
    main()