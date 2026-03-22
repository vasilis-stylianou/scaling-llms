from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from scaling_llms.constants import CKPT_FILES
from scaling_llms.experiments import ExperimentRunner
from scaling_llms.registries import (
    make_dataset_registry,
    make_run_registry,
)
from scaling_llms.utils.loggers import setup_console_logging


_REQUIRED_TOP_LEVEL_YAML_KEYS = {
    "experiment_config_module",
    "registries",
}

_REQUIRED_EXPERIMENT_CONFIG_KEYS = {
    "run_name",
    "method"
}

_REQUIRED_REGISTRY_KEYS = {
    "artifacts_root",
    "table_name",
}

_VALID_METHODS = {"start", "resume", "start_from_checkpoint"}
_VALID_SYNC_HOOK_TYPES = {None, "rclone"}

# -------------------------------
# INPUT HELPER FUNCTIONS
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

def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    load_dotenv()

    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    if raw is None:
        raise ValueError("Config file is empty")
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a top-level mapping")

    missing = _REQUIRED_TOP_LEVEL_YAML_KEYS.difference(raw.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Missing required config keys: {missing_keys}")

    out = dict(raw)

    experiment_config_module = out["experiment_config_module"]
    if not isinstance(experiment_config_module, str) or not experiment_config_module.strip():
        raise ValueError("experiment_config_module must be a non-empty string")

    registries = out["registries"]
    if not isinstance(registries, dict):
        raise ValueError("registries must be a mapping")
    if "runs" not in registries or "datasets" not in registries:
        raise ValueError("registries must contain both 'runs' and 'datasets'")

    database_url_env_name = registries.get("database_url_env_name")
    if not isinstance(database_url_env_name, str) or not database_url_env_name.strip():
        raise ValueError("registries.database_url_env_name must be a non-empty string")

    database_url_env_name = database_url_env_name.strip()

    out["registries"] = {
        "database_url_env_name": database_url_env_name,
        "runs": _normalize_registry_config(
            name="runs",
            cfg=registries["runs"],
            env_var_name=database_url_env_name,
        ),
        "datasets": _normalize_registry_config(
            name="datasets",
            cfg=registries["datasets"],
            env_var_name=database_url_env_name,
        ),
    }

    return out


def _normalize_registry_config(
    *,
    name: str,
    cfg: Any,
    env_var_name: str,
) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError(f"registries.{name} must be a mapping")

    missing = _REQUIRED_REGISTRY_KEYS.difference(cfg.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"registries.{name} missing required keys: {missing_keys}")

    out = dict(cfg)

    artifacts_root = out["artifacts_root"]
    if not isinstance(artifacts_root, str) or not artifacts_root.strip():
        raise ValueError(f"registries.{name}.artifacts_root must be a non-empty path string")

    table_name = out["table_name"]
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError(f"registries.{name}.table_name must be a non-empty string")

    sync_hooks_type = out.get("sync_hooks_type")
    if sync_hooks_type not in _VALID_SYNC_HOOK_TYPES:
        allowed = ", ".join(sorted(x for x in _VALID_SYNC_HOOK_TYPES if x is not None))
        raise ValueError(
            f"registries.{name}.sync_hooks_type must be one of: None, {allowed}"
        )

    sync_hooks_args = out.get("sync_hooks_args")
    if sync_hooks_type is None:
        if sync_hooks_args is not None and not isinstance(sync_hooks_args, dict):
            raise ValueError(f"registries.{name}.sync_hooks_args must be a dict or null")
    else:
        if sync_hooks_args is None:
            raise ValueError(
                f"registries.{name}.sync_hooks_args must be provided when "
                f"sync_hooks_type={sync_hooks_type!r}"
            )
        if not isinstance(sync_hooks_args, dict):
            raise ValueError(f"registries.{name}.sync_hooks_args must be a dict")

    database_url = out.get("database_url")
    if database_url is None or (isinstance(database_url, str) and not database_url.strip()):
        database_url = os.getenv(env_var_name)
    if database_url is None or not isinstance(database_url, str) or not database_url.strip():
        raise ValueError(
            f"registries.{name}.database_url is missing. Set it in YAML or define "
            f"{env_var_name} in your .env"
        )

    out["artifacts_root"] = str(Path(artifacts_root).expanduser())
    out["table_name"] = table_name.strip()
    out["database_url"] = database_url.strip()

    return out


def _load_experiment_config_module(experiment_config_module_path: str) -> Any:
    try:
        return importlib.import_module(experiment_config_module_path)
    except ModuleNotFoundError as exc:
        module_file = Path.cwd() / (experiment_config_module_path.replace(".", "/") + ".py")
        if not module_file.exists():
            raise

        spec = importlib.util.spec_from_file_location(experiment_config_module_path, module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load config module from path: {module_file}") from exc

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _validate_experiment_config_module(config_module: Any) -> dict[str, Any]:
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

        missing = _REQUIRED_EXPERIMENT_CONFIG_KEYS.difference(run_cfg.keys())
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"RUNS[{idx}] missing required keys: {missing_keys}")

        if not isinstance(run_cfg["run_name"], str) or not run_cfg["run_name"].strip():
            raise ValueError(f"RUNS[{idx}]['run_name'] must be a non-empty string")

        method = run_cfg["method"]
        if method not in _VALID_METHODS:
            allowed = ", ".join(sorted(_VALID_METHODS))
            raise ValueError(f"RUNS[{idx}]['method'] must be one of: {allowed}")

        if method == "start":
            required_start_keys = {
                "dataset_kwargs",
                "dataloader_kwargs",
                "gpt_hparams",
                "trainer_kwargs",
            }
            missing_start = required_start_keys.difference(run_cfg.keys())
            if missing_start:
                missing_keys = ", ".join(sorted(missing_start))
                raise ValueError(
                    f"RUNS[{idx}] with method='start' missing required keys: {missing_keys}"
                )
            for key in required_start_keys:
                if not isinstance(run_cfg[key], dict):
                    raise ValueError(f"RUNS[{idx}]['{key}'] must be a dict")

        elif method == "start_from_checkpoint":
            required_transfer_keys = {
                "dataset_kwargs",
                "dataloader_kwargs",
                "ckpt_exp_name",
                "ckpt_run_name",
                "ckpt_filename",
            }
            missing_transfer = required_transfer_keys.difference(run_cfg.keys())
            if missing_transfer:
                missing_keys = ", ".join(sorted(missing_transfer))
                raise ValueError(
                    f"RUNS[{idx}] with method='start_from_checkpoint' missing required keys: "
                    f"{missing_keys}"
                )
            for key in {"dataset_kwargs", "dataloader_kwargs"}:
                if not isinstance(run_cfg[key], dict):
                    raise ValueError(f"RUNS[{idx}]['{key}'] must be a dict")
            for key in {"ckpt_exp_name", "ckpt_run_name", "ckpt_filename"}:
                if not isinstance(run_cfg[key], str) or not run_cfg[key].strip():
                    raise ValueError(f"RUNS[{idx}]['{key}'] must be a non-empty string")

        elif method == "resume":
            if "ckpt_filename" in run_cfg and (
                not isinstance(run_cfg["ckpt_filename"], str)
                or not run_cfg["ckpt_filename"].strip()
            ):
                raise ValueError(f"RUNS[{idx}]['ckpt_filename'] must be a non-empty string")

        if "max_steps" in run_cfg and run_cfg["max_steps"] is not None:
            if not isinstance(run_cfg["max_steps"], int) or run_cfg["max_steps"] <= 0:
                raise ValueError(f"RUNS[{idx}]['max_steps'] must be a positive int or None")

        normalized_runs.append(run_cfg)

    return {
        "experiment_name": experiment_name.strip(),
        "runs": normalized_runs,
    }


# -------------------------------
# EXPERIMENT RUNNER
# -------------------------------
def run_experiments(exp_cfg, run_registry_cfg, dataset_registry_cfg) -> None:
    setup_console_logging()
    
    experiment_name = exp_cfg["experiment_name"]
    runs = exp_cfg["runs"]

    run_registry = make_run_registry(
        database_url=run_registry_cfg.get("database_url"),
        table_name=run_registry_cfg["table_name"],
        artifacts_root=run_registry_cfg["artifacts_root"],
        sync_hooks_type=run_registry_cfg.get("sync_hooks_type"),
        sync_hooks_args=run_registry_cfg.get("sync_hooks_args"),
    )
    dataset_registry = make_dataset_registry(
        database_url=dataset_registry_cfg.get("database_url"),
        table_name=dataset_registry_cfg["table_name"],
        artifacts_root=dataset_registry_cfg["artifacts_root"],
        sync_hooks_type=dataset_registry_cfg.get("sync_hooks_type"),
        sync_hooks_args=dataset_registry_cfg.get("sync_hooks_args"),
    )
    exp_runner = ExperimentRunner(
        exp_name=experiment_name,
        run_registry=run_registry,
        dataset_registry=dataset_registry,
    )

    for run_cfg in runs:
        run_name = run_cfg["run_name"]
        method = run_cfg.get("method", "start")
        print(f"[run] starting: {experiment_name}/{run_name}")

        if method == "start":
            _ = exp_runner.start(
                run_name=run_name,
                dataset_kwargs=run_cfg["dataset_kwargs"],
                dataloader_kwargs=run_cfg["dataloader_kwargs"],
                gpt_hparams=run_cfg["gpt_hparams"],
                trainer_kwargs=run_cfg["trainer_kwargs"],
                max_steps=run_cfg.get("max_steps"),
                overwrite=run_cfg.get("overwrite", False),
            )
        elif method == "resume":
            _ = exp_runner.resume(
                run_name=run_name,
                ckpt_filename=run_cfg.get("ckpt_filename", CKPT_FILES.best_ckpt),
                max_steps=run_cfg.get("max_steps"),
            )
        elif method == "start_from_checkpoint":
            _ = exp_runner.start_from_checkpoint(
                run_name=run_name,
                dataset_kwargs=run_cfg["dataset_kwargs"],
                dataloader_kwargs=run_cfg["dataloader_kwargs"],
                ckpt_exp_name=run_cfg["ckpt_exp_name"],
                ckpt_run_name=run_cfg["ckpt_run_name"],
                ckpt_filename=run_cfg["ckpt_filename"],
                max_steps=run_cfg.get("max_steps"),
            )

        print(f"[run] finished and synced: {experiment_name}/{run_name}")


# -------------------------------
# ENTRY POINT
# -------------------------------
def main():
    # Load config from YAML and validate
    args = _parse_args()
    cfg_path = Path(args.config_path).expanduser().resolve()
    cfg = _load_config(cfg_path)

    # Load and validate the experiment config module
    config_module = _load_experiment_config_module(cfg["experiment_config_module"])
    exp_cfg = _validate_experiment_config_module(config_module)
    
    # Run the experiments
    run_experiments(
        exp_cfg=exp_cfg,
        run_registry_cfg=cfg["registries"]["runs"],
        dataset_registry_cfg=cfg["registries"]["datasets"],
    )


if __name__ == "__main__":
    main()