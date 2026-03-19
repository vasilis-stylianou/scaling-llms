from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any

import yaml

from scaling_llms.constants import CKPT_FILES
from scaling_llms.experiments import ExperimentRunner
from scaling_llms.registries.datasets.registry import DataRegistry
from scaling_llms.registries.runs.identity import RunIdentity
from scaling_llms.registries.runs.registry import RunRegistry
from scaling_llms.storage.base import RegistryStorage
from scaling_llms.storage.local_disk import LocalDiskConfigs, setup_local_storage
from scaling_llms.storage.rclone import (
    RCloneDiskConfigs,
    make_remote_storage,
    sync_local_to_remote,
    sync_remote_to_local,
)
from scaling_llms.utils.loggers import setup_console_logging


_REQUIRED_YAML_KEYS = {
    "experiment_config_module",
    "local_project_root",
    "remote_project_subdir",
}

_REQUIRED_EXPERIMENT_CONFIG_KEYS = {
    "run_name",
}

_VALID_METHODS = {"start", "resume", "start_from_checkpoint"}

# -------------------------------
# REMOTE RUN CONFIGS HELPERS
# -------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run remote-backed experiments locally via ExperimentRunner",
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="remote_run_config.yaml",
        help="Path to remote run YAML config (default: remote_run_config.yaml)",
    )
    return parser.parse_args()


def _load_run_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a top-level mapping")

    missing = _REQUIRED_YAML_KEYS.difference(raw.keys())
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Missing required config keys: {missing_keys}")

    out = dict(raw)
    out.setdefault("remote_name", "gdrive")
    out.setdefault("rclone_executable", "rclone")
    out.setdefault("rclone_extra_args", [])
    out.setdefault("overwrite_runs", False)

    if not isinstance(out["rclone_extra_args"], list):
        raise ValueError("rclone_extra_args must be a list of strings")

    return out


# -------------------------------
# EXPERIMENT CONFIGS HELPERS
# -------------------------------
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


def _validate_experiment_config_module(config_module: Any) -> tuple[str, list[dict[str, Any]]]:
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

        method = run_cfg.get("method", "start")
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

    return experiment_name, normalized_runs


# -------------------------------
# REGISTRY HELPERS
# -------------------------------
def _bootstrap_local_run_registry(
    *,
    local_storage: RegistryStorage,
    remote_storage: RegistryStorage,
    rclone_executable: str,
    rclone_extra_args: list[str] | None,
) -> RunRegistry:
    local_artifacts_root = local_storage.runs_artifacts_root
    if not local_artifacts_root.exists():
        local_artifacts_root.mkdir(parents=True, exist_ok=True)
    else:
        for child in local_artifacts_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)

    sync_remote_to_local(
        remote_path=str(remote_storage.runs_db_path),
        local_path=local_storage.run_registry_root,
        mode="copy",
        rclone_executable=rclone_executable,
        extra_args=rclone_extra_args,
    )

    return RunRegistry.from_storage(local_storage)


def _bootstrap_local_data_registry_db(
    *,
    local_storage: RegistryStorage,
    remote_storage: RegistryStorage,
    rclone_executable: str,
    rclone_extra_args: list[str] | None,
) -> DataRegistry:
    local_storage.datasets_artifacts_root.mkdir(parents=True, exist_ok=True)

    sync_remote_to_local(
        remote_path=str(remote_storage.datasets_db_path),
        local_path=local_storage.data_registry_root,
        mode="copy",
        rclone_executable=rclone_executable,
        extra_args=rclone_extra_args,
    )

    local_data_registry = DataRegistry.from_storage(local_storage)
    local_data_registry.storage = remote_storage
    return local_data_registry


def _sync_remote_run_artifacts_to_local(
    *,
    identity: RunIdentity,
    local_run_registry: RunRegistry,
    remote_storage: RegistryStorage,
    rclone_executable: str,
    rclone_extra_args: list[str] | None,
) -> None:
    local_run_dir = local_run_registry.get_run_dir(identity)
    relative_run_path = local_run_dir.relative_to(local_run_registry.artifacts_root)
    remote_run_dir = Path(remote_storage.runs_artifacts_root) / relative_run_path

    sync_remote_to_local(
        remote_path=str(remote_run_dir),
        local_path=local_run_dir,
        mode="copy",
        rclone_executable=rclone_executable,
        extra_args=rclone_extra_args,
    )


def main() -> None:
    setup_console_logging()

    # Load and validate the remote run config YAML
    args = _parse_args()
    cfg_path = Path(args.config_path).expanduser().resolve()
    cfg = _load_run_config(cfg_path)

    # Load and validate the experiment config module
    config_module = _load_experiment_config_module(cfg["experiment_config_module"])
    experiment_name, runs = _validate_experiment_config_module(config_module)

    # Setup local and remote storage
    local_project_root = Path(cfg["local_project_root"]).expanduser().resolve()
    local_project_root.mkdir(parents=True, exist_ok=True)
    local_storage = setup_local_storage(LocalDiskConfigs(project_root=local_project_root))

    remote_configs = RCloneDiskConfigs(
        remote_name=cfg["remote_name"],
        remote_project_subdir=cfg["remote_project_subdir"],
    )
    remote_storage = make_remote_storage(remote_configs)

    rclone_extra_args = cfg["rclone_extra_args"] if cfg["rclone_extra_args"] else None

    # Create local registries and sync remote DBs for run/dataset lookup
    local_run_registry = _bootstrap_local_run_registry(
        local_storage=local_storage,
        remote_storage=remote_storage,
        rclone_executable=cfg["rclone_executable"],
        rclone_extra_args=rclone_extra_args,
    )
    local_data_registry = _bootstrap_local_data_registry_db(
        local_storage=local_storage,
        remote_storage=remote_storage,
        rclone_executable=cfg["rclone_executable"],
        rclone_extra_args=rclone_extra_args,
    )

    print(f"[bootstrap] local run DB: {local_run_registry.db_path}")
    print(f"[bootstrap] local run artifacts dir (must start empty): {local_run_registry.artifacts_root}")
    print(f"[bootstrap] local data DB: {local_data_registry.db_path}")

    exp_runner = ExperimentRunner(
        exp_name=experiment_name,
        run_registry=local_run_registry,
        data_registry=local_data_registry,
    )

    # Run each experiment sequentially, syncing the run artifacts to remote after each run completes
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
                overwrite=cfg["overwrite_runs"],
            )
        elif method == "resume":
            _sync_remote_run_artifacts_to_local(
                identity=RunIdentity(experiment_name, run_name),
                local_run_registry=local_run_registry,
                remote_storage=remote_storage,
                rclone_executable=cfg["rclone_executable"],
                rclone_extra_args=rclone_extra_args,
            )
            _ = exp_runner.resume(
                run_name=run_name,
                ckpt_filename=run_cfg.get("ckpt_filename", CKPT_FILES.best_ckpt),
                max_steps=run_cfg.get("max_steps"),
            )
        elif method == "start_from_checkpoint":
            _sync_remote_run_artifacts_to_local(
                identity=RunIdentity(run_cfg["ckpt_exp_name"], run_cfg["ckpt_run_name"]),
                local_run_registry=local_run_registry,
                remote_storage=remote_storage,
                rclone_executable=cfg["rclone_executable"],
                rclone_extra_args=rclone_extra_args,
            )
            _ = exp_runner.start_from_checkpoint(
                run_name=run_name,
                dataset_kwargs=run_cfg["dataset_kwargs"],
                dataloader_kwargs=run_cfg["dataloader_kwargs"],
                ckpt_exp_name=run_cfg["ckpt_exp_name"],
                ckpt_run_name=run_cfg["ckpt_run_name"],
                ckpt_filename=run_cfg["ckpt_filename"],
                max_steps=run_cfg.get("max_steps"),
            )

        sync_local_to_remote(
            local_path=local_run_registry.root,
            remote_path=str(remote_storage.run_registry_root),
            mode="copy",
            rclone_executable=cfg["rclone_executable"],
            extra_args=rclone_extra_args,
        )

        print(f"[run] finished and synced: {experiment_name}/{run_name}")


if __name__ == "__main__":
    main()
