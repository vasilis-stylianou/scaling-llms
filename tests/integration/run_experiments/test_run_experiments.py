from __future__ import annotations

import importlib.util
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml


# -----------------------
# Helpers
# -----------------------
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repo root")


def _load_run_experiments_script():
    script_path = _repo_root() / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("run_experiments_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_database_url(monkeypatch):
    import os

    db = os.getenv("DATABASE_URL")
    if not db:
        pytest.skip("DATABASE_URL not set")
    monkeypatch.setenv("DATABASE_URL", db)


def _has_rclone():
    try:
        subprocess.run(["rclone", "version"], check=True, capture_output=True)
        return True
    except Exception:
        return False


def _has_rclone_remote(name: str):
    try:
        result = subprocess.run(
            ["rclone", "listremotes"],
            check=True,
            text=True,
            capture_output=True,
        )
        return any(r.strip().rstrip(":") == name for r in result.stdout.splitlines())
    except Exception:
        return False


# -----------------------
# Dynamic experiment config
# -----------------------
def _write_experiment_config(tmp_path: Path, run_name: str) -> str:
    module_path = tmp_path / "test_experiment_config.py"

    content = f"""
EXPERIMENT_NAME = "test_run_experiments"

RUNS = [
    {{
        "method": "start",
        "run_name": "{run_name}",
        "dataset_kwargs": {{
            "dataset_name": "super_glue",
            "dataset_config": "cb",
            "train_split": "train[:1%]",
            "eval_split": "test[:1%]",
            "tokenizer_name": "gpt2_tiktoken",
            "text_field": "premise",
        }},
        "dataloader_kwargs": {{
            "seq_len": 16,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "start_sample_idx": 0,
            "seed": 1,
        }},
        "gpt_hparams": {{
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 1,
        }},
        "trainer_kwargs": {{
            "num_steps": 1,
            "lr": 3e-4,
            "accum_steps": 1,
            "lr_schedule": "linear",
            "enable_tb": False,
            "net_log_freq": 1,
            "sys_log_freq": 1,
            "eval_log_freq": 1,
            "ckpt_log_freq": 1,
        }},
        "max_steps": 2,
    }}
]
"""
    module_path.write_text(content)
    return module_path.stem  # importable module name


# -----------------------
# YAML writers
# -----------------------
def _write_yaml(path: Path, payload: dict[str, Any]):
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _make_local_yaml(path: Path, module: str, tmp_path: Path, suffix: str):
    _write_yaml(
        path,
        {
            "experiment_config_module": module,
            "registries": {
                "database_url_env_name": "DATABASE_URL",
                "runs": {
                    "artifacts_root": str(tmp_path / "runs"),
                    "table_name": f"runs_{suffix}",
                    "sync_hooks_type": None,
                    "sync_hooks_args": {},
                },
                "datasets": {
                    "artifacts_root": str(tmp_path / "datasets"),
                    "table_name": f"datasets_{suffix}",
                    "sync_hooks_type": None,
                    "sync_hooks_args": {},
                },
            },
        },
    )


def _make_rclone_yaml(path: Path, module: str, tmp_path: Path, suffix: str):
    _write_yaml(
        path,
        {
            "experiment_config_module": module,
            "registries": {
                "database_url_env_name": "DATABASE_URL",
                "runs": {
                    "artifacts_root": str(tmp_path / "runs"),
                    "table_name": f"runs_{suffix}",
                    "sync_hooks_type": "rclone",
                    "sync_hooks_args": {
                        "remote_rclone_name": "gdrive",
                        "remote_artifacts_root": "ml-experiments/scaling-llms-dev/test_run_experiments",
                    },
                },
                "datasets": {
                    "artifacts_root": str(tmp_path / "datasets"),
                    "table_name": f"datasets_{suffix}",
                    "sync_hooks_type": "rclone",
                    "sync_hooks_args": {
                        "remote_rclone_name": "gdrive",
                        "remote_artifacts_root": "ml-experiments/scaling-llms-dev/test_run_experiments",
                    },
                },
            },
        },
    )


def _assert_non_empty(path: Path):
    assert path.exists()
    assert any(p.is_file() for p in path.rglob("*"))


# -----------------------
# Tests
# -----------------------
def test_run_experiments_local(tmp_path, monkeypatch):
    _require_database_url(monkeypatch)

    run_experiments_script = _load_run_experiments_script()

    suffix = uuid.uuid4().hex[:6]
    run_name = f"test_run_{suffix}"

    module_name = _write_experiment_config(tmp_path, run_name)
    monkeypatch.syspath_prepend(str(tmp_path))

    yaml_path = tmp_path / "local.yaml"
    _make_local_yaml(yaml_path, module_name, tmp_path, suffix)

    cfg = run_experiments_script._load_config(yaml_path)
    mod = run_experiments_script._load_experiment_config_module(cfg["experiment_config_module"])
    exp_cfg = run_experiments_script._validate_experiment_config_module(mod)

    run_experiments_script.run_experiments(
        exp_cfg=exp_cfg,
        run_registry_cfg=cfg["registries"]["runs"],
        dataset_registry_cfg=cfg["registries"]["datasets"],
    )

    _assert_non_empty(tmp_path / "runs")
    _assert_non_empty(tmp_path / "datasets")


def test_run_experiments_rclone(tmp_path, monkeypatch):
    _require_database_url(monkeypatch)

    if not _has_rclone() or not _has_rclone_remote("gdrive"):
        pytest.skip("rclone or remote not available")

    run_experiments_script = _load_run_experiments_script()

    suffix = uuid.uuid4().hex[:6]
    run_name = f"test_run_{suffix}"

    module_name = _write_experiment_config(tmp_path, run_name)
    monkeypatch.syspath_prepend(str(tmp_path))

    yaml_path = tmp_path / "rclone.yaml"
    _make_rclone_yaml(yaml_path, module_name, tmp_path, suffix)

    cfg = run_experiments_script._load_config(yaml_path)
    mod = run_experiments_script._load_experiment_config_module(cfg["experiment_config_module"])
    exp_cfg = run_experiments_script._validate_experiment_config_module(mod)

    run_experiments_script.run_experiments(
        exp_cfg=exp_cfg,
        run_registry_cfg=cfg["registries"]["runs"],
        dataset_registry_cfg=cfg["registries"]["datasets"],
    )

    _assert_non_empty(tmp_path / "runs")
    _assert_non_empty(tmp_path / "datasets")