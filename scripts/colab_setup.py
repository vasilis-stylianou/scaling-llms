from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import urllib.request


ENV_SETUP_URL = "https://raw.githubusercontent.com/vasilis-stylianou/scaling-llms/refs/heads/main/scripts/env_setup.py?cachebust=1"
LOCAL_ENV_SETUP_PATH = Path("./env_setup.py")

def _ensure_local_env_setup_file() -> Path:
    if LOCAL_ENV_SETUP_PATH.exists():
        return LOCAL_ENV_SETUP_PATH

    print(f"Downloading env setup helper from {ENV_SETUP_URL}")
    urllib.request.urlretrieve(ENV_SETUP_URL, LOCAL_ENV_SETUP_PATH)
    return LOCAL_ENV_SETUP_PATH


def _load_setup_env_for_git_commit():
    env_setup_path = _ensure_local_env_setup_file()
    spec = importlib.util.spec_from_file_location("env_setup", env_setup_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {env_setup_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    setup_fn = getattr(module, "setup_env_for_git_commit", None)
    if setup_fn is None:
        raise RuntimeError("env_setup.py does not expose setup_env_for_git_commit")
    return setup_fn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin Colab bootstrap that delegates to env_setup.setup_env_for_git_commit."
    )
    parser.add_argument(
        "--git-commit",
        dest="git_commit",
        type=str,
        default=None,
        help="Git commit SHA to checkout. If omitted, latest commit on main is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_fn = _load_setup_env_for_git_commit()
    setup_fn(args.git_commit)


if __name__ == "__main__":
    main()
