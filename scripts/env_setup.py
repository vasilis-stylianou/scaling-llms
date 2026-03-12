from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_URL = "https://github.com/vasilis-stylianou/scaling-llms.git"
REPO_DIR = Path("/content/scaling-llms")
MAIN_REF = "refs/heads/main"


def _in_colab() -> bool:
    return (
        ("COLAB_GPU" in os.environ)
        or ("COLAB_TPU_ADDR" in os.environ)
        or ("google.colab" in sys.modules)
    )


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )


def _run_stdout(cmd: list[str], cwd: Path | None = None) -> str:
    return _run(cmd, cwd=cwd).stdout.strip()


def _latest_main_commit(repo_url: str) -> str:
    line = _run_stdout(["git", "ls-remote", repo_url, MAIN_REF])
    if not line:
        raise RuntimeError(f"Could not resolve latest commit for {MAIN_REF} from {repo_url}")
    return line.split()[0]


def _normalize_repo_url(url: str) -> str:
    normalized = url.strip().lower().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def _repo_is_valid_git_checkout(repo_dir: Path) -> bool:
    if not repo_dir.exists() or not repo_dir.is_dir():
        return False

    try:
        inside = _run_stdout(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_dir)
        if inside != "true":
            return False

        origin_url = _run_stdout(["git", "config", "--get", "remote.origin.url"], cwd=repo_dir)
    except Exception:
        return False

    return _normalize_repo_url(origin_url) == _normalize_repo_url(REPO_URL)


def _ensure_repo(repo_url: str, repo_dir: Path) -> None:
    if repo_dir.exists() and not _repo_is_valid_git_checkout(repo_dir):
        print(f"Found invalid repo checkout at {repo_dir}; deleting and recloning.")
        shutil.rmtree(repo_dir)

    if repo_dir.exists():
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {repo_url} -> {repo_dir}")
    _run(["git", "clone", repo_url, str(repo_dir)])


def _checkout_commit(repo_dir: Path, commit: str) -> str:
    _run(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)
    _run(["git", "reset", "--hard"], cwd=repo_dir)
    _run(["git", "clean", "-fd"], cwd=repo_dir)
    _run(["git", "checkout", "--detach", commit], cwd=repo_dir)
    return _run_stdout(["git", "rev-parse", "HEAD"], cwd=repo_dir)


def _restart_kernel() -> None:
    try:
        if _in_colab():
            from google.colab import output  # type: ignore

            output.eval_js("google.colab.kernel.restart()")
            return
    except Exception:
        pass

    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip and hasattr(ip, "kernel") and ip.kernel is not None:
            ip.kernel.do_shutdown(restart=True)
            return
    except Exception as exc:
        raise RuntimeError("Automatic kernel restart failed") from exc

    raise RuntimeError("Automatic kernel restart failed: no compatible notebook kernel API available")


def setup_env_for_git_commit(git_commit: str | None = None) -> str | None:
    if _in_colab():
        os.environ["SCALING_LLMS_ENV"] = "colab"
        print("Working environment: colab")

        target_commit = git_commit or _latest_main_commit(REPO_URL)
        print(f"Target commit: {target_commit}")

        _ensure_repo(REPO_URL, REPO_DIR)
        resolved_commit = _checkout_commit(REPO_DIR, target_commit)
        print(f"Resolved commit: {resolved_commit}")

        os.chdir(REPO_DIR)
        print("CWD:", Path.cwd())

        src_path = str(REPO_DIR / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            print(f"Added to sys.path: {src_path}")
        else:
            print(f"Already in sys.path: {src_path}")

        os.environ["SCALING_LLMS_REPO_DIR"] = str(REPO_DIR)
        os.environ["SCALING_LLMS_GIT_COMMIT"] = resolved_commit

        print("Repo ready; imports should work without restarting the kernel.")
        
        return resolved_commit

    os.environ["SCALING_LLMS_ENV"] = "local"
    print("Working environment: local")
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
            print("Enabled autoreload")
    except Exception:
        pass

    print("Not in Colab, skipping git/repo setup")
    return None


def setup_env_from_run_identity(
    run_identity: object,
    *,
    project_subdir: str | None = None,
    auto_restart_kernel: bool = True,
) -> str:
    if not hasattr(run_identity, "experiment_name") or not hasattr(run_identity, "run_name"):
        raise TypeError("run_identity must have 'experiment_name' and 'run_name' attributes")

    # This wrapper intentionally reads registry metadata using the currently imported
    # project code before switching commits, assuming that registry access remains
    # backward-compatible enough to fetch run metadata.
    from scaling_llms.constants import PROJECT_DEV_NAME, PROJECT_NAME
    from scaling_llms.registries.runs.identity import RunIdentity
    from scaling_llms.storage.google_drive import make_gdrive_run_registry

    identity = RunIdentity(
        experiment_name=str(getattr(run_identity, "experiment_name")),
        run_name=str(getattr(run_identity, "run_name")),
    )

    candidate_subdirs = [project_subdir] if project_subdir is not None else [PROJECT_DEV_NAME, PROJECT_NAME]
    checked_subdirs: list[str] = []

    for subdir in candidate_subdirs:
        if subdir in checked_subdirs:
            continue
        checked_subdirs.append(subdir)

        registry = make_gdrive_run_registry(project_subdir=subdir)
        if not registry.run_exists(identity):
            continue

        commit = registry.get_git_commit(identity)
        if not commit:
            raise ValueError(
                f"Run {identity} exists in project_subdir='{subdir}' but has no logged git_commit"
            )

        resolved_commit = setup_env_for_git_commit(commit)
        if resolved_commit is None:
            raise RuntimeError("Failed to setup environment for resolved git commit")

        if auto_restart_kernel:
            print("Restarting kernel for safety...")
            _restart_kernel()

        return resolved_commit

    raise FileNotFoundError(f"Run {identity} not found in project subdirs: {checked_subdirs}")


__all__ = [
    "setup_env_for_git_commit",
    "setup_env_from_run_identity",
]
