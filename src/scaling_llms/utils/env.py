import os
from pathlib import Path
import sys

LOCAL_TIMEZONE = "Europe/Athens"
DESKTOP_DRIVE_MOUNTPOINT = "/Users/vasilis/Library/CloudStorage/GoogleDrive-stylianouvasilis@gmail.com"
PROJECT_NAME = "scaling-llms"
REPO_URL = "https://github.com/vasilis-stylianou/scaling-llms.git"
REPO_DIR = "scaling-llms"
SRC_DIR = "src"


def is_colab() -> bool:
    colab_env_vars = ("COLAB_GPU", "COLAB_TPU_ADDR")
    return any(k in os.environ for k in colab_env_vars)

def setup_colab_repo(
    repo_url: str = REPO_URL,
    repo_dir: str = REPO_DIR,
    src_dir: str = SRC_DIR,
) -> None:
    if not is_colab():
        print("Not in Colab, skipping setup")
        return

    print("Working in Colab")
    repo_path = Path(repo_dir)

    if not repo_path.exists():
        os.system(f"git clone {repo_url}")

    os.chdir(repo_path)

    src_path = str(Path.cwd() / src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    print(f"Added {src_path} to sys.path")
