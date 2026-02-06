import os
from pathlib import Path
import subprocess
import sys

def _in_colab() -> bool:
    # Best-effort Colab detection
    return (
        ("COLAB_GPU" in os.environ)
        or ("COLAB_TPU_ADDR" in os.environ)
        or ("google.colab" in sys.modules)
    )

if _in_colab():
    print("Working in Colab")

    REPO_URL = "https://github.com/vasilis-stylianou/scaling-llms.git"
    REPO_DIR = Path("/content/scaling-llms")

    if not REPO_DIR.exists():
        print(f"Cloning {REPO_URL} -> {REPO_DIR}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)

    os.chdir(REPO_DIR)
    print("CWD:", Path.cwd())

    # Make imports work immediately (no kernel restart)
    src_path = str(REPO_DIR / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    print(f"Added {src_path} to sys.path")
    print("Repo ready; imports should work without restarting the kernel.")
    
else:
    print("Not in Colab, skipping setup")
