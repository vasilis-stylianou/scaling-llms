# scaling-llms

## Registry sync snippet (rclone + Google Drive)

Use this when you want to push a local run into a Google Drive-backed remote registry path
without changing runner logic yet.

```python
from pathlib import Path

from scaling_llms.registries.runs.identity import RunIdentity
from scaling_llms.registries.runs.registry import RunRegistry
from scaling_llms.storage.local_disk import make_local_run_registry


registry = make_local_run_registry(project_root="/tmp/scaling-llms-local")
identity = RunIdentity("my_experiment", "my_run")

# Example: source run directory/state prepared elsewhere.
src_run_dir = Path("/tmp/local_source_run")
src_run_state = registry.get_run_state(identity)

registry.sync_run_from_local(
	identity=identity,
	src_run_dir=src_run_dir,
	src_run_state=src_run_state,
	mode="gdrive-rclone",
	gdrive_project_subdir="scaling-llms-dev",
	gdrive_remote_subpath_override="ml-experiments",  # optional
)
```

Notes:
- `mode="gdrive-rclone"` is opt-in and fails fast if required config is missing.
- Local-only workflows should keep using `mode="shutil"` (or `"rclone"` for local-to-local rclone copy).
- The adapter maps to remote paths like `gdrive:ml-experiments/<project_subdir>/run_registry`.
