import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scaling_llms.registries.core.metadata_backend import MetadataBackend


_REQUIRED_REGISTRY_KEYS = {
    "artifacts_root",
    "table_name",
}
_VALID_SYNC_HOOK_TYPES = {None, "rclone"}


@dataclass(slots=True)
class MakeRegistryConfig:
    table_name: str
    artifacts_root: str
    database_url: str | None = None
    backend: MetadataBackend | None = None
    sync_hooks_type: str | None = None
    sync_hooks_args: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.database_url is None and self.backend is None:
            raise ValueError(
                "MakeRegistryConfig requires either database_url or backend"
            )

    @classmethod
    def from_raw(
        cls,
        *,
        name: str,
        data: dict[str, Any],
        env_var_name: str,
    ) -> "MakeRegistryConfig":
        if not isinstance(data, dict):
            raise ValueError(f"registries.{name} must be a mapping")

        missing = _REQUIRED_REGISTRY_KEYS.difference(data.keys())
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(
                f"registries.{name} missing required keys: {missing_keys}"
            )

        artifacts_root = data["artifacts_root"]
        if not isinstance(artifacts_root, str) or not artifacts_root.strip():
            raise ValueError(
                f"registries.{name}.artifacts_root must be a non-empty path string"
            )

        table_name = data["table_name"]
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError(
                f"registries.{name}.table_name must be a non-empty string"
            )

        sync_hooks_type = data.get("sync_hooks_type")
        if sync_hooks_type not in _VALID_SYNC_HOOK_TYPES:
            allowed = ", ".join(
                sorted(x for x in _VALID_SYNC_HOOK_TYPES if x is not None)
            )
            raise ValueError(
                f"registries.{name}.sync_hooks_type must be one of: None, {allowed}"
            )

        sync_hooks_args = data.get("sync_hooks_args")
        if sync_hooks_type is None:
            if sync_hooks_args is not None and not isinstance(sync_hooks_args, dict):
                raise ValueError(
                    f"registries.{name}.sync_hooks_args must be a dict or null"
                )
        else:
            if sync_hooks_args is None:
                raise ValueError(
                    f"registries.{name}.sync_hooks_args must be provided when "
                    f"sync_hooks_type={sync_hooks_type!r}"
                )
            if not isinstance(sync_hooks_args, dict):
                raise ValueError(
                    f"registries.{name}.sync_hooks_args must be a dict"
                )

        database_url = data.get("database_url")
        backend = data.get("backend")

        if database_url is None or (
            isinstance(database_url, str) and not database_url.strip()
        ):
            database_url = os.getenv(env_var_name)

        if (
            database_url is None
            or not isinstance(database_url, str)
            or not database_url.strip()
        ):
            database_url = None
        else:
            database_url = database_url.strip()

        return cls(
            database_url=database_url,
            backend=backend,
            table_name=table_name.strip(),
            artifacts_root=str(
                Path(artifacts_root).expanduser().resolve()
            ),
            sync_hooks_type=sync_hooks_type,
            sync_hooks_args=sync_hooks_args,
        )