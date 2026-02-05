from dataclasses import dataclass


# -------------------------
# METRIC TRACKING SCHEMA
# -------------------------
@dataclass(frozen=True)
class SchemaColumns:
    step: str = "step"
    metric: str = "metric"
    value: str = "value"


# -------------------------
# METRIC CATEGORIES
# -------------------------
@dataclass(frozen=True)
class MetricCategories:
    network: str = "network"
    system: str = "system"
    train: str = "train"
    eval: str = "eval"
    
    def as_list(self) -> list[str]:
        """Return all categories as a list."""
        return [self.network, self.system, self.train, self.eval]


# -------------------------
# FILE & DIRECTORY NAMES
# -------------------------
@dataclass(frozen=True)
class FileNames:
    trainer_config: str = "trainer_configs.json"


@dataclass(frozen=True)
class RunDirNames:
    metadata: str = "metadata"
    metrics: str = "metrics"
    checkpoints: str = "checkpoints"
    tensorboard: str = "tb"
    
    def as_list(self) -> list[str]:
        """Return all directory names as a list."""
        return [self.metadata, self.metrics, self.checkpoints, self.tensorboard]


# -------------------------
# INSTANTIATE SINGLETONS
# -------------------------
SCHEMA = SchemaColumns()
METRICS = MetricCategories()
FILES = FileNames()
DIRS = RunDirNames()