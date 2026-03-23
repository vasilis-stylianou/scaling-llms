from __future__ import annotations


class OrchestratorError(Exception):
    pass


class ConfigError(OrchestratorError):
    pass


class RunPodError(OrchestratorError):
    pass


class PodNotFoundError(RunPodError):
    pass


class PodNotReadyError(RunPodError):
    pass


class SSHError(OrchestratorError):
    pass


class ProvisioningError(OrchestratorError):
    pass


class JobLaunchError(OrchestratorError):
    pass