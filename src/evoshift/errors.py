class EvoShiftError(RuntimeError):
    """Base exception for user-facing EvoShift failures."""


class ConfigurationError(EvoShiftError):
    """Raised when a resolved configuration is invalid."""


class ProviderError(EvoShiftError):
    """Raised when an LLM provider cannot complete a request."""

    def __init__(self, message: str, status_code: int = 0, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class BudgetExceeded(EvoShiftError):
    """Raised before a request that would exceed a configured experiment budget."""


class DatasetError(EvoShiftError):
    """Raised when a benchmark cannot be downloaded, parsed, or verified."""


class EvolutionError(EvoShiftError):
    """Raised when an invalid candidate or policy mutation is proposed."""
