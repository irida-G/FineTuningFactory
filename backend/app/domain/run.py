from enum import StrEnum

class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({
        RunStatus.RUNNING,
        RunStatus.CANCELLED,
    }),
    RunStatus.RUNNING: frozenset({
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

def can_transition(
        current: RunStatus,
        target: RunStatus,
) -> bool:
    return target in _ALLOWED_TRANSITIONS[current]