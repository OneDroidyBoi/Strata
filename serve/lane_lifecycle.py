"""Opt-in persist-and-stop orchestration over an injected lane worker.

No native Strata mapping is supplied: session_file(SAVE/RESTORE) addresses one
current conversation under the service FIFO, not an enumerable set of all slots.
Backend durable-checkpoint/migration capabilities therefore remain false.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
from typing import Protocol, Sequence

from serve.backends import ModelIdentity, SessionMetadata
from serve.checkpoint_store import CheckpointStore


class LaneWorker(Protocol):
    """All methods must be serialized with the worker's ordinary serving owner.

    gate_admission is sticky and must prevent new work until explicitly reopened
    by that owner. wait_drained includes device work, cancellation drains and
    state mutation. list_sessions must enumerate EVERY restorable session,
    including parked state; export must be non-destructive. unload must return
    only when stopped, or raise if stop cannot be confirmed.
    """
    def gate_admission(self) -> None: ...
    def wait_drained(self, timeout_s: float) -> bool: ...
    def list_sessions(self) -> Sequence[SessionMetadata]: ...
    def export_session(self, session: SessionMetadata, path: Path) -> None: ...
    def unload(self) -> None: ...
    def restore_session(self, session: SessionMetadata, path: Path) -> None: ...


@dataclass(frozen=True)
class LifecycleResult:
    state: str  # stopped, restored, blocked; blocked never promises worker liveness
    checkpoint_id: str | None = None
    error: str | None = None


class LaneLifecycle:
    def __init__(self, worker: LaneWorker, store: CheckpointStore, model: ModelIdentity):
        if not isinstance(model, ModelIdentity):
            raise TypeError("model must be ModelIdentity")
        self.worker, self.store, self.model = worker, store, model
        self._lock = threading.Lock()
        self._stopped = False
        self._checkpoint_id = None

    @staticmethod
    def _timeout(timeout_s):
        if not isinstance(timeout_s, (float, int)) or not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and nonnegative")

    def persist_and_stop(self, timeout_s: float = 30.0) -> LifecycleResult:
        self._timeout(timeout_s)
        if not self._lock.acquire(blocking=False):
            return LifecycleResult("blocked", error="lane lifecycle operation already running")
        checkpoint_id = None
        try:
            if self._stopped:
                return LifecycleResult("stopped", self._checkpoint_id)
            self.worker.gate_admission()
            if not self.worker.wait_drained(timeout_s):
                raise RuntimeError("lane did not drain; admission remains gated")
            sessions = tuple(self.worker.list_sessions())
            if any(not isinstance(s, SessionMetadata) or s.model != self.model for s in sessions):
                raise ValueError("lane contains incompatible session metadata")
            checkpoint_id = self.store.publish(self.model, sessions, self.worker.export_session)
            # Detect a worker violating its gate/drain contract. Never unload on
            # incomplete enumeration or changes during export.
            if tuple(self.worker.list_sessions()) != sessions:
                raise RuntimeError("lane sessions changed while checkpointing")
            self.worker.unload()
            self._stopped = True
            self._checkpoint_id = checkpoint_id
            return LifecycleResult("stopped", checkpoint_id)
        except Exception as exc:
            # No fallback kill/unload, and no automatic admission reopening.
            # An exporter/unloader may itself fail its worker; do not claim it
            # is alive merely because this controller did not stop it.
            return LifecycleResult("blocked", checkpoint_id, f"{type(exc).__name__}: {exc}")
        finally:
            self._lock.release()

    def lazy_restore(self, checkpoint_id: str, session_id: str,
                     timeout_s: float = 30.0) -> LifecycleResult:
        """Restore only the requested session to a caller-provided ready worker.

        Admission remains gated; opening it and any engine load are owned by the
        caller. A failed restore may have mutated native state; blocked is not
        permission to serve that state or retry blindly.
        """
        self._timeout(timeout_s)
        if not self._lock.acquire(blocking=False):
            return LifecycleResult("blocked", checkpoint_id, "lane lifecycle operation already running")
        try:
            if self._stopped:
                raise RuntimeError("use a new lifecycle with a ready worker after stop")
            self.worker.gate_admission()
            if not self.worker.wait_drained(timeout_s):
                raise RuntimeError("restore lane did not drain")
            self.store.restore(checkpoint_id, session_id, self.model, self.worker.restore_session)
            return LifecycleResult("restored", checkpoint_id)
        except Exception as exc:
            return LifecycleResult("blocked", checkpoint_id, f"{type(exc).__name__}: {exc}")
        finally:
            self._lock.release()
