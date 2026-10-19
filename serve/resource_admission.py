"""Opt-in, process-local resource reservations; no hardware inspection.

Estimates must include peak request allocations. This ledger cannot constrain native
allocations or coordinate separate processes. The caller must retain a reservation
until its iterator is closed/exhausted AND all associated engine work has drained.
"""
from dataclasses import dataclass
import math
import threading
import time
import uuid


class AdmissionRejected(RuntimeError):
    """The newest request or worker cannot fit the configured budget."""


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class WorkerHandle:
    worker_id: str
    generation: str


@dataclass(frozen=True)
class Reservation:
    worker: WorkerHandle
    reservation_id: str
    host_bytes: int
    gpu_bytes: int
    context_tokens: int


@dataclass
class _Worker:
    handle: WorkerHandle
    gpu_id: str
    max_slots: int
    fixed_host_bytes: int
    fixed_gpu_bytes: int
    draining: bool = False
    active: int = 0


class ResourceLedger:
    def __init__(self, host_bytes, gpu_bytes):
        self._host_limit = _count(host_bytes, "host_bytes")
        self._gpu_limits = dict(gpu_bytes)
        for gpu_id, limit in self._gpu_limits.items():
            if not isinstance(gpu_id, str) or not gpu_id:
                raise ValueError("gpu IDs must be nonempty strings")
            _count(limit, "gpu_bytes")
        self._condition = threading.Condition()
        self._host_used = 0
        self._gpu_used = dict.fromkeys(self._gpu_limits, 0)
        self._workers = {}
        self._reservations = {}

    def _worker(self, handle):
        worker = self._workers.get(handle.worker_id)
        if worker is None or worker.handle != handle:
            raise AdmissionRejected("worker generation is not registered")
        return worker

    def _fits(self, gpu_id, host_bytes, gpu_bytes):
        return (self._host_used + host_bytes <= self._host_limit and
                self._gpu_used[gpu_id] + gpu_bytes <= self._gpu_limits[gpu_id])

    def register_worker(self, worker_id, gpu_id, max_slots,
                        fixed_host_bytes=0, fixed_gpu_bytes=0):
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a nonempty string")
        _count(max_slots, "max_slots")
        if not max_slots:
            raise ValueError("max_slots must be positive")
        _count(fixed_host_bytes, "fixed_host_bytes")
        _count(fixed_gpu_bytes, "fixed_gpu_bytes")
        with self._condition:
            if gpu_id not in self._gpu_limits:
                raise ValueError("unknown GPU")
            if worker_id in self._workers:
                raise AdmissionRejected("worker is already registered")
            if not self._fits(gpu_id, fixed_host_bytes, fixed_gpu_bytes):
                raise AdmissionRejected("worker fixed costs exceed budget")
            handle = WorkerHandle(worker_id, uuid.uuid4().hex)
            self._workers[worker_id] = _Worker(
                handle, gpu_id, max_slots, fixed_host_bytes, fixed_gpu_bytes)
            self._host_used += fixed_host_bytes
            self._gpu_used[gpu_id] += fixed_gpu_bytes
            return handle

    def acquire(self, handle, host_bytes=0, gpu_bytes=0, context_tokens=0):
        _count(host_bytes, "host_bytes")
        _count(gpu_bytes, "gpu_bytes")
        _count(context_tokens, "context_tokens")
        with self._condition:
            worker = self._worker(handle)
            if worker.draining:
                raise AdmissionRejected("worker is draining")
            if worker.active >= worker.max_slots:
                raise AdmissionRejected("worker slots are full")
            if not self._fits(worker.gpu_id, host_bytes, gpu_bytes):
                raise AdmissionRejected("request exceeds available budget")
            lease = Reservation(handle, uuid.uuid4().hex, host_bytes,
                                gpu_bytes, context_tokens)
            self._reservations[lease.reservation_id] = lease
            worker.active += 1
            self._host_used += host_bytes
            self._gpu_used[worker.gpu_id] += gpu_bytes
            return lease

    def release(self, reservation, *, iterator_closed, work_drained):
        """Release only after both lifecycle conditions are positively confirmed.

        Returns False for a duplicate, stale, foreign, or altered reservation.
        Cancellation alone is not confirmation of native engine quiescence.
        """
        if iterator_closed is not True or work_drained is not True:
            raise ValueError("release requires a closed iterator and drained engine work")
        with self._condition:
            stored = self._reservations.get(reservation.reservation_id)
            if stored != reservation:
                return False
            worker = self._worker(stored.worker)
            del self._reservations[stored.reservation_id]
            worker.active -= 1
            self._host_used -= stored.host_bytes
            self._gpu_used[worker.gpu_id] -= stored.gpu_bytes
            self._condition.notify_all()
            return True

    def begin_drain(self, handle):
        """Block new admissions; existing work keeps its reservations."""
        with self._condition:
            self._worker(handle).draining = True
            self._condition.notify_all()

    def reopen_worker(self, handle):
        """Explicitly resume the same registered worker after a complete drain.

        This does not validate engine/checkpoint readiness: the lifecycle owner
        must do so before calling. Fixed residency and generation stay intact.
        """
        with self._condition:
            worker = self._worker(handle)
            if not worker.draining or worker.active:
                raise AdmissionRejected("worker must be draining and quiescent before resume")
            worker.draining = False
            self._condition.notify_all()

    def wait_quiescent(self, handle, timeout=0):
        """Wait at most 60 seconds for leases to release; never cancels work."""
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
                not math.isfinite(timeout) or not 0 <= timeout <= 60):
            raise ValueError("timeout must be finite and between 0 and 60 seconds")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._worker(handle).active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def unregister_worker(self, handle):
        """Remove fixed costs only after drain and full request quiescence."""
        with self._condition:
            worker = self._worker(handle)
            if not worker.draining or worker.active:
                raise AdmissionRejected("worker must be draining and quiescent")
            self._host_used -= worker.fixed_host_bytes
            self._gpu_used[worker.gpu_id] -= worker.fixed_gpu_bytes
            del self._workers[handle.worker_id]
            self._condition.notify_all()

    def snapshot(self):
        """Return detached accounting values for diagnostics."""
        with self._condition:
            return {"host_bytes": self._host_used,
                    "gpu_bytes": dict(self._gpu_used),
                    "active_reservations": len(self._reservations),
                    "workers": {key: {"generation": w.handle.generation,
                                      "active": w.active, "draining": w.draining}
                                for key, w in self._workers.items()}}
