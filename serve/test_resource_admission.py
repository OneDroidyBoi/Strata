import concurrent.futures
from dataclasses import replace
import threading

import pytest

from serve.resource_admission import AdmissionRejected, ResourceLedger


def done(ledger, lease):
    return ledger.release(lease, iterator_closed=True, work_drained=True)


def test_fixed_costs_request_capacity_and_release():
    ledger = ResourceLedger(100, {"0": 80})
    worker = ledger.register_worker("a", "0", 2, 20, 30)
    lease = ledger.acquire(worker, 80, 50, 262144)
    with pytest.raises(AdmissionRejected):
        ledger.acquire(worker, 1, 0)
    assert ledger.snapshot()["host_bytes"] == 100
    assert done(ledger, lease)
    assert not done(ledger, lease)
    assert ledger.snapshot()["gpu_bytes"] == {"0": 30}


def test_atomic_last_slot_under_race():
    ledger = ResourceLedger(100, {"0": 100})
    worker = ledger.register_worker("a", "0", 1)
    barrier = threading.Barrier(12)
    def attempt(_):
        barrier.wait()
        try:
            return ledger.acquire(worker, 10, 10)
        except AdmissionRejected:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        leases = list(pool.map(attempt, range(12)))
    assert len([lease for lease in leases if lease]) == 1
    assert ledger.snapshot()["host_bytes"] == 10


def test_shared_host_capacity_and_independent_gpu_budgets():
    ledger = ResourceLedger(30, {"0": 10, "1": 20})
    a = ledger.register_worker("a", "0", 3)
    b = ledger.register_worker("b", "1", 3)
    ledger.acquire(a, 10, 10)
    ledger.acquire(b, 20, 20)
    with pytest.raises(AdmissionRejected):
        ledger.acquire(a, 0, 1)
    with pytest.raises(AdmissionRejected):
        ledger.acquire(b, 1, 0)
    assert ledger.snapshot()["active_reservations"] == 2


def test_cancellation_does_not_release_until_iterator_and_engine_finish():
    ledger = ResourceLedger(10, {"0": 10})
    worker = ledger.register_worker("a", "0", 1)
    lease = ledger.acquire(worker, 10, 10)
    for closed, drained in [(False, False), (True, False), (False, True), (1, True)]:
        with pytest.raises(ValueError):
            ledger.release(lease, iterator_closed=closed, work_drained=drained)
    with pytest.raises(AdmissionRejected):
        ledger.acquire(worker)
    assert done(ledger, lease)
    ledger.acquire(worker)


def test_drain_wait_and_generation_protection():
    ledger = ResourceLedger(20, {"0": 20})
    old = ledger.register_worker("a", "0", 1, 5, 5)
    lease = ledger.acquire(old, 5, 5)
    ledger.begin_drain(old)
    with pytest.raises(AdmissionRejected):
        ledger.acquire(old)
    with pytest.raises(AdmissionRejected):
        ledger.unregister_worker(old)
    assert not ledger.wait_quiescent(old, 0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(ledger.wait_quiescent, old, 1)
        done(ledger, lease)
        assert waiter.result(timeout=2)
    ledger.unregister_worker(old)
    new = ledger.register_worker("a", "0", 1, 5, 5)
    new_lease = ledger.acquire(new, 5, 5)
    assert new.generation != old.generation
    assert not done(ledger, lease)
    with pytest.raises(AdmissionRejected):
        ledger.acquire(old)
    assert ledger.snapshot()["active_reservations"] == 1
    assert done(ledger, new_lease)


def test_altered_and_foreign_leases_cannot_change_accounting():
    ledger = ResourceLedger(10, {"0": 10})
    worker = ledger.register_worker("a", "0", 1)
    lease = ledger.acquire(worker, 5, 5)
    assert not done(ledger, replace(lease, host_bytes=0))
    other = ResourceLedger(10, {"0": 10})
    assert not done(other, lease)
    assert ledger.snapshot()["host_bytes"] == 5


@pytest.mark.parametrize("bad", [True, False, -1, 1.2, "1", None])
def test_invalid_counts(bad):
    with pytest.raises(ValueError):
        ResourceLedger(bad, {"0": 10})
    with pytest.raises(ValueError):
        ResourceLedger(10, {"0": bad})
    ledger = ResourceLedger(10, {"0": 10})
    with pytest.raises(ValueError):
        ledger.register_worker("bad", "0", bad)
    worker = ledger.register_worker("a", "0", 1)
    for name in ("host_bytes", "gpu_bytes", "context_tokens"):
        with pytest.raises(ValueError):
            ledger.acquire(worker, **{name: bad})


@pytest.mark.parametrize("bad", [True, -1, 61, float("nan"), float("inf"), "1"])
def test_invalid_wait_timeout(bad):
    ledger = ResourceLedger(10, {"0": 10})
    worker = ledger.register_worker("a", "0", 1)
    with pytest.raises(ValueError):
        ledger.wait_quiescent(worker, bad)


def test_registration_failure_is_atomic_and_snapshot_detached():
    ledger = ResourceLedger(10, {"0": 10})
    worker = ledger.register_worker("a", "0", 1, 10, 10)
    with pytest.raises(AdmissionRejected):
        ledger.register_worker("b", "0", 1, 1, 1)
    with pytest.raises(AdmissionRejected):
        ledger.register_worker("a", "0", 1)
    snap = ledger.snapshot()
    snap["gpu_bytes"]["0"] = -100
    snap["workers"]["a"]["active"] = -100
    assert ledger.snapshot()["gpu_bytes"]["0"] == 10
    ledger.begin_drain(worker)
    ledger.unregister_worker(worker)
    assert ledger.snapshot()["host_bytes"] == 0


def test_cross_worker_budget_race_has_one_winner():
    ledger = ResourceLedger(10, {"0": 10, "1": 10})
    workers = [ledger.register_worker(str(i), str(i), 1) for i in range(2)]
    barrier = threading.Barrier(2)
    def attempt(worker):
        barrier.wait()
        try:
            return ledger.acquire(worker, 10, 10)
        except AdmissionRejected:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(attempt, workers))
    assert sum(lease is not None for lease in leases) == 1
    assert ledger.snapshot()["host_bytes"] == 10
    assert sum(ledger.snapshot()["gpu_bytes"].values()) == 10


def test_reopen_requires_same_registered_draining_quiescent_worker():
    ledger = ResourceLedger(30, {"0": 30})
    handle = ledger.register_worker("a", "0", 1, 5, 7)
    with pytest.raises(AdmissionRejected, match="draining and quiescent"):
        ledger.reopen_worker(handle)
    lease = ledger.acquire(handle, 10, 10)
    ledger.begin_drain(handle)
    with pytest.raises(AdmissionRejected, match="draining and quiescent"):
        ledger.reopen_worker(handle)
    assert ledger.snapshot()["host_bytes"] == 15
    done(ledger, lease)
    ledger.reopen_worker(handle)
    snap = ledger.snapshot()
    assert snap["workers"]["a"]["generation"] == handle.generation
    assert not snap["workers"]["a"]["draining"]
    assert snap["host_bytes"] == 5
    assert snap["gpu_bytes"] == {"0": 7}
    next_lease = ledger.acquire(handle, 10, 10)
    assert not done(ledger, lease)
    assert ledger.snapshot()["active_reservations"] == 1
    done(ledger, next_lease)
    ledger.begin_drain(handle)
    ledger.unregister_worker(handle)
    replacement = ledger.register_worker("a", "0", 1, 5, 7)
    ledger.begin_drain(replacement)
    with pytest.raises(AdmissionRejected, match="generation"):
        ledger.reopen_worker(handle)
    assert ledger.snapshot()["workers"]["a"]["draining"]


def test_reopen_race_allows_only_one_explicit_resume():
    ledger = ResourceLedger(10, {"0": 10})
    handle = ledger.register_worker("a", "0", 1)
    ledger.begin_drain(handle)
    barrier = threading.Barrier(8)
    def attempt(_):
        barrier.wait()
        try:
            ledger.reopen_worker(handle)
            return True
        except AdmissionRejected:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(8))) == 1
    assert not ledger.snapshot()["workers"]["a"]["draining"]
