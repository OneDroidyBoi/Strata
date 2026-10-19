"""Batch shutdown transaction using real admission, receipts and durable store.

The protocol engine is a CPU-only double; no process, GPU or network is opened.
"""
from copy import deepcopy
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from serve.backends import ModelIdentity
from serve.checkpoint_store import CheckpointStore
from serve.drain_receipt import DrainReceipt
from serve.native_lane import NativeLaneWorker
from serve.request_admission import RequestAdmission
from serve.resource_admission import AdmissionRejected


class FakeProcess:
    def __init__(self):
        self.exited = False

    def poll(self):
        return 0 if self.exited else None


class BatchProtocolEngine:
    batch = 2
    slot_groups = 1
    session_lifecycle = True
    session_batch_export = True

    def __init__(self, service, model, root, synced):
        self.service, self.model_identity = service, model
        self.root, self.synced = root, synced
        self.proc = FakeProcess()
        self.original_proc = self.proc
        self.gen = 11
        self.ctl = threading.Lock()
        self.slot_cv = threading.Condition()
        self.slot_busy = [False, False]
        self.slot_live = [None, None]
        self.slot_claims, self.slot_releasing = set(), set()
        self.waiting = self.admitting = 0
        self.wait_lens = []
        self.starting = self._ctl_erred = False
        self.precision_policy_report = {
            'enforced': True, 'draft': 'disabled',
            'artifact_audit': {'policy_passed': True,
                'policy': {'minimum_bits': 4, 'preferred_bits': 8},
                'components': {'target': {'files': [{'architecture': 'qwen4exp'}]}}},
            'pack_audit': {'packed_payload_verified_against_source': True}}
        self.entries = [{'id': 'live', 'kind': 'live', 'tokens': 0},
                        {'id': 'slot-0', 'kind': 'slot', 'tokens': 80},
                        {'id': 'slot-1', 'kind': 'slot', 'tokens': 85}]
        self.inventory_calls = 0
        self.exports = []
        self.close_calls = 0
        self.fail_export = None
        self.exit_confirmed = True
        self.events = []

    def _assert_owned(self):
        assert self.service.fifo.locked()
        assert self.ctl.locked()
        assert self.service.request_admission.wait_drained()
        assert self.service.native_lane_gated

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def session_inventory(self):
        self._assert_owned()
        self.inventory_calls += 1
        return {'generation': 'process11-epoch2', 'sessions': deepcopy(self.entries)}

    def session_export(self, generation, session_id, path):
        self._assert_owned()
        assert generation == 'process11-epoch2'
        assert self.original_proc.poll() is None
        self.exports.append(session_id)
        if session_id == self.fail_export:
            raise OSError('one slot export failed')
        Path(path).write_bytes(('native-state:' + session_id).encode())
        self.events.append(('export', session_id))

    def session_file(self, action, path):
        raise AssertionError('shutdown must not restore')

    def close(self):
        self._assert_owned()
        self.close_calls += 1
        bundles = [path for path in self.root.iterdir() if not path.name.startswith('.')]
        assert len(bundles) == 1
        bundle = bundles[0]
        # Real store fsync was observed on every payload, manifest, and both
        # directories before close; merely existing files would not suffice.
        for path in [*bundle.glob('session-*.bin'), bundle / 'manifest.json', bundle, self.root]:
            assert path.stat().st_ino in self.synced, path
        store = CheckpointStore(self.root)
        sessions = store.sessions(bundle.name, self.model_identity)
        assert len(sessions) == len(self.entries)
        for session in sessions:
            payloads = []
            store.restore(bundle.name, session.session_id, self.model_identity,
                          lambda metadata, path: payloads.append(path.read_bytes()))
            assert payloads == [('native-state:' + session.session_id.split(':', 1)[1]).encode()]
        # Fixed residency cannot disappear until captured process exit.
        assert self.service.request_admission.ledger.snapshot()['host_bytes'] == 10
        self.events.append(('durable-before-exit', len(sessions)))
        if self.exit_confirmed:
            self.original_proc.exited = True
        self.proc = None


@pytest.fixture
def lane(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    synced = set()
    actual_fsync = os.fsync
    def observed_fsync(fd):
        actual_fsync(fd)
        synced.add(os.fstat(fd).st_ino)
    monkeypatch.setattr(os, 'fsync', observed_fsync)
    admission = RequestAdmission(dict(host_capacity_bytes=10000, gpu_capacity_bytes=10000,
        gpu_id='0', max_requests=2, fixed_host_bytes=10, fixed_gpu_bytes=20,
        host_request_bytes=1, gpu_request_bytes=1, host_bytes_per_token=1,
        gpu_bytes_per_token=1, max_context_tokens=100, lookahead_tokens=0))
    service = SimpleNamespace(fifo=threading.Lock(), request_admission=admission)
    model = ModelIdentity('qwen4exp', 'all-weights', 'tokenizer')
    service.engine = BatchProtocolEngine(service, model, root, synced)
    worker = NativeLaneWorker(service, model, enabled=True)
    return service, worker, CheckpointStore(root)


def pending_request(service, slot):
    receipt = DrainReceipt()
    tickets = []
    def events():
        tickets.append(receipt.begin(service.engine.gen, service.engine.proc, slot=slot))
        yield ('event', slot)
    iterator = service.request_admission.run(events, 10, 10, drain_receipt=receipt)
    assert next(iterator) == ('event', slot)
    iterator.close()
    return receipt, tickets[0]


def assert_no_checkpoint_or_stop(service, store):
    assert service.engine.inventory_calls == 0
    assert service.engine.exports == []
    assert service.engine.close_calls == 0
    assert list(store.root.iterdir()) == []


def test_two_closed_requests_wait_for_both_exact_receipts_before_checkpoint(lane):
    service, worker, store = lane
    first, t1 = pending_request(service, 0)
    second, t2 = pending_request(service, 1)
    assert service.request_admission.ledger.snapshot()['active_reservations'] == 2
    result = worker.persist_and_stop(store, timeout_s=0)
    assert result.state == 'blocked'
    assert_no_checkpoint_or_stop(service, store)
    assert service.native_lane_gated
    with pytest.raises(AdmissionRejected):
        service.request_admission.acquire(1, 1)
    assert not first.acknowledge(t1, 12, slot=0)
    assert not second.acknowledge(t2, 11, slot=0)
    assert_no_checkpoint_or_stop(service, store)
    assert first.acknowledge(t1, 11, slot=0)
    assert worker.persist_and_stop(store, timeout_s=0).state == 'blocked'
    assert_no_checkpoint_or_stop(service, store)
    assert second.acknowledge(t2, 11, slot=1)
    result = worker.persist_and_stop(store, timeout_s=1)
    assert result.state == 'stopped', result.error
    assert service.engine.exports == ['live', 'slot-0', 'slot-1']
    assert service.engine.original_proc.poll() == 0
    assert service.engine.events[-1] == ('durable-before-exit', 3)
    assert service.request_admission.ledger.snapshot()['workers'] == {}
    assert service.request_admission.ledger.snapshot()['host_bytes'] == 0
    assert not service.fifo.locked()
    assert not service.engine.ctl.locked()
    with pytest.raises(AdmissionRejected):
        service.request_admission.acquire(1, 1)


def test_export_failure_after_first_slot_never_unloads(lane):
    service, worker, store = lane
    receipts = [pending_request(service, slot) for slot in range(2)]
    for slot, (receipt, ticket) in enumerate(receipts):
        receipt.acknowledge(ticket, 11, slot=slot)
    service.engine.fail_export = 'slot-1'
    result = worker.persist_and_stop(store)
    assert result.state == 'blocked'
    assert 'export failed' in result.error
    assert service.engine.exports == ['live', 'slot-0', 'slot-1']
    assert service.engine.close_calls == 0
    assert service.engine.original_proc.poll() is None
    assert all(path.name.startswith('.pending-') for path in store.root.iterdir())
    assert service.request_admission.ledger.snapshot()['host_bytes'] == 10
    assert not service.fifo.locked()
    assert not service.engine.ctl.locked()
    with pytest.raises(AdmissionRejected):
        service.request_admission.acquire(1, 1)


def test_all_files_durable_but_unconfirmed_exit_retains_fixed_cost(lane):
    service, worker, store = lane
    service.engine.exit_confirmed = False
    result = worker.persist_and_stop(store)
    assert result.state == 'blocked'
    assert 'exit was not confirmed' in result.error
    assert result.checkpoint_id is not None
    assert service.engine.events[-1] == ('durable-before-exit', 3)
    assert service.engine.original_proc.poll() is None
    assert service.request_admission.ledger.snapshot()['host_bytes'] == 10
    assert service.request_admission.ledger.snapshot()['gpu_bytes'] == {'0': 20}
    assert service.native_lane_gated


def test_shutdown_waiter_unblocks_only_after_second_delayed_receipt(lane):
    from concurrent.futures import ThreadPoolExecutor
    service, worker, store = lane
    first, t1 = pending_request(service, 0)
    second, t2 = pending_request(service, 1)
    gated = threading.Event()
    original_gate = service.request_admission.gate_admission
    def signal_gate():
        original_gate()
        gated.set()
    service.request_admission.gate_admission = signal_gate
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.persist_and_stop, store, 2)
        try:
            assert gated.wait(timeout=1)
            assert not future.done()
            assert_no_checkpoint_or_stop(service, store)
            first.acknowledge(t1, 11, slot=0)
            assert service.request_admission.ledger.snapshot()['active_reservations'] == 1
            assert not future.done()
            assert_no_checkpoint_or_stop(service, store)
            assert not second.acknowledge(t2, 12, slot=1)
            assert not future.done()
            second.acknowledge(t2, 11, slot=1)
            result = future.result(timeout=2)
            assert result.state == 'stopped', result.error
            assert service.engine.close_calls == 1
        finally:
            first.acknowledge(t1, 11, slot=0)
            second.acknowledge(t2, 11, slot=1)
