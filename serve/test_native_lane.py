"""Native lane ownership integration with a fake protocol engine; CPU/files only."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

from serve.backends import ModelIdentity
from serve.checkpoint_store import CheckpointStore
from serve.native_lane import NativeLaneWorker
from serve.request_admission import RequestAdmission
from serve.resource_admission import AdmissionRejected


def admission():
    return RequestAdmission(dict(host_capacity_bytes=10000, gpu_capacity_bytes=10000,
        gpu_id='0', max_requests=2, fixed_host_bytes=10, fixed_gpu_bytes=10,
        host_request_bytes=1, gpu_request_bytes=1, host_bytes_per_token=1,
        gpu_bytes_per_token=1, max_context_tokens=100, lookahead_tokens=0))


class Process:
    def __init__(self):
        self.exited = False

    def poll(self):
        return 0 if self.exited else None


class Engine:
    batch = 0
    session_lifecycle = True

    def __init__(self, model, fifo):
        self.model_identity = model
        self.fifo = fifo
        self.proc = Process()
        self.precision_policy_report = {
            'enforced': True, 'draft': 'disabled',
            'artifact_audit': {'policy_passed': True,
                'policy': {'minimum_bits': 4, 'preferred_bits': 8},
                'components': {'target': {'files': [{'architecture': 'qwen4exp'}]}}},
            'pack_audit': {'packed_payload_verified_against_source': True}}
        self.generation = 'process1-epoch2'
        self.entries = [{'id': 'live', 'kind': 'live', 'tokens': 3},
                        {'id': 'parked-1', 'kind': 'parked', 'tokens': 2}]
        self.exports, self.restores = [], []
        self.close_count = 0
        self.fail_export = False
        self.mutate_export = False
        self.unconfirmed_exit = False
        self.inventory_calls = 0

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def session_inventory(self):
        assert self.fifo.locked()
        self.inventory_calls += 1
        return {'generation': self.generation, 'sessions': deepcopy(self.entries)}

    def session_export(self, generation, sid, path):
        assert self.fifo.locked()
        if generation != self.generation:
            raise ValueError('stale native generation')
        if self.fail_export:
            raise OSError('export failed')
        self.exports.append((generation, sid))
        Path(path).write_bytes(('state:' + sid).encode())
        if self.mutate_export:
            self.generation = 'process1-epoch3'

    def close(self):
        assert self.fifo.locked()
        self.close_count += 1
        if not self.unconfirmed_exit:
            self.proc.exited = True
        self.proc = None  # Dropping a handle is not proof of exit.

    def session_file(self, action, path):
        assert self.fifo.locked()
        assert action == 'restore'
        self.restores.append(Path(path).read_bytes())


class NativeLaneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = CheckpointStore(Path(self.tmp.name).resolve())
        self.model = ModelIdentity('qwen4exp', 'all-weights', 'tokenizer')
        self.service = SimpleNamespace(fifo=threading.Lock(), request_admission=admission())
        self.engine = Engine(self.model, self.service.fifo)
        self.service.engine = self.engine
        self.worker = NativeLaneWorker(self.service, self.model, enabled=True)

    def test_full_inventory_published_before_confirmed_stop(self):
        original = deepcopy(self.engine.entries)
        result = self.worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'stopped', result.error)
        sessions = self.store.sessions(result.checkpoint_id, self.model)
        self.assertEqual([s.session_id for s in sessions],
                         ['process1-epoch2:live', 'process1-epoch2:parked-1'])
        self.assertEqual(self.engine.exports, [('process1-epoch2', 'live'), ('process1-epoch2', 'parked-1')])
        self.assertEqual(self.engine.entries, original)
        self.assertEqual(self.engine.close_count, 1)
        accounting = self.service.request_admission.ledger.snapshot()
        self.assertEqual(accounting['host_bytes'], 0)
        self.assertEqual(accounting['gpu_bytes'], {'0': 0})
        self.assertEqual(accounting['workers'], {})
        self.assertTrue(self.service.native_lane_gated)
        self.assertFalse(self.service.fifo.locked())
        with self.assertRaises(AdmissionRejected):
            self.service.request_admission.acquire(1, 1)
        self.assertEqual(self.worker.persist_and_stop(self.store).checkpoint_id, result.checkpoint_id)
        self.assertEqual(self.engine.close_count, 1)

    def test_protocol_failure_blocks_lazy_restore_without_native_command(self):
        self.engine._session_protocol_failed = True
        result = self.worker.lazy_restore(self.store, '0' * 32, 'session')
        self.assertEqual(result.state, 'blocked')
        self.assertIn('protocol is blocked', result.error)
        self.assertEqual(self.engine.restores, [])
        self.assertEqual(self.engine.close_count, 0)

    def test_active_request_prevents_inventory_and_stop(self):
        lease = self.service.request_admission.acquire(1, 1)
        result = self.worker.persist_and_stop(self.store, timeout_s=0)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(self.engine.inventory_calls, 0)
        self.assertEqual(self.engine.close_count, 0)
        self.assertEqual(self.service.request_admission.ledger.snapshot()['active_reservations'], 1)
        self.service.request_admission.ledger.release(lease, iterator_closed=True, work_drained=True)
        self.assertEqual(self.worker.persist_and_stop(self.store).state, 'stopped')

    def test_fifo_timeout_keeps_admission_gated_and_engine_alive(self):
        self.service.fifo.acquire()
        result = self.worker.persist_and_stop(self.store, timeout_s=0)
        self.assertEqual(result.state, 'blocked')
        self.assertTrue(self.service.fifo.locked())
        self.service.fifo.release()
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.engine.inventory_calls, 0)

    def test_export_failure_never_unloads_and_releases_fifo(self):
        self.engine.fail_export = True
        result = self.worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(self.engine.close_count, 0)
        self.assertFalse(self.service.fifo.locked())
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.service.request_admission.ledger.snapshot()['host_bytes'], 10)

    def test_generation_change_blocks_stop(self):
        self.engine.mutate_export = True
        result = self.worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(self.engine.close_count, 0)
        self.assertFalse(self.service.fifo.locked())

    def test_handle_drop_is_not_confirmed_process_exit(self):
        self.engine.unconfirmed_exit = True
        result = self.worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('exit was not confirmed', result.error)
        self.assertIsNotNone(result.checkpoint_id)
        accounting = self.service.request_admission.ledger.snapshot()
        self.assertEqual(accounting['host_bytes'], 10)
        self.assertEqual(accounting['gpu_bytes'], {'0': 10})
        self.assertEqual(len(accounting['workers']), 1)

    def test_restore_selected_session_on_new_ready_lane(self):
        saved = self.worker.persist_and_stop(self.store)
        ready = SimpleNamespace(fifo=threading.Lock(), request_admission=admission())
        ready.engine = Engine(self.model, ready.fifo)
        worker = NativeLaneWorker(ready, self.model, enabled=True)
        result = worker.lazy_restore(self.store, saved.checkpoint_id, 'process1-epoch2:parked-1')
        self.assertEqual(result.state, 'restored', result.error)
        self.assertEqual(ready.engine.restores, [b'state:parked-1'])
        self.assertFalse(ready.fifo.locked())
        self.assertEqual(ready.engine.close_count, 0)
        with self.assertRaises(AdmissionRejected):
            ready.request_admission.acquire(1, 1)

    def test_identity_and_precision_fail_closed_before_control(self):
        for modify in (
            lambda e: setattr(e, 'session_lifecycle', False),
            lambda e: setattr(e, 'batch', 2),
            lambda e: setattr(e, 'model_identity', ModelIdentity('qwen4exp', 'other', 'tokenizer')),
            lambda e: e.precision_policy_report.update(enforced=False),
            lambda e: e.precision_policy_report.update(draft='enabled'),
            lambda e: e.precision_policy_report['artifact_audit']['policy'].update(minimum_bits=2),
            lambda e: e.precision_policy_report['artifact_audit']['policy'].update(preferred_bits=4),
            lambda e: e.precision_policy_report['pack_audit'].clear(),
        ):
            with self.subTest(modify=modify):
                engine = Engine(self.model, self.service.fifo)
                modify(engine)
                self.service.engine = engine
                with self.assertRaises(ValueError):
                    NativeLaneWorker(self.service, self.model, enabled=True)
                self.assertEqual(engine.inventory_calls, 0)
        with self.assertRaises(ValueError):
            NativeLaneWorker(self.service, self.model)

    def test_direct_protocol_calls_cannot_bypass_transaction(self):
        for call in (self.worker.gate_admission, self.worker.list_sessions, self.worker.unload):
            with self.assertRaises(RuntimeError):
                call()

    def test_empty_inventory_still_checks_generation_before_stop(self):
        self.engine.entries = []
        original = self.engine.session_inventory
        def changing_inventory():
            if self.engine.inventory_calls:
                self.engine.generation = 'process1-epoch3'
            return original()
        self.engine.session_inventory = changing_inventory
        result = self.worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('inventory changed', result.error)
        self.assertEqual(self.engine.close_count, 0)
        self.assertEqual(self.service.request_admission.ledger.snapshot()['host_bytes'], 10)

    def test_inventory_tokens_match_native_protocol(self):
        for generation, sid in [('a.b', 'live'), ('valid', 'a.b')]:
            with self.assertRaises(ValueError):
                self.worker._parse_inventory({'generation': generation,
                    'sessions': [{'id': sid, 'kind': 'live', 'tokens': 1}]})

    def test_bad_inventory_never_publishes_or_stops(self):
        for entries in ([{'id': 'live', 'kind': 'live', 'tokens': -1}],
                        [{'id': 'live', 'kind': 'live', 'tokens': True}],
                        self.engine.entries * 2):
            self.engine.entries = entries
            result = self.worker.persist_and_stop(self.store)
            self.assertEqual(result.state, 'blocked')
            self.assertEqual(self.engine.close_count, 0)
        self.assertFalse(self.service.fifo.locked())


if __name__ == '__main__':
    unittest.main()
