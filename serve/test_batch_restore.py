"""CPU integration: durable bundles, wire parser, routing, and admission.

The native wire peer is a deterministic in-memory double. No model process or
GPU is started; native snapshot serialization and device parity are not tested.
"""
from copy import deepcopy
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest

from serve.backends import ModelIdentity
from serve.checkpoint_store import CheckpointStore
from serve.native_lane import NativeLaneWorker
from serve.request_admission import RequestAdmission
from serve.resource_admission import AdmissionRejected
from serve.server import StrataEngine
from serve.test_native_lane import BatchEngine


class WireEngine(BatchEngine):
    session_batch_restore = True
    max_context = 100
    session_inventory = StrataEngine.session_inventory
    session_export = StrataEngine.session_export
    session_restore_slot = StrataEngine.session_restore_slot
    pick_slot = StrataEngine.pick_slot
    TURN_ID = StrataEngine.TURN_ID
    SLOT_PREFIX_MIN = StrataEngine.SLOT_PREFIX_MIN

    def __init__(self, model, fifo, states=None):
        super().__init__(model, fifo)
        self.states = deepcopy(states or {})
        self.slot_held = [list(self.states.get(i, [])) for i in range(2)]
        self.slot_order = [0, 1]
        self.slot_group = [0, 0]
        self.slot_gs = 0
        self.slot_used = [0.0, 0.0]
        self.lines = queue.Queue()
        self.commands = []
        self.epoch = 0

    def _send(self, command):
        assert self.fifo.locked() and self.ctl.locked()
        self.commands.append(command)
        parts = command.split(' ', 3)
        if parts[0] == 'INVENTORY':
            generation = f'wire-{self.epoch}'
            self.lines.put(f'INVENTORY {generation} {len(self.states)}')
            for slot, ids in sorted(self.states.items()):
                self.lines.put(f'SESSION_ENTRY slot-{slot} slot {len(ids)}')
            self.lines.put(f'INVENTORY_END {generation}')
        elif parts[0] == 'EXPORT':
            _, generation, sid, path = parts
            assert generation == f'wire-{self.epoch}'
            ids = self.states[int(sid.removeprefix('slot-'))]
            data = json.dumps(ids).encode()
            Path(path).write_bytes(data)
            self.lines.put(f'SAVED {len(ids)} {len(data)} 1.0')
        elif parts[0] == 'RESTORE_SLOT':
            _, slot_text, path = command.split(' ', 2)
            slot = int(slot_text)
            if slot in self.states:
                raise AssertionError('adapter dispatched a restore over an occupied slot')
            data = Path(path).read_bytes()
            ids = json.loads(data)
            self.states[slot] = list(ids)
            self.epoch += 1
            self.lines.put('SWAIT transfer 1')
            self.lines.put(f'SESSION {len(data)} {len(data)}')
            # Exercise accumulation of multiple prefix chunks in the real parser.
            for offset in range(0, len(ids), 2):
                self.lines.put(f'SLOT_IDS {slot} ' + ','.join(map(str, ids[offset:offset + 2])))
            self.lines.put(f'SLOT_RESTORED {slot} {len(ids)} {len(data)} 1.0')
        else:
            raise AssertionError('unexpected native command: ' + command)


class BatchRestoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.store = CheckpointStore(self.root)
        self.model = ModelIdentity('qwen4exp', 'complete-weight-inventory', 'tokenizer')
        self.prefixes = {0: [11, 12, 13, 14], 1: [21, 22, 23, 24, 25]}
        self.source, source_worker = self.lane(self.prefixes)
        saved = source_worker.persist_and_stop(self.store)
        self.assertEqual(saved.state, 'stopped', saved.error)
        self.bundle = saved.checkpoint_id
        # Reopen from disk: subsequent restores cannot rely on writer memory.
        self.store = CheckpointStore(self.root)
        self.sessions = self.store.sessions(self.bundle, self.model)
        self.ready, self.worker = self.lane()

    def lane(self, states=None, capacity=10000):
        controller = RequestAdmission(dict(host_capacity_bytes=capacity, gpu_capacity_bytes=capacity,
            gpu_id='0', max_requests=2, fixed_host_bytes=10, fixed_gpu_bytes=10,
            host_request_bytes=1, gpu_request_bytes=1, host_bytes_per_token=1,
            gpu_bytes_per_token=1, max_context_tokens=100, lookahead_tokens=0))
        service = SimpleNamespace(fifo=threading.Lock(), request_admission=controller)
        service.engine = WireEngine(self.model, service.fifo, states)
        return service, NativeLaneWorker(service, self.model, enabled=True)

    def restore(self, index, slot=None):
        return self.worker.restore_slot(self.store, self.bundle,
            self.sessions[index].session_id, index if slot is None else slot)

    def assert_restored(self, index):
        result = self.restore(index)
        self.assertEqual(result.state, 'restored', result.error)
        self.assertFalse(self.ready.fifo.locked())
        self.assertFalse(self.ready.engine.ctl.locked())
        with self.assertRaises(AdmissionRejected):
            self.ready.request_admission.acquire(1, 1)

    def test_two_durable_sessions_route_and_resume_with_two_reservations(self):
        self.assert_restored(0)
        self.assert_restored(1)
        engine = self.ready.engine
        self.assertEqual(engine.states, self.prefixes)
        self.assertEqual(engine.slot_held, [self.prefixes[0], self.prefixes[1]])
        with engine.slot_cv:
            # Reverse order defeats the empty/LRU default and tests exact prefix routing.
            self.assertEqual(engine.pick_slot(self.prefixes[1] + [26]), 1)
            self.assertEqual(engine.pick_slot(self.prefixes[0] + [15]), 0)
        result = self.worker.resume()
        self.assertEqual(result.state, 'resumed', result.error)
        self.assertFalse(self.ready.native_lane_gated)
        admission = self.ready.request_admission
        leases = [admission.acquire(len(self.prefixes[i]) + 1, 2) for i in range(2)]
        try:
            accounting = admission.ledger.snapshot()
            self.assertEqual(accounting['active_reservations'], 2)
            self.assertEqual(accounting['host_bytes'], 27)
            self.assertEqual(accounting['gpu_bytes'], {'0': 27})
            with self.assertRaises(AdmissionRejected):
                admission.acquire(1, 1)
        finally:
            for lease in leases:
                admission.ledger.release(lease, iterator_closed=True, work_drained=True)
        self.assertEqual(admission.ledger.snapshot()['host_bytes'], 10)

    def test_corrupt_payload_never_dispatches_native_restore(self):
        path = self.root / self.bundle / 'session-0000.bin'
        path.write_bytes(b'corrupted payload')
        result = self.restore(0)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('checksum', result.error)
        self.assertFalse(any(c.startswith('RESTORE_SLOT ') for c in self.ready.engine.commands))
        self.assertEqual(self.ready.engine.states, {})
        self.assertEqual(self.ready.engine.slot_held, [[], []])
        self.assertTrue(self.ready.engine.alive())
        self.assertEqual(self.worker.resume().state, 'blocked')
        with self.assertRaises(AdmissionRejected):
            self.ready.request_admission.acquire(1, 1)

    def test_occupied_slot_refusal_preserves_both_sessions(self):
        self.assert_restored(0)
        self.assert_restored(1)
        original = deepcopy(self.ready.engine.states)
        before = sum(c.startswith('RESTORE_SLOT ') for c in self.ready.engine.commands)
        result = self.restore(1, slot=0)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(self.ready.engine.states, original)
        self.assertEqual(self.ready.engine.slot_held, [self.prefixes[0], self.prefixes[1]])
        self.assertEqual(sum(c.startswith('RESTORE_SLOT ') for c in self.ready.engine.commands), before)
        self.assertTrue(self.ready.engine.alive())
        self.assertEqual(self.worker.resume().state, 'blocked')
        with self.assertRaises(AdmissionRejected):
            self.ready.request_admission.acquire(1, 1)

    def test_resumed_lane_still_obeys_byte_and_context_budgets(self):
        self.ready, self.worker = self.lane(capacity=25)
        self.assert_restored(0)
        self.assert_restored(1)
        result = self.worker.resume()
        self.assertEqual(result.state, 'resumed', result.error)
        admission = self.ready.request_admission
        lease = admission.acquire(8, 2)  # fixed 10 plus 11 request bytes
        try:
            with self.assertRaises(AdmissionRejected):
                admission.acquire(3, 2)  # one slot remains, but only four bytes
            with self.assertRaises(AdmissionRejected):
                admission.acquire(99, 2)
            self.assertEqual(admission.ledger.snapshot()['active_reservations'], 1)
        finally:
            admission.ledger.release(lease, iterator_closed=True, work_drained=True)


if __name__ == '__main__':
    unittest.main()
