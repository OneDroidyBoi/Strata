"""CPU integration for durable cross-lane migration and parked-agent rotation.

Real worker, store, protocol parser, admission and coordinator; only the native
wire peer/device state are fake. This is not GPU snapshot parity validation.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from serve.backends import ModelIdentity
from serve.checkpoint_store import CheckpointStore
from serve.lane_migration import MigrationCoordinator
from serve.native_lane import NativeLaneWorker
from serve.test_batch_restore import WireEngine
from serve.test_native_lane import admission


class MigrationWireEngine(WireEngine):
    session_batch_route = True
    session_batch_release = True

    def __init__(self, model, fifo, states, name, events):
        super().__init__(model, fifo, states)
        self.name, self.events = name, events
        self.local = threading.local()
        self.pending = []
        self.dispatched = []
        self.defer_ack = False

    @contextmanager
    def bound_slot(self, slot):
        previous = getattr(self.local, 'slot', None)
        self.local.slot = slot
        try:
            yield
        finally:
            self.local.slot = previous

    @contextmanager
    def track_drain(self, receipt):
        previous = getattr(self.local, 'receipt', None)
        self.local.receipt = receipt
        try:
            yield receipt
        finally:
            self.local.receipt = previous

    def generate(self, ids, max_new, sampling, cancel):
        slot, receipt = self.local.slot, self.local.receipt
        ticket = receipt.begin(self.gen, self.proc, slot)
        self.dispatched.append((list(ids), slot))
        with self.slot_cv:
            self.slot_busy[slot] = True
        pending = (receipt, ticket, self.gen, slot)
        self.pending.append(pending)
        try:
            yield 777
        finally:
            if not self.defer_ack:
                self.acknowledge(pending)

    def acknowledge(self, pending):
        receipt, ticket, generation, slot = pending
        assert receipt.acknowledge(ticket, generation, slot)
        with self.slot_cv:
            self.slot_busy[slot] = False
            self.slot_cv.notify_all()
        self.pending.remove(pending)

    def _send(self, command):
        self.events.append((self.name, command.split()[0]))
        if command.startswith('EXPORT ') and self.fail_export:
            raise OSError('injected snapshot export failure')
        if command.startswith('RELEASE_SLOT '):
            assert self.fifo.locked() and self.ctl.locked()
            slot = int(command.split()[1])
            self.commands.append(command)
            self.states.pop(slot, None)
            self.epoch += 1
            self.lines.put(f'SLOT_RELEASED {slot}')
            return
        return super()._send(command)

    def session_release_slot(self, slot):
        from serve.session_protocol import SessionProtocol
        return SessionProtocol(self).release_slot(slot)

    def close(self):
        self.events.append((self.name, 'CLOSE'))
        super().close()


class LaneMigrationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.store = CheckpointStore(self.root)
        self.model = ModelIdentity('qwen4exp', 'all-weights', 'tokenizer')
        self.events, self.services, self.workers, self.original = [], {}, {}, {}
        agents = {}
        for lane_index in range(3):
            name = f'lane-{lane_index}'
            states = {}
            for slot in range(2):
                agent = f'agent-{2 * lane_index + slot}'
                self.original[agent] = [100 * lane_index + 10 * slot + i for i in range(1, 5)]
                states[slot] = list(self.original[agent])
                agents[agent] = (name, slot)
            service = SimpleNamespace(fifo=threading.Lock(), request_admission=admission())
            service.engine = MigrationWireEngine(self.model, service.fifo, states, name, self.events)
            self.services[name] = service
            self.workers[name] = NativeLaneWorker(service, self.model, enabled=True)
        self.coordinator = MigrationCoordinator(self.workers, self.store, agents, enabled=True)
        self.destination = 'lane-2'

    def consolidate(self):
        result = self.coordinator.consolidate(self.destination)
        self.assertEqual(result.state, 'consolidated', result.error)
        return result

    def test_three_lanes_consolidate_with_four_agents_durably_parked(self):
        self.consolidate()
        routes = self.coordinator.routes
        self.assertEqual(set(routes), set(self.original))
        resident = {a: r for a, r in routes.items() if r.lane is not None}
        parked = {a: r for a, r in routes.items() if r.lane is None}
        self.assertEqual(len(resident), 2)
        self.assertEqual(len(parked), 4)
        self.assertEqual({r.lane for r in resident.values()}, {self.destination})
        self.assertEqual({r.slot for r in resident.values()}, {0, 1})
        for agent, route in routes.items():
            saved = []
            self.store.restore(route.checkpoint_id, route.session_id, self.model,
                               lambda session, path: saved.append(json.loads(path.read_bytes())))
            self.assertEqual(saved, [self.original[agent]])
        first_stop = next(i for i, (_, command) in enumerate(self.events) if command == 'CLOSE')
        self.assertEqual(sum(command == 'EXPORT' for _, command in self.events[:first_stop]), 6)
        for name, service in self.services.items():
            snapshot = service.request_admission.ledger.snapshot()
            self.assertEqual(snapshot['host_bytes'], 10 if name == self.destination else 0)
            self.assertEqual(service.engine.close_count, 0 if name == self.destination else 1)
        self.assertTrue(Path(self.coordinator.journal_path).is_file())

    def test_rotation_preserves_prefix_advanced_since_consolidation(self):
        self.consolidate()
        routes = self.coordinator.routes
        resident = {a: r for a, r in routes.items() if r.lane is not None}
        engine = self.services[self.destination].engine
        advanced = {}
        for agent, route in resident.items():
            advanced[agent] = self.original[agent] + [901, 902]
            engine.states[route.slot] = list(advanced[agent])
            engine.slot_held[route.slot] = list(advanced[agent])
        engine.epoch += 1  # completed native generation, with its receipt already drained
        incoming = next(a for a, r in routes.items() if r.lane is None)
        result = self.coordinator.activate(incoming)
        self.assertEqual(result.state, 'ready', result.error)
        updated = self.coordinator.routes
        evicted = next(a for a in resident if updated[a].lane is None)
        self.assertEqual(engine.states[updated[incoming].slot], self.original[incoming])
        result = self.coordinator.activate(evicted)
        self.assertEqual(result.state, 'ready', result.error)
        final = self.coordinator.routes
        self.assertEqual(engine.states[final[evicted].slot], advanced[evicted])
        self.assertEqual(engine.slot_held[final[evicted].slot], advanced[evicted])
        self.assertEqual(len([r for r in final.values() if r.lane is None]), 4)
        service = self.services[self.destination]
        self.assertFalse(service.native_lane_gated)
        lease = service.request_admission.acquire(8, 1)
        service.request_admission.ledger.release(lease, iterator_closed=True, work_drained=True)

    def test_export_failure_never_stops_any_lane(self):
        self.services['lane-1'].engine.fail_export = True
        result = self.coordinator.consolidate(self.destination)
        self.assertEqual(result.state, 'blocked')
        for service in self.services.values():
            self.assertEqual(service.engine.close_count, 0)
            self.assertTrue(service.engine.alive())
            self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 10)
        self.assertFalse(any(command == 'CLOSE' for _, command in self.events))

    def test_uncertain_source_exit_retains_fixed_resource_charges(self):
        source = self.services['lane-0']
        source.engine.unconfirmed_exit = True
        result = self.coordinator.consolidate(self.destination)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(source.engine.close_count, 1)
        snapshot = source.request_admission.ledger.snapshot()
        self.assertEqual(snapshot['host_bytes'], 10)
        self.assertEqual(snapshot['gpu_bytes'], {'0': 10})
        self.assertEqual(len(snapshot['workers']), 1)

    def test_generate_distinct_agents_concurrently_dispatches_bound_slots(self):
        self.consolidate()
        resident = [(a, r) for a, r in self.coordinator.routes.items() if r.lane is not None]
        rendezvous = threading.Barrier(3)
        finish = threading.Event()
        engine = self.services[self.destination].engine
        def consume(agent):
            iterator = self.coordinator.generate(agent, self.original[agent] + [42],
                                                  2, {}, threading.Event())
            try:
                self.assertEqual(next(iterator), 777)
                rendezvous.wait(timeout=2)
                self.assertTrue(finish.wait(2))
                self.assertEqual(list(iterator), [])
            finally:
                iterator.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(consume, agent) for agent, _ in resident]
            try:
                rendezvous.wait(timeout=2)
                self.assertEqual(self.coordinator._active, {a for a, _ in resident})
                self.assertEqual(self.services[self.destination].request_admission.ledger.snapshot()['active_reservations'], 2)
                actual = {tuple(ids): slot for ids, slot in engine.dispatched}
                for agent, route in resident:
                    self.assertEqual(actual[tuple(self.original[agent] + [42])], route.slot)
            finally:
                finish.set()
            for job in jobs:
                job.result(timeout=2)
        self.assertEqual(engine.pending, [])
        self.assertEqual(self.services[self.destination].request_admission.ledger.snapshot()['active_reservations'], 0)

    def test_consolidation_waits_for_turn_end_and_exact_native_drain(self):
        service = self.services['lane-0']
        engine = service.engine
        engine.defer_ack = True
        iterator = self.coordinator.generate('agent-0', self.original['agent-0'] + [42],
                                              2, {}, threading.Event())
        self.assertEqual(next(iterator), 777)
        turn_wait, drain_wait = threading.Event(), threading.Event()
        class ObservedCondition(threading.Condition):
            def wait(self, timeout=None):
                turn_wait.set()
                return super().wait(timeout)
        self.coordinator._cv = ObservedCondition()
        original_wait = service.request_admission.wait_drained
        def observed_drain(timeout_s):
            drain_wait.set()
            return original_wait(timeout_s)
        with patch.object(service.request_admission, 'wait_drained', observed_drain):
            with ThreadPoolExecutor(max_workers=1) as pool:
                job = pool.submit(self.coordinator.consolidate, self.destination, 3)
                try:
                    self.assertTrue(turn_wait.wait(1))
                    self.assertFalse(any(command == 'EXPORT' for _, command in self.events))
                    iterator.close()
                    self.assertTrue(drain_wait.wait(1))
                    self.assertEqual(service.request_admission.ledger.snapshot()['active_reservations'], 1)
                    self.assertFalse(any(command in ('EXPORT', 'CLOSE') for _, command in self.events))
                    receipt, ticket, generation, slot = engine.pending[0]
                    self.assertFalse(receipt.acknowledge(ticket, generation, 1))
                    self.assertEqual(service.request_admission.ledger.snapshot()['active_reservations'], 1)
                    engine.acknowledge(engine.pending[0])
                    result = job.result(timeout=2)
                    self.assertEqual(result.state, 'consolidated', result.error)
                finally:
                    iterator.close()
                    for pending in list(engine.pending):
                        engine.acknowledge(pending)
        self.assertEqual(service.engine.close_count, 1)
        self.assertEqual(service.request_admission.ledger.snapshot()['active_reservations'], 0)

    def test_journal_failure_before_retirement_keeps_all_lanes_loaded(self):
        original = self.coordinator._journal
        def fail_routes(phase, error=None):
            if phase == 'routes-durable':
                raise OSError('injected journal durability failure')
            return original(phase, error)
        with patch.object(self.coordinator, '_journal', fail_routes):
            result = self.coordinator.consolidate(self.destination)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('journal durability failure', result.error)
        for service in self.services.values():
            self.assertTrue(service.engine.alive())
            self.assertEqual(service.engine.close_count, 0)
            self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 10)
        self.assertEqual(sum(command == 'EXPORT' for _, command in self.events), 6)
        self.assertFalse(any(command == 'CLOSE' for _, command in self.events))

    def test_identity_mismatch_before_migration_does_not_clear_native_state(self):
        before = {name: deepcopy(service.engine.states) for name, service in self.services.items()}
        self.services['lane-0'].engine.model_identity = ModelIdentity('qwen4exp', 'wrong-weights', 'tokenizer')
        result = self.coordinator.consolidate(self.destination)
        self.assertEqual(result.state, 'blocked')
        for name, service in self.services.items():
            self.assertEqual(service.engine.states, before[name])
            self.assertEqual(service.engine.close_count, 0)
        self.assertFalse(any(command in ('RELEASE_SLOT', 'RESTORE_SLOT', 'CLOSE') for _, command in self.events))

    def test_restarted_destination_is_not_reused_under_old_agent_routes(self):
        self.consolidate()
        agent = next(a for a, r in self.coordinator.routes.items() if r.lane is not None)
        engine = self.services[self.destination].engine
        engine.proc = type(engine.proc)()
        engine.gen += 1
        engine.states = {}
        engine.slot_held = [[], []]
        start = len(self.events)
        result = self.coordinator.activate(agent)
        self.assertEqual(result.state, 'blocked')
        self.assertFalse(any(command in ('RELEASE_SLOT', 'RESTORE_SLOT')
                             for _, command in self.events[start:]))
        self.assertTrue(self.services[self.destination].native_lane_gated)
        self.assertEqual(self.services[self.destination].request_admission.ledger.snapshot()['host_bytes'], 10)

    def test_duplicate_admission_worker_rejected_before_native_work(self):
        first, second = self.workers['lane-0'], self.workers['lane-1']
        second.admission = first.admission
        second.service.request_admission = first.admission
        with self.assertRaisesRegex(ValueError, 'distinct registered admission'):
            MigrationCoordinator(self.workers, self.store,
                                 {'a': ('lane-0', 0), 'b': ('lane-1', 0)}, enabled=True)
        self.assertEqual(self.events, [])

    def test_invalid_owner_requests_do_not_poison_healthy_lanes(self):
        for operation, args in ((self.coordinator.activate, ('missing',)),
                                (self.coordinator.activate, ([],)),
                                (self.coordinator.consolidate, ('missing',)),
                                (self.coordinator.consolidate, ([],))):
            with self.subTest(operation=operation.__name__, args=args):
                with self.assertRaises(ValueError):
                    operation(*args)
                self.assertFalse(self.coordinator._blocked)
        for operation, name in ((self.coordinator.activate, 'agent-0'),
                                (self.coordinator.consolidate, self.destination)):
            for timeout in (True, -1, float('nan'), 61):
                with self.subTest(operation=operation.__name__, timeout=timeout):
                    with self.assertRaises(ValueError):
                        operation(name, timeout_s=timeout)
                    self.assertFalse(self.coordinator._blocked)
        for agent in ('missing', []):
            with self.assertRaises(ValueError):
                with self.coordinator.route(agent):
                    self.fail('invalid agent was routed')
        self.assertFalse(self.coordinator._blocked)
        self.assertIsNone(self.coordinator._destination)
        self.assertEqual(self.events, [])
        for service in self.services.values():
            self.assertFalse(getattr(service, 'native_lane_gated', False))
        self.consolidate()

    def test_route_lock_timeout_is_retryable_and_preserves_owner_lock(self):
        from serve.lane_migration import MigrationBusy
        def attempt():
            with self.coordinator.route('agent-0', timeout_s=0.01):
                self.fail('route acquired an operation owned by another thread')
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.coordinator._operation.acquire()
            try:
                future = pool.submit(attempt)
                with self.assertRaises(MigrationBusy):
                    future.result(timeout=0.3)
                self.assertTrue(self.coordinator._operation.locked())
                self.assertEqual(self.events, [])
                self.assertFalse(self.coordinator._blocked)
                self.assertEqual(self.coordinator._active, set())
            finally:
                self.coordinator._operation.release()
        with self.coordinator.route('agent-0', timeout_s=0):
            self.assertIn('agent-0', self.coordinator._active)
        self.assertEqual(self.coordinator._active, set())

    def test_active_turn_timeout_is_retryable_without_gating_or_native_mutation(self):
        with self.coordinator.route('agent-0'):
            result = self.coordinator.consolidate(self.destination, timeout_s=0)
            self.assertEqual(result.state, 'waiting')
            self.assertEqual(self.events, [])
            self.assertFalse(self.coordinator._blocked)
            for service in self.services.values():
                self.assertFalse(getattr(service, 'native_lane_gated', False))
                lease = service.request_admission.acquire(1, 1)
                service.request_admission.ledger.release(lease, iterator_closed=True, work_drained=True)
        self.consolidate()

    def test_parked_agent_wait_timeout_does_not_stop_resident_work(self):
        self.consolidate()
        parked = next(a for a, r in self.coordinator.routes.items() if r.lane is None)
        resident = next(a for a, r in self.coordinator.routes.items() if r.lane is not None)
        start = len(self.events)
        with self.coordinator.route(resident):
            result = self.coordinator.activate(parked, timeout_s=0)
            self.assertEqual(result.state, 'waiting')
            self.assertEqual(len(self.events), start)
            self.assertFalse(self.services[self.destination].native_lane_gated)
        self.assertEqual(self.coordinator.activate(parked).state, 'ready')

    def test_corrupt_parked_payload_is_rejected_before_any_release(self):
        self.consolidate()
        routes = self.coordinator.routes
        agent, route = next((a, r) for a, r in routes.items() if r.lane is None)
        manifest = json.loads((self.root / route.checkpoint_id / 'manifest.json').read_text())
        entry = next(e for e in manifest['sessions'] if e['session']['session_id'] == route.session_id)
        (self.root / route.checkpoint_id / entry['file']).write_bytes(b'corrupted checkpoint')
        engine = self.services[self.destination].engine
        before, start = deepcopy(engine.states), len(self.events)
        result = self.coordinator.activate(agent)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(engine.states, before)
        self.assertFalse(any(command in ('RELEASE_SLOT', 'RESTORE_SLOT')
                             for _, command in self.events[start:]))


if __name__ == '__main__':
    unittest.main()
