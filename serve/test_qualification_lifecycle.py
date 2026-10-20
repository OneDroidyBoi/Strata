"""Runner contract integration with real lifecycle, admission, and wire parser.

Only model execution/native wire storage are simulated. No GPU or server runs.
QUALIFICATION_CLIENT_PATH can select the runner's separately frozen candidate.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve.backends import ModelIdentity
from serve.checkpoint_store import CheckpointStore
from serve.lane_migration import MigrationCoordinator
from serve.native_lane import NativeLaneWorker
from serve.resource_admission import AdmissionRejected
from serve.test_lane_migration import MigrationWireEngine
from serve.test_native_lane import admission


class QualificationWireEngine(MigrationWireEngine):
    def __init__(self, *args):
        super().__init__(*args)
        self.proc.pid = 4242
        self.info = {'slot_cache': 1, 'session_export_batch': 1, 'session_restore_batch': 1}
        self.last = {}

    def generate(self, ids, max_new, sampling, cancel):
        slot = self.local.slot
        held = self.slot_held[slot]
        reused = len(held) if held and ids[:len(held)] == held else 0
        output = [901 + i for i in range(max_new)]
        # Preserve the engine contract: the final sample has not been consumed.
        self.states[slot] = list(ids) + output[:-1]
        self.slot_held[slot] = list(self.states[slot])
        self.epoch += 1
        self.last = {'reused': reused, 'prompt_read': len(ids) - reused,
                     'finish': 'length', 'generated': max_new}
        iterator = super().generate(ids, max_new, sampling, cancel)
        try:
            next(iterator)  # Real DrainReceipt registration on the fake transport.
            yield from output
        finally:
            iterator.close()


class QualificationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = ModelIdentity('qwen4exp', 'weights', 'tokenizer')
        self.events = []

    def harness(self, count=1):
        root = self.root / str(len(list(self.root.iterdir())))
        root.mkdir()
        store = CheckpointStore(root)
        workers, services, agents = {}, {}, {}
        for i in range(count):
            name = f'lane-{i}'
            service = SimpleNamespace(fifo=threading.Lock(), request_admission=admission())
            service.engine = QualificationWireEngine(self.model, service.fifo, {}, name, self.events)
            service.engine.proc.pid += i
            workers[name] = NativeLaneWorker(service, self.model, enabled=True)
            services[name] = service
            for slot in range(2):
                agents[f'agent-{i * 2 + slot}'] = (name, slot)
        coordinator = MigrationCoordinator(workers, store, agents, enabled=True)
        frontend = SimpleNamespace(encode_prompt=lambda messages, tools, kwargs, store=False:
                                   [int(x) for x in messages[0]['content'].split(',')])
        return SimpleNamespace(stage='single_worker' if count == 1 else 'three_worker',
                               workers=workers, services=services, coordinator=coordinator,
                               store=store, frontend=frontend)

    def client(self):
        path = os.environ.get('QUALIFICATION_CLIENT_PATH') or str(
            Path(__file__).resolve().parents[1] / 'tools/qualification_window28/qualification_client.py')
        spec = importlib.util.spec_from_file_location('lifecycle_qualification_client', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def stop(self, harness):
        for worker in harness.workers.values():
            if worker.engine.alive():
                result = worker.persist_and_stop(harness.store)
                self.assertEqual(result.state, 'stopped', result.error)

    def test_inventory_owns_native_protocol_and_requires_explicit_resume(self):
        h = self.harness()
        worker = h.workers['lane-0']
        with self.assertRaisesRegex(RuntimeError, 'public lifecycle'):
            worker._snapshot()
        result = worker.quiesce_inventory()
        self.assertEqual(result.state, 'inventoried', result.error)
        self.assertEqual(result.sessions, ())
        self.assertEqual(result.held, ((0, ()), (1, ())))
        self.assertTrue(worker.service.native_lane_gated)
        self.assertFalse(worker.service.fifo.locked())
        self.assertFalse(worker.engine.ctl.locked())
        with self.assertRaises(AdmissionRejected):
            worker.admission.acquire(1, 1)
        self.assertEqual(worker.resume().state, 'resumed')
        self.assertFalse(worker.service.native_lane_gated)
        self.stop(h)

    def test_inventory_is_not_permission_to_release_undurable_state(self):
        h = self.harness()
        worker = h.workers['lane-0']
        list(h.coordinator.generate('agent-0', [1, 2, 3], 1, {}, threading.Event()))
        result = worker.quiesce_inventory()
        self.assertEqual(result.held[0], (0, (1, 2, 3)))
        self.assertEqual(result.sessions, (('slot-0', 'slot', 3),))
        result = worker.release_slot(0)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('durable snapshot', result.error)
        self.assertEqual(worker.engine.states[0], [1, 2, 3])
        self.assertFalse(any(cmd == 'RELEASE_SLOT' for _, cmd in self.events))

    def test_inventory_rejects_existing_gate_without_consuming_resume_proof(self):
        h = self.harness()
        worker = h.workers['lane-0']
        self.assertEqual(worker.quiesce_snapshot(h.store).state, 'parked')
        commands = list(worker.engine.commands)
        self.assertEqual(worker.quiesce_inventory().state, 'blocked')
        self.assertEqual(worker.engine.commands, commands)
        self.assertEqual(worker.resume().state, 'resumed')
        self.stop(h)

    def test_inventory_protocol_failure_stays_gated_and_does_not_resume(self):
        h = self.harness()
        worker = h.workers['lane-0']
        with patch.object(worker.engine, 'session_inventory', side_effect=RuntimeError('bad wire')):
            result = worker.quiesce_inventory()
        self.assertEqual(result.state, 'blocked')
        self.assertIn('bad wire', result.error)
        self.assertFalse(worker.service.fifo.locked())
        self.assertFalse(worker.engine.ctl.locked())
        self.assertTrue(worker.service.native_lane_gated)
        self.assertEqual(worker.resume().state, 'blocked')

    def test_invalid_inventory_timeout_does_not_gate_serving_lane(self):
        h = self.harness()
        worker = h.workers['lane-0']
        for timeout in (True, -1, 61, float('nan')):
            with self.assertRaises(ValueError):
                worker.quiesce_inventory(timeout)
        self.assertFalse(getattr(worker.service, 'native_lane_gated', False))
        self.assertEqual(worker.engine.commands, [])
        self.stop(h)

    def test_changed_inventory_or_process_cannot_resume(self):
        for change in ('inventory', 'process'):
            with self.subTest(change=change):
                h = self.harness()
                worker = h.workers['lane-0']
                self.assertEqual(worker.quiesce_inventory().state, 'inventoried')
                if change == 'inventory':
                    worker.engine.epoch += 1
                else:
                    worker.engine.gen += 1
                self.assertEqual(worker.resume().state, 'blocked')
                self.assertTrue(worker.service.native_lane_gated)
                with self.assertRaises(AdmissionRejected):
                    worker.admission.acquire(1, 1)

    def test_inventory_waits_for_exact_drain_and_never_reopens_on_failure(self):
        h = self.harness()
        worker = h.workers['lane-0']
        worker.engine.defer_ack = True
        list(h.coordinator.generate('agent-0', [1, 2], 1, {}, threading.Event()))
        self.assertEqual(worker.admission.ledger.snapshot()['active_reservations'], 1)
        start = len(self.events)
        result = worker.quiesce_inventory(timeout_s=0)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(len(self.events), start)
        self.assertEqual(worker.resume().state, 'blocked')
        for pending in list(worker.engine.pending):
            worker.engine.acknowledge(pending)
        self.assertTrue(worker.service.native_lane_gated)

    def test_real_lifecycle_seed_baseline_checkpoint_cleanup(self):
        qc = self.client()
        h = self.harness()
        closed = []
        def close(obj):
            self.stop(obj)
            closed.append(obj)
        rows = qc.fresh_isolated_native_baseline(lambda: h,
            fixtures={'a': ('agent-0', [{'role': 'user', 'content': '1,2,3,4,5'}], 2)},
            close_harness=close)
        self.assertEqual(rows['a']['seed']['held_after_token_ids'], [1, 2, 3])
        self.assertEqual(rows['a']['baseline']['reused'], 3)
        self.assertEqual(closed, [h])
        self.assertFalse(h.workers['lane-0'].engine.alive())
        self.assertTrue(any(cmd == 'EXPORT' for _, cmd in self.events))

    def test_client_refuses_failed_resume_before_seed_or_continuation(self):
        qc = self.client()
        for fail_on in (1, 2):
            with self.subTest(fail_on=fail_on):
                h = self.harness()
                worker = h.workers['lane-0']
                resume = worker.resume
                attempts, closed = [], []
                def changed_resume(*args, **kwargs):
                    attempts.append(True)
                    if len(attempts) == fail_on:
                        worker.engine.epoch += 1
                    return resume(*args, **kwargs)
                with patch.object(worker, 'resume', changed_resume):
                    with self.assertRaisesRegex(RuntimeError, 'resume'):
                        qc.fresh_isolated_native_baseline(lambda: h,
                            fixtures={'a': ('agent-0', [{'role': 'user', 'content': '1,2,3,4,5'}], 2)},
                            close_harness=closed.append)
                self.assertEqual(closed, [h])
                self.assertEqual(len(worker.engine.dispatched), fail_on - 1)
                self.assertTrue(worker.service.native_lane_gated)
                with self.assertRaises(AdmissionRejected):
                    worker.admission.acquire(1, 1)

    def test_six_seeded_agents_checkpoint_consolidate_rotate_and_continue(self):
        qc = self.client()
        h = self.harness(3)
        prompts = {f'agent-{i}': [100 * i + n for n in range(1, 6)] for i in range(6)}
        for agent, ids in prompts.items():
            row = qc.seed_owner_prefix_fixture(h, agent=agent, prompt_ids=ids, tail_tokens=2)
            self.assertEqual(row['status'], 'PASS')
        result = h.coordinator.consolidate('lane-2')
        self.assertEqual(result.state, 'consolidated', result.error)
        self.assertEqual(sum(r.lane is None for r in h.coordinator.routes.values()), 4)
        for agent, ids in prompts.items():
            self.assertEqual(h.coordinator.activate(agent).state, 'ready')
            row = qc.collect_token_ids(h.coordinator, h.workers, agent=agent,
                                       prompt_ids=ids, max_new=4)
            self.assertEqual(row['output_token_ids'], [901, 902, 903, 904])
            self.assertEqual(row['reused'], 3)
            self.assertEqual(row['prompt_read'], 2)
        # Revisit advanced histories after eviction. Recovering the initial C
        # would lose completed work and would not satisfy this reuse assertion.
        for agent, ids in reversed(list(prompts.items())):
            self.assertEqual(h.coordinator.activate(agent).state, 'ready')
            advanced = ids + [901, 902, 903]
            row = qc.collect_token_ids(h.coordinator, h.workers, agent=agent,
                                       prompt_ids=advanced + [999], max_new=4)
            self.assertEqual(row['reused'], len(advanced))
            self.assertEqual(row['prompt_read'], 1)
        self.stop(h)
        self.assertTrue(all(not w.engine.alive() for w in h.workers.values()))
        self.assertTrue(any(cmd == 'RELEASE_SLOT' for _, cmd in self.events))
        self.assertTrue(any(cmd == 'RESTORE_SLOT' for _, cmd in self.events))


if __name__ == '__main__':
    unittest.main()
