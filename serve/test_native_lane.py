"""Native lane ownership integration with a fake protocol engine; CPU/files only."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

from serve.backends import ModelIdentity, SessionMetadata
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
        self.gen = 1
        self.precision_policy_report = {
            'enforced': True, 'draft': 'disabled',
            'artifact_audit': {'policy_passed': True,
                'policy': {'minimum_bits': 4, 'preferred_bits': 8},
                'components': {'target': {'architecture': 'qwen4exp', 'files': [{'architecture': 'qwen4exp'}]}}},
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
        return {'tokens': next(e['tokens'] for e in self.entries if e['id'] == sid),
                'bytes': Path(path).stat().st_size, 'ms': 0.0}

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


class BatchEngine(Engine):
    batch = 2
    session_batch_export = True
    session_batch_restore = True
    session_batch_release = True
    slot_groups = 1

    def __init__(self, model, fifo):
        super().__init__(model, fifo)
        self.ctl = threading.Lock()
        self.slot_cv = threading.Condition()
        self.waiting = self.admitting = 0
        self.wait_lens = []
        self.slot_claims = set()
        self.slot_releasing = set()
        self.slot_busy = [False, False]
        self.slot_live = [None, None]
        self.slot_held = [[1, 2], [3, 4]]
        self.entries.extend([{'id': 'slot-0', 'kind': 'slot', 'tokens': 10},
                             {'id': 'slot-1', 'kind': 'slot', 'tokens': 20}])

    def session_inventory(self):
        assert self.ctl.locked()
        return super().session_inventory()

    def session_export(self, generation, sid, path):
        assert self.ctl.locked()
        return super().session_export(generation, sid, path)

    def session_restore_slot(self, slot, path):
        assert self.fifo.locked() and self.ctl.locked()
        ids = [int(i) for i in Path(path).read_text().split(',')]
        self.restores.append((slot, ids))
        self.generation = f'process1-restored{len(self.restores)}'
        self.entries.append({'id': f'slot-{slot}', 'kind': 'slot', 'tokens': len(ids)})
        return {'tokens': len(ids), 'bytes': Path(path).stat().st_size, 'ms': 1.0, 'ids': ids}

    def session_release_slot(self, slot):
        assert self.fifo.locked() and self.ctl.locked()
        self.entries = [e for e in self.entries if e['id'] != f'slot-{slot}']
        self.generation += '-release'
        return {'slot': slot}

    def close(self):
        assert self.ctl.locked()
        return super().close()


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

    def shard_audit(self, architectures):
        from tools.model_manifest import build_manifest
        from tools.test_model_manifest import fixture
        paths = []
        for index, architecture in enumerate(architectures):
            path = Path(self.tmp.name) / f'model-{index + 1:05d}-of-00004.gguf'
            metadata = [('general.architecture', 8, architecture)] if architecture is not None else []
            paths.append(fixture(path, [(f'weight.{index}', [32], 8, 0, 34)], metadata))
        return build_manifest({'target': paths}, content_hash=True)

    def test_real_split_audit_with_metadata_only_in_first_shard(self):
        audit = self.shard_audit(['qwen4exp', None, None, None])
        self.assertTrue(audit['policy_passed'])
        target = audit['components']['target']
        self.assertEqual(target['architecture'], 'qwen4exp')
        self.assertEqual([f['architecture'] for f in target['files']], ['qwen4exp', None, None, None])
        self.engine.precision_policy_report['artifact_audit'] = audit
        NativeLaneWorker(self.service, self.model, enabled=True)
        self.assertEqual(self.engine.inventory_calls, 0)
        self.assertEqual(self.engine.close_count, 0)

    def test_real_split_audit_rejects_conflicts_absent_metadata_and_missing_shards(self):
        for architectures in (['qwen4exp', 'other', None, None], [None] * 4,
                              ['qwen4exp', None, None]):
            with self.subTest(architectures=architectures):
                self.engine.precision_policy_report['artifact_audit'] = self.shard_audit(architectures)
                with self.assertRaises(ValueError):
                    NativeLaneWorker(self.service, self.model, enabled=True)
        self.assertEqual(self.engine.inventory_calls, 0)
        self.assertEqual(self.engine.close_count, 0)

    def test_shard_architecture_report_requires_consistent_aggregate_and_explicit_value(self):
        audit = self.shard_audit(['qwen4exp', None, None, None])
        for change in ({'architecture': None}, {'architecture': 'other'}, {'files': []},
                       {'files': [None]}, {'files': [{'architecture': None}] * 4},
                       {'files': [{'architecture': 'qwen4exp'}, {'architecture': 'other'}]}):
            with self.subTest(change=change):
                bad = deepcopy(audit)
                bad['components']['target'].update(change)
                self.engine.precision_policy_report['artifact_audit'] = bad
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    NativeLaneWorker(self.service, self.model, enabled=True)

    def test_export_acknowledgement_must_match_inventory_and_written_file(self):
        for change in ({'tokens': 999}, {'bytes': 999}, {'tokens': True}, {'ms': float('nan')}):
            with self.subTest(change=change):
                # A fresh lane per scenario: failed checkpointing remains gated.
                engine = Engine(self.model, self.service.fifo)
                service = SimpleNamespace(engine=engine, fifo=self.service.fifo, request_admission=admission())
                worker = NativeLaneWorker(service, self.model, enabled=True)
                export = engine.session_export
                def bad_export(*args):
                    return {**export(*args), **change}
                engine.session_export = bad_export
                result = worker.persist_and_stop(self.store)
                self.assertEqual(result.state, 'blocked')
                self.assertIn('export acknowledgement', result.error)
                self.assertEqual(engine.close_count, 0)
                self.assertTrue(service.native_lane_gated)
                self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 10)
        self.assertFalse(any(p.name[0] != '.' for p in self.store.root.iterdir()))

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


class NativeBatchLaneTests(unittest.TestCase):
    setUp = NativeLaneTests.setUp
    def batch_lane(self):
        service = SimpleNamespace(fifo=threading.Lock(), request_admission=admission())
        service.engine = BatchEngine(self.model, service.fifo)
        return service, NativeLaneWorker(service, self.model, enabled=True)

    def test_batch_exports_all_idle_cached_slots_under_both_locks(self):
        service, worker = self.batch_lane()
        result = worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'stopped', result.error)
        entries = self.store.sessions(result.checkpoint_id, self.model)
        self.assertEqual({s.session_id for s in entries},
            {'process1-epoch2:live', 'process1-epoch2:parked-1',
             'process1-epoch2:slot-0', 'process1-epoch2:slot-1'})
        self.assertEqual(service.engine.close_count, 1)
        self.assertFalse(service.fifo.locked())
        self.assertFalse(service.engine.ctl.locked())
        self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 0)

    def test_batch_inventory_cannot_omit_known_cached_slot(self):
        service, worker = self.batch_lane()
        service.engine.entries = [e for e in service.engine.entries if e['id'] != 'slot-1']
        result = worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('omitted a known cached slot', result.error)
        self.assertEqual(service.engine.close_count, 0)
        self.assertFalse(service.fifo.locked())
        self.assertFalse(service.engine.ctl.locked())

    def test_batch_unresolved_ownership_never_inventories_or_stops(self):
        cases = {'waiting': 1, 'admitting': 1, 'wait_lens': [[5, 0, True]],
                 'slot_claims': {0}, 'slot_releasing': {1},
                 'slot_busy': [False, True], 'slot_live': [None, {}]}
        for name, value in cases.items():
            with self.subTest(name=name):
                service, worker = self.batch_lane()
                setattr(service.engine, name, value)
                result = worker.persist_and_stop(self.store, timeout_s=0)
                self.assertEqual(result.state, 'blocked')
                self.assertEqual(service.engine.inventory_calls, 0)
                self.assertEqual(service.engine.close_count, 0)
                self.assertFalse(service.fifo.locked())
                self.assertFalse(service.engine.ctl.locked())
                self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 10)

    def test_batch_waits_for_receipt_cleanup_bookkeeping(self):
        service, worker = self.batch_lane()
        waiting = threading.Event()
        class ObservedCondition(threading.Condition):
            def wait(self, timeout=None):
                waiting.set()
                return super().wait(timeout)
        service.engine.slot_cv = ObservedCondition()
        service.engine.slot_busy[0] = True
        service.engine.slot_releasing.add(0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(worker.persist_and_stop, self.store, 2)
            self.assertTrue(waiting.wait(1))
            self.assertTrue(service.fifo.locked())
            self.assertTrue(service.engine.ctl.locked())
            with service.engine.slot_cv:
                service.engine.slot_busy[0] = False
                service.engine.slot_releasing.clear()
                service.engine.slot_cv.notify_all()
            stopped = result.result(timeout=2)
        self.assertEqual(stopped.state, 'stopped', stopped.error)
        self.assertFalse(service.fifo.locked())
        self.assertFalse(service.engine.ctl.locked())

    def test_batch_foreign_control_lock_times_out_and_releases_only_fifo(self):
        service, worker = self.batch_lane()
        service.engine.ctl.acquire()
        result = worker.persist_and_stop(self.store, timeout_s=0)
        self.assertEqual(result.state, 'blocked')
        self.assertFalse(service.fifo.locked())
        self.assertTrue(service.engine.ctl.locked())
        self.assertEqual(service.engine.inventory_calls, 0)
        service.engine.ctl.release()

    def test_batch_export_failure_keeps_process_and_releases_locks(self):
        service, worker = self.batch_lane()
        service.engine.fail_export = True
        result = worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'blocked')
        self.assertTrue(service.engine.alive())
        self.assertEqual(service.engine.close_count, 0)
        self.assertFalse(service.fifo.locked())
        self.assertFalse(service.engine.ctl.locked())

    def test_batch_generation_or_process_swap_during_export_blocks_stop(self):
        for change in ('generation', 'process'):
            with self.subTest(change=change):
                service, worker = self.batch_lane()
                original = service.engine.session_export
                def changed(*args):
                    original(*args)
                    if change == 'generation':
                        service.engine.gen += 1
                    else:
                        service.engine.proc = Process()
                service.engine.session_export = changed
                result = worker.persist_and_stop(self.store)
                self.assertEqual(result.state, 'blocked')
                self.assertIn('process or generation changed', result.error)
                self.assertEqual(service.engine.close_count, 0)
                self.assertFalse(service.fifo.locked())
                self.assertFalse(service.engine.ctl.locked())

    def test_batch_close_cannot_retire_fixed_costs_of_restarted_engine(self):
        service, worker = self.batch_lane()
        original = service.engine.close
        def restarted():
            original()
            service.engine.proc = Process()
            service.engine.gen += 1
        service.engine.close = restarted
        result = worker.persist_and_stop(self.store)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('restarted during lane shutdown', result.error)
        self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 10)
        self.assertFalse(service.fifo.locked())
        self.assertFalse(service.engine.ctl.locked())

    def test_batch_destination_restore_is_explicitly_rejected(self):
        service, worker = self.batch_lane()
        with self.assertRaisesRegex(ValueError, 'unbatched destination'):
            worker.lazy_restore(self.store, 'unused', 'unused')
        self.assertEqual(service.engine.restores, [])
        self.assertFalse(service.fifo.locked())
        self.assertFalse(service.engine.ctl.locked())

    def test_batch_capability_or_group_mismatch_rejects_before_control(self):
        for key, value in [('session_batch_export', False), ('slot_groups', 2)]:
            service, _ = self.batch_lane()
            setattr(service.engine, key, value)
            with self.assertRaises(ValueError):
                NativeLaneWorker(service, self.model, enabled=True)
            self.assertEqual(service.engine.inventory_calls, 0)


class NativeSlotRestoreTests(unittest.TestCase):
    batch_lane = NativeBatchLaneTests.batch_lane
    def setUp(self):
        NativeLaneTests.setUp(self)
        self.ready, self.restore_worker = self.batch_lane()
        self.ready.engine.entries = [e for e in self.ready.engine.entries if e['kind'] != 'slot']
        self.ready.engine.slot_held = [[], []]
        sessions = [SessionMetadata('source-a', self.model), SessionMetadata('source-b', self.model)]
        self.bundle = self.store.publish(self.model, sessions,
            lambda session, path: path.write_text('1,2,3' if session.session_id == 'source-a' else '4,5'))

    def restore(self, session='source-a', slot=0):
        return self.restore_worker.restore_slot(self.store, self.bundle, session, slot)

    def test_distinct_slot_restore_and_explicit_resume_retains_fixed_costs(self):
        other = deepcopy(self.ready.engine.entries)
        self.assertEqual(self.restore().state, 'restored')
        self.assertEqual(self.restore('source-b', 1).state, 'restored')
        self.assertEqual([e for e in self.ready.engine.entries if e['kind'] != 'slot'], other)
        self.assertEqual(self.ready.engine.slot_held, [[1, 2, 3], [4, 5]])
        with self.assertRaises(AdmissionRejected):
            self.ready.request_admission.acquire(1, 1)
        resumed = self.restore_worker.resume()
        self.assertEqual(resumed.state, 'resumed', resumed.error)
        self.assertFalse(self.ready.native_lane_gated)
        self.assertEqual(self.ready.request_admission.ledger.snapshot()['host_bytes'], 10)
        lease = self.ready.request_admission.acquire(1, 1)
        self.ready.request_admission.ledger.release(lease, iterator_closed=True, work_drained=True)
        self.assertFalse(self.ready.fifo.locked())
        self.assertFalse(self.ready.engine.ctl.locked())

    def test_failed_restore_that_changes_other_state_cannot_resume(self):
        original = self.ready.engine.session_restore_slot
        def corrupt(*args):
            reply = original(*args)
            self.ready.engine.entries[0]['tokens'] += 1
            return reply
        self.ready.engine.session_restore_slot = corrupt
        result = self.restore()
        self.assertEqual(result.state, 'blocked')
        self.assertIn('changed other state', result.error)
        self.assertEqual(self.ready.engine.slot_held, [[], []])
        self.assertEqual(self.restore_worker.resume().state, 'blocked')
        with self.assertRaises(AdmissionRejected):
            self.ready.request_admission.acquire(1, 1)
        self.assertEqual(self.ready.engine.close_count, 0)
        self.assertFalse(self.ready.fifo.locked())
        self.assertFalse(self.ready.engine.ctl.locked())

    def test_bad_acknowledgement_poison_prevents_retry_and_resume(self):
        original = self.ready.engine.session_restore_slot
        def bad_count(*args):
            reply = original(*args)
            reply['ids'] = [1]
            return reply
        self.ready.engine.session_restore_slot = bad_count
        self.assertEqual(self.restore().state, 'blocked')
        self.assertEqual(self.restore('source-b', 1).state, 'blocked')
        self.assertEqual(self.restore_worker.resume().state, 'blocked')
        self.assertEqual(len(self.ready.engine.restores), 1)
        self.assertEqual(self.ready.engine.slot_held, [[], []])

    def test_process_replacement_between_restores_blocks_without_dispatch(self):
        self.assertEqual(self.restore().state, 'restored')
        self.ready.engine.proc = Process()
        self.ready.engine.gen += 1
        result = self.restore('source-b', 1)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('between gated', result.error)
        self.assertEqual(len(self.ready.engine.restores), 1)
        self.assertEqual(self.restore_worker.resume().state, 'blocked')

    def test_resume_without_verified_restore_never_reopens_admission(self):
        result = self.restore_worker.resume()
        self.assertEqual(result.state, 'blocked')
        self.assertTrue(self.ready.native_lane_gated)
        with self.assertRaises(AdmissionRejected):
            self.ready.request_admission.acquire(1, 1)

    def test_invalid_slot_precheck_also_gates_and_blocks_resume(self):
        result = self.restore(slot=-1)
        self.assertEqual(result.state, 'blocked')
        self.assertTrue(self.ready.native_lane_gated)
        self.assertEqual(self.ready.engine.restores, [])
        with self.assertRaises(AdmissionRejected):
            self.ready.request_admission.acquire(1, 1)
        self.assertEqual(self.restore_worker.resume().state, 'blocked')

    def test_inventory_or_routing_changes_after_restore_block_resume(self):
        self.assertEqual(self.restore().state, 'restored')
        self.ready.engine.slot_held[0] = [8, 9, 10]
        result = self.restore_worker.resume()
        self.assertEqual(result.state, 'blocked')
        self.assertIn('routing state changed', result.error)
        self.assertTrue(self.ready.native_lane_gated)


class NativeParkedLaneTests(unittest.TestCase):
    setUp = NativeLaneTests.setUp
    batch_lane = NativeBatchLaneTests.batch_lane

    def test_snapshot_stays_loaded_and_resume_needs_no_restore(self):
        service, worker = self.batch_lane()
        parked = worker.quiesce_snapshot(self.store)
        self.assertEqual(parked.state, 'parked', parked.error)
        self.assertEqual(len(self.store.sessions(parked.checkpoint_id, self.model)), 4)
        self.assertEqual(service.engine.close_count, 0)
        self.assertTrue(service.engine.alive())
        self.assertTrue(service.native_lane_gated)
        self.assertEqual(worker.resume().state, 'resumed')
        lease = service.request_admission.acquire(1, 1)
        service.request_admission.ledger.release(lease, iterator_closed=True, work_drained=True)
        self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 10)

    def test_empty_unbatched_snapshot_can_resume(self):
        self.engine.entries = []
        parked = self.worker.quiesce_snapshot(self.store)
        self.assertEqual(parked.state, 'parked', parked.error)
        self.assertEqual(self.worker.resume().state, 'resumed')
        self.assertTrue(self.engine.alive())
        self.assertFalse(self.service.native_lane_gated)

    def test_stop_parked_commits_only_matching_durable_snapshot(self):
        service, worker = self.batch_lane()
        parked = worker.quiesce_snapshot(self.store)
        result = worker.stop_parked(self.store, parked.checkpoint_id)
        self.assertEqual(result.state, 'stopped', result.error)
        self.assertEqual(service.engine.close_count, 1)
        self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 0)
        self.assertEqual(worker.stop_parked(self.store, parked.checkpoint_id).state, 'stopped')
        self.assertEqual(service.engine.close_count, 1)
        self.assertFalse(service.fifo.locked())
        self.assertFalse(service.engine.ctl.locked())

    def test_repeated_parked_stop_does_not_hide_unexpected_restart(self):
        service, worker = self.batch_lane()
        parked = worker.quiesce_snapshot(self.store)
        self.assertEqual(worker.stop_parked(self.store, parked.checkpoint_id).state, 'stopped')
        service.engine.proc = Process()
        service.engine.gen += 1
        result = worker.stop_parked(self.store, parked.checkpoint_id)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('restarted after parked stop', result.error)
        self.assertEqual(service.engine.close_count, 1)

    def test_corrupt_parked_bundle_prevents_stop(self):
        service, worker = self.batch_lane()
        parked = worker.quiesce_snapshot(self.store)
        (self.store.root / parked.checkpoint_id / 'session-0000.bin').write_bytes(b'corrupted')
        result = worker.stop_parked(self.store, parked.checkpoint_id)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('checksum', result.error)
        self.assertEqual(service.engine.close_count, 0)
        self.assertTrue(service.engine.alive())
        self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 10)

    def test_changed_native_inventory_prevents_parked_stop(self):
        service, worker = self.batch_lane()
        parked = worker.quiesce_snapshot(self.store)
        service.engine.generation += '-changed'
        result = worker.stop_parked(self.store, parked.checkpoint_id)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(service.engine.close_count, 0)
        self.assertEqual(worker.resume().state, 'blocked')

    def test_consecutive_durable_releases_preserve_neighbors_then_resume(self):
        service, worker = self.batch_lane()
        parked = worker.quiesce_snapshot(self.store)
        other = [e for e in deepcopy(service.engine.entries) if e['kind'] != 'slot']
        self.assertEqual(worker.release_slot(0).state, 'released')
        self.assertEqual(worker.release_slot(1).state, 'released')
        self.assertEqual(service.engine.entries, other)
        self.assertEqual(service.engine.slot_held, [[], []])
        self.assertEqual(len(self.store.sessions(parked.checkpoint_id, self.model)), 4)
        self.assertEqual(worker.resume().state, 'resumed')
        self.assertEqual(service.engine.close_count, 0)
        self.assertEqual(service.request_admission.ledger.snapshot()['host_bytes'], 10)

    def test_release_without_full_snapshot_is_blocked_before_native_mutation(self):
        service, worker = self.batch_lane()
        before = deepcopy(service.engine.entries)
        result = worker.release_slot(0)
        self.assertEqual(result.state, 'blocked')
        self.assertEqual(service.engine.entries, before)
        self.assertTrue(service.native_lane_gated)
        self.assertEqual(worker.resume().state, 'blocked')

    def test_release_then_restore_requires_new_snapshot_before_next_eviction(self):
        service, worker = self.batch_lane()
        worker.quiesce_snapshot(self.store)
        self.assertEqual(worker.release_slot(0).state, 'released')
        restored_bundle = self.store.publish(self.model, [SessionMetadata('new', self.model)],
                                             lambda _, p: p.write_text('8,9,10'))
        result = worker.restore_slot(self.store, restored_bundle, 'new', 0)
        self.assertEqual(result.state, 'restored', result.error)
        before = deepcopy(service.engine.entries)
        blocked = worker.release_slot(0)
        self.assertEqual(blocked.state, 'blocked')
        self.assertEqual(service.engine.entries, before)

    def test_new_snapshot_after_restore_allows_safe_rotation(self):
        service, worker = self.batch_lane()
        worker.quiesce_snapshot(self.store)
        self.assertEqual(worker.release_slot(0).state, 'released')
        restored_bundle = self.store.publish(self.model, [SessionMetadata('new', self.model)],
                                             lambda _, p: p.write_text('8,9,10'))
        self.assertEqual(worker.restore_slot(self.store, restored_bundle, 'new', 0).state, 'restored')
        # Fake export must serialize the restored IDs just as native export would.
        original = service.engine.session_export
        def export(generation, sid, path):
            reply = original(generation, sid, path)
            if sid == 'slot-0':
                Path(path).write_text('8,9,10')
            return {**reply, 'bytes': Path(path).stat().st_size}
        service.engine.session_export = export
        parked = worker.quiesce_snapshot(self.store)
        self.assertEqual(parked.state, 'parked', parked.error)
        self.assertEqual(worker.release_slot(0).state, 'released')
        self.assertEqual(worker.resume().state, 'resumed')

    def test_release_acknowledgement_cannot_hide_neighbor_mutation(self):
        service, worker = self.batch_lane()
        worker.quiesce_snapshot(self.store)
        original = service.engine.session_release_slot
        def corrupt(slot):
            reply = original(slot)
            service.engine.entries[0]['tokens'] += 1
            return reply
        service.engine.session_release_slot = corrupt
        result = worker.release_slot(0)
        self.assertEqual(result.state, 'blocked')
        self.assertIn('changed other state', result.error)
        self.assertEqual(service.engine.slot_held[0], [1, 2])
        self.assertEqual(worker.resume().state, 'blocked')
        self.assertEqual(service.engine.close_count, 0)


if __name__ == '__main__':
    unittest.main()
