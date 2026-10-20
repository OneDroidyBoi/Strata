"""Opt-in native lane checkpoint adapter; no background handlers or automatic unload.

Only audited Flash-Next lanes with advertised native export support are admitted.
The public operations hold FIFO and, for batching, the engine control lock from
completed request drain through export/stop. Batch restore additionally requires
its own advertised capability; ordinary RESTORE remains unbatched.
Engine inventory/export methods are protocol parsers; they must not acquire the
service FIFO themselves. ModelIdentity remains an owner-supplied identity binding.

The owner must derive that identity from the immutable, complete weight/config,
PLE/draft inventory and tokenizer actually loaded, then bind engine.model_identity
to it after the strict launch audit and READY. A display model name is insufficient.
Do not copy an identity onto a replacement engine without validating its artifacts.
The service must reject control-plane reload/restore after native_lane_gated is
set (check under FIFO). Resuming requires a new registered admission worker and a
ready, identity-validated engine; clearing this flag alone is never a resume.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import re
from pathlib import Path
import threading
import time

from serve.backends import ModelIdentity, SessionMetadata
from serve.lane_lifecycle import LaneLifecycle, LifecycleResult
from serve.request_admission import RequestAdmission


class NativeLaneWorker:
    def __init__(self, service, model: ModelIdentity, *, enabled=False):
        if enabled is not True:
            raise ValueError('native lane lifecycle requires explicit opt-in')
        if not isinstance(model, ModelIdentity) or model.architecture != 'qwen4exp':
            raise ValueError('native lane lifecycle requires a qwen4exp model identity')
        self.service, self.model = service, model
        self.engine = service.engine
        self.admission = getattr(service, 'request_admission', None)
        if not isinstance(self.admission, RequestAdmission):
            raise ValueError('native lane lifecycle requires the service RequestAdmission controller')
        self._operation = threading.Lock()
        self._owner = None
        self._fifo_held = False
        self._ctl_held = None
        self._process_pin = None
        self._inventory = None
        self._inventory_detail = None
        self._lifecycle = None
        self._store = None
        self._gated_process = None
        self._restore_failed = False
        self._resume_inventory = None
        self._resume_held = {}
        self._parked_checkpoint = None
        self._parked_snapshot = None
        self._parked_sessions = ()
        self._durable_inventory = None
        self._parked_stopped = False
        self._validate_engine()

    def _validate_engine(self):
        if getattr(self.engine, '_session_protocol_failed', False):
            raise RuntimeError('native session protocol is blocked; restore cannot safely use its command stream')
        if self.service.engine is not self.engine or self.service.request_admission is not self.admission:
            raise RuntimeError('lane engine or admission controller changed')
        batch = getattr(self.engine, 'batch', 0)
        if type(batch) is not int or batch < 0:
            raise ValueError('invalid native batch size')
        if batch and (getattr(self.engine, 'session_batch_export', False) is not True
                      or getattr(self.engine, 'slot_groups', 1) != 1):
            raise ValueError('native batch session export is not advertised for this lane')
        if getattr(self.engine, 'starting', False) or getattr(self.engine, '_ctl_erred', False):
            raise RuntimeError('native lane control state is not ready')
        if getattr(self.engine, 'session_lifecycle', False) is not True:
            raise ValueError('native session lifecycle flag is not enabled')
        if getattr(self.engine, 'model_identity', None) != self.model:
            raise ValueError('native engine model identity does not match this lane')
        report = getattr(self.engine, 'precision_policy_report', None)
        if not isinstance(report, dict) or report.get('enforced') is not True or report.get('draft') != 'disabled':
            raise ValueError('native lane lifecycle requires an enforced precision policy with draft disabled')
        audit = report.get('artifact_audit', {})
        policy = audit.get('policy', {})
        floor, preferred = policy.get('minimum_bits'), policy.get('preferred_bits')
        if (audit.get('policy_passed') is not True or type(floor) is not int or floor < 4
                or type(preferred) is not int or preferred < 8 or preferred < floor
                or report.get('pack_audit', {}).get('packed_payload_verified_against_source') is not True):
            raise ValueError('native lane requires verified Q4 minimum / Q8 preferred artifacts and pack')
        # GGUF split writers may put model metadata only in the first shard.
        # The audit validates the complete shard set and aggregates explicit values.
        target = audit.get('components', {}).get('target', {})
        files = target.get('files') if isinstance(target, dict) else None
        if (not isinstance(target, dict) or target.get('architecture') != self.model.architecture
                or not isinstance(files, list) or not files
                or any(not isinstance(f, dict) or f.get('architecture') not in (None, self.model.architecture)
                       for f in files)
                or not any(f.get('architecture') == self.model.architecture for f in files)):
            raise ValueError('audited target architecture does not match lane identity')
        for method in ('session_inventory', 'session_export', 'session_file', 'close', 'alive'):
            if not callable(getattr(self.engine, method, None)):
                raise TypeError('native engine is missing ' + method)

    @contextmanager
    def _transaction(self):
        if not self._operation.acquire(blocking=False):
            raise RuntimeError('native lane lifecycle operation already running')
        self._owner = threading.get_ident()
        self._inventory = None
        self._inventory_detail = None
        self._process_pin = (getattr(self.engine, 'proc', None), getattr(self.engine, 'gen', None))
        try:
            yield
        finally:
            if self._ctl_held is not None:
                self._ctl_held.release()
                self._ctl_held = None
            if self._fifo_held:
                self.service.fifo.release()
                self._fifo_held = False
            self._owner = None
            self._process_pin = None
            self._operation.release()

    def _owned(self, *, drained=False):
        if self._owner != threading.get_ident():
            raise RuntimeError('use the native lane public lifecycle operations')
        if drained and not self._fifo_held:
            raise RuntimeError('native lane has not drained under the service FIFO')
        self._validate_engine()
        self._same_process()
        if drained and getattr(self.engine, 'batch', 0):
            if self._ctl_held is None or self._ctl_held is not self.engine.ctl:
                raise RuntimeError('native batch control lock is not owned')
            self._batch_idle()

    def _same_process(self):
        if self._process_pin is not None:
            proc, generation = self._process_pin
            if self.engine.proc is not proc or getattr(self.engine, 'gen', None) != generation:
                raise RuntimeError('native process or generation changed during lane transaction')

    def _batch_idle(self, deadline=None):
        # Receipt acknowledgement can release a lease immediately before its
        # background waiter clears busy/releasing under slot_cv. Give that last
        # bookkeeping the same bounded drain budget, without ever clearing flags
        # ourselves. ctl stays held; Condition.wait releases only slot_cv.
        cv = getattr(self.engine, 'slot_cv', None)
        remaining = max(0.0, deadline - time.monotonic()) if deadline is not None else 0.0
        if cv is None or not cv.acquire(timeout=remaining):
            raise RuntimeError('native batch ownership cannot be inspected')
        try:
            while True:
                problem = self._batch_problem()
                if problem is None:
                    return
                remaining = max(0.0, deadline - time.monotonic()) if deadline is not None else 0.0
                if remaining <= 0:
                    raise RuntimeError(problem)
                cv.wait(remaining)
                self._same_process()
        finally:
            cv.release()

    def _batch_problem(self):
        # Caller holds slot_cv. A stale/orphan busy flag is still outstanding
        # ownership for checkpoint purposes, never permission to discard state.
        e, n = self.engine, self.engine.batch
        for name in ('waiting', 'admitting'):
            if type(getattr(e, name, None)) is not int or getattr(e, name) != 0:
                return 'native batch ownership remains: ' + name
        for name in ('slot_claims', 'slot_releasing'):
            value = getattr(e, name, None)
            if not isinstance(value, set) or value:
                return 'native batch ownership remains: ' + name
        busy, live, waiting = (getattr(e, name, None) for name in ('slot_busy', 'slot_live', 'wait_lens'))
        if (not isinstance(busy, list) or len(busy) != n or any(v is not False for v in busy)
                or not isinstance(live, list) or len(live) != n or any(v is not None for v in live)
                or not isinstance(waiting, list) or waiting):
            return 'native batch slots or waiters remain owned'
        return None

    def _controller(self, store):
        if self._lifecycle is None:
            self._store = store
            self._lifecycle = LaneLifecycle(self, store, self.model)
        elif store is not self._store:
            raise ValueError('a lane lifecycle cannot switch checkpoint stores')
        return self._lifecycle

    def persist_and_stop(self, store, timeout_s=30.0):
        with self._transaction():
            result = self._controller(store).persist_and_stop(timeout_s)
            if result.state == 'blocked':
                self._restore_failed = True
            return result

    def lazy_restore(self, store, checkpoint_id, session_id, timeout_s=30.0):
        if getattr(self.engine, 'batch', 0):
            raise ValueError('native RESTORE requires an unbatched destination lane')
        with self._transaction():
            return self._controller(store).lazy_restore(checkpoint_id, session_id, timeout_s)

    def _failure(self, exc, checkpoint_id=None):
        self._restore_failed = True
        self.service.native_lane_gated = True
        try:
            self.admission.gate_admission()
        except Exception:
            pass
        return LifecycleResult('blocked', checkpoint_id, f'{type(exc).__name__}: {exc}')

    def _capture_routing(self):
        if not getattr(self.engine, 'batch', 0):
            return {}
        with self.engine.slot_cv:
            held = getattr(self.engine, 'slot_held', None)
            if not isinstance(held, list) or len(held) != self.engine.batch:
                raise ValueError('invalid native cached-slot routing metadata')
            return {i: tuple(tokens) for i, tokens in enumerate(held)}

    def quiesce_snapshot(self, store, timeout_s=30.0):
        """Durably checkpoint EVERY native session and remain loaded, gated.

        A coordinator can park several lanes before committing any unload. This
        operation also establishes a verified resume proof, including empty lanes.
        """
        checkpoint_id = None
        with self._transaction():
            try:
                if self._restore_failed:
                    raise RuntimeError('a prior lane operation failed; owner recovery is required')
                self._controller(store)
                self.gate_admission()
                if not self.wait_drained(timeout_s):
                    raise RuntimeError('snapshot lane did not drain')
                sessions = self.list_sessions()
                before = self._inventory_detail
                routing = self._capture_routing()
                checkpoint_id = store.publish(self.model, sessions, self.export_session)
                if self.list_sessions() != sessions or self._capture_routing() != routing:
                    raise RuntimeError('lane state changed during durable snapshot')
                self._parked_checkpoint = checkpoint_id
                self._parked_snapshot = before
                self._parked_sessions = sessions
                self._durable_inventory = before
                self._parked_stopped = False
                self._resume_inventory = before
                self._resume_held = routing
                return LifecycleResult('parked', checkpoint_id)
            except Exception as exc:
                return self._failure(exc, checkpoint_id)

    def _verify_parked_bundle(self, store, checkpoint_id):
        if (store is not self._store or checkpoint_id != self._parked_checkpoint
                or self._parked_snapshot is None):
            raise ValueError('checkpoint is not this lane latest verified full snapshot')
        if tuple(store.sessions(checkpoint_id, self.model)) != tuple(self._parked_sessions):
            raise ValueError('parked checkpoint inventory changed')
        # Revalidate payload integrity before discarding the only in-memory copy.
        # No native restore occurs: the reader is deliberately a no-op.
        for session in self._parked_sessions:
            store.restore(checkpoint_id, session.session_id, self.model, lambda *_: None)
        self._owned(drained=True)

    def stop_parked(self, store, checkpoint_id, timeout_s=30.0):
        """Stop only the exact, unchanged lane represented by its durable bundle."""
        with self._transaction():
            try:
                if self._restore_failed:
                    raise RuntimeError('a prior lane operation failed; owner recovery is required')
                if store is not self._store or checkpoint_id != self._parked_checkpoint:
                    raise ValueError('checkpoint is not this lane latest verified full snapshot')
                if self._parked_stopped:
                    old_proc, old_generation = self._gated_process
                    current = getattr(self.engine, 'proc', None)
                    if ((current is not None and (current is not old_proc or current.poll() is None))
                            or getattr(self.engine, 'gen', None) != old_generation):
                        raise RuntimeError('native process restarted after parked stop')
                    return LifecycleResult('stopped', checkpoint_id)
                self.gate_admission()
                if not self.wait_drained(timeout_s):
                    raise RuntimeError('parked lane did not drain')
                if self._snapshot()[2] != self._parked_snapshot:
                    raise RuntimeError('parked native inventory changed before stop')
                if self._capture_routing() != self._resume_held:
                    raise RuntimeError('parked slot routing changed before stop')
                self._verify_parked_bundle(store, checkpoint_id)
                self.unload()
                self._parked_stopped = True
                return LifecycleResult('stopped', checkpoint_id)
            except Exception as exc:
                return self._failure(exc, checkpoint_id)

    def release_slot(self, slot, timeout_s=30.0):
        """Evict a slot only while its current state is durably covered.

        Consecutive releases from one full snapshot are safe. A restore or resumed
        generation invalidates this proof; snapshot again before another eviction.
        """
        with self._transaction():
            try:
                if self._restore_failed:
                    raise RuntimeError('a prior lane operation failed; owner recovery is required')
                if (not getattr(self.engine, 'batch', 0)
                        or getattr(self.engine, 'session_batch_release', False) is not True
                        or not callable(getattr(self.engine, 'session_release_slot', None))):
                    raise ValueError('native batch slot release is not advertised')
                if type(slot) is not int or not 0 <= slot < self.engine.batch:
                    raise ValueError('release target slot is outside this lane')
                self.gate_admission()
                if not self.wait_drained(timeout_s):
                    raise RuntimeError('release lane did not drain')
                before = self._snapshot()[2]
                if self._durable_inventory is None or before != self._durable_inventory:
                    raise RuntimeError('slot release requires an unchanged verified full durable snapshot')
                target = f'slot-{slot}'
                if not any(row[0] == target and row[1] == 'slot' for row in before[1]):
                    raise ValueError('release target slot is not cached')
                if self._capture_routing() != self._resume_held:
                    raise RuntimeError('parked slot routing changed before release')
                self._verify_parked_bundle(self._store, self._parked_checkpoint)
                reply = self.engine.session_release_slot(slot)
                self._owned(drained=True)
                if not isinstance(reply, dict) or type(reply.get('slot')) is not int or reply['slot'] != slot:
                    raise ValueError('invalid native slot release acknowledgement')
                after = self._snapshot(released_slot=slot)[2]
                if (after[0] == before[0]
                        or after[1] != tuple(row for row in before[1] if row[0] != target)):
                    raise RuntimeError('slot release changed other state or failed to clear target')
                with self.engine.slot_cv:
                    self.engine.slot_held[slot] = []
                    used = getattr(self.engine, 'slot_used', None)
                    if isinstance(used, list) and len(used) == self.engine.batch:
                        used[slot] = 0.0
                self._resume_held[slot] = ()
                self._durable_inventory = after
                self._resume_inventory = after
                return LifecycleResult('released', self._parked_checkpoint)
            except Exception as exc:
                return self._failure(exc, self._parked_checkpoint)

    def restore_slot(self, store, checkpoint_id, session_id, slot, timeout_s=30.0):
        """Restore one verified bundle entry into an EMPTY native batch slot.

        Both native inventory and Python routing state must show an empty target.
        Other slots remain untouched. Multiple distinct slots can be restored
        under the sticky gate; any failed attempt requires owner recovery before
        resume. Fixed host residency estimates must cover retained slot cache storage
        AND the full checkpoint read-image scratch peak during each restore.
        """
        with self._transaction():
            try:
                if self._restore_failed:
                    raise RuntimeError('a prior lane operation failed; owner recovery is required')
                self._controller(store)
                if (not getattr(self.engine, 'batch', 0)
                        or getattr(self.engine, 'session_batch_restore', False) is not True
                        or not callable(getattr(self.engine, 'session_restore_slot', None))):
                    raise ValueError('native batch slot restore is not advertised')
                if type(slot) is not int or not 0 <= slot < self.engine.batch:
                    raise ValueError('restore target slot is outside this lane')
                self.gate_admission()
                if not self.wait_drained(timeout_s):
                    raise RuntimeError('restore lane did not drain')
                before = self._snapshot()[2]
                if self._resume_inventory is not None and before != self._resume_inventory:
                    raise RuntimeError('native inventory changed between slot restores')
                target = f'slot-{slot}'
                if any(row[0] == target for row in before[1]):
                    raise ValueError('restore target slot is already cached')
                with self.engine.slot_cv:
                    held = getattr(self.engine, 'slot_held', None)
                    if not isinstance(held, list) or len(held) != self.engine.batch or held[slot]:
                        raise ValueError('restore target has cached or invalid Python routing state')

                def restore(session, path):
                    if session.model != self.model:
                        raise ValueError('checkpoint session model identity mismatch')
                    reply = self.engine.session_restore_slot(slot, str(path))
                    self._owned(drained=True)
                    if not isinstance(reply, dict):
                        raise ValueError('invalid native slot restore acknowledgement')
                    count, ids = reply.get('tokens'), reply.get('ids')
                    if (type(count) is not int or count <= 0 or not isinstance(ids, list)
                            or len(ids) != count or any(type(i) is not int or not 0 <= i <= 0x7fffffff for i in ids)):
                        raise ValueError('restored slot token IDs/count are invalid')
                    after = self._snapshot()[2]
                    other = tuple(row for row in after[1] if row[0] != target)
                    new = [row for row in after[1] if row[0] == target]
                    if (after[0] == before[0] or other != before[1]
                            or new != [(target, 'slot', count)]):
                        raise RuntimeError('slot restore changed other state or failed to publish the target')
                    with self.engine.slot_cv:
                        self.engine.slot_held[slot] = list(ids)
                        used = getattr(self.engine, 'slot_used', None)
                        if isinstance(used, list) and len(used) == self.engine.batch:
                            used[slot] = time.time()
                    self._resume_held[slot] = tuple(ids)
                    self._resume_inventory = after
                    # New content is not covered by this lane's last full bundle.
                    self._durable_inventory = None

                store.restore(checkpoint_id, session_id, self.model, restore)
                return LifecycleResult('restored', checkpoint_id)
            except Exception as exc:
                self._restore_failed = True
                self.service.native_lane_gated = True
                try:
                    self.admission.gate_admission()
                except Exception:
                    pass
                return LifecycleResult('blocked', checkpoint_id, f'{type(exc).__name__}: {exc}')

    def resume(self, timeout_s=30.0):
        """Explicitly reopen a verified, unchanged restore or parked snapshot.

        No verified snapshot/restore, any failed operation, replacement or unresolved
        ownership or changed routing/inventory leaves admission gated. Reopening
        uses the same registered ledger worker, retaining its fixed residency.
        """
        with self._transaction():
            try:
                if self._restore_failed or self._resume_inventory is None:
                    raise RuntimeError('resume requires a verified snapshot or restore and no failed lane operation')
                if getattr(self.service, 'native_lane_gated', False) is not True:
                    raise RuntimeError('native lane is not gated')
                self.gate_admission()
                if not self.wait_drained(timeout_s):
                    raise RuntimeError('resume lane did not drain')
                if self._snapshot()[2] != self._resume_inventory:
                    raise RuntimeError('native inventory changed before resume')
                if self._resume_held:
                    with self.engine.slot_cv:
                        for slot, ids in self._resume_held.items():
                            if tuple(self.engine.slot_held[slot]) != ids:
                                raise RuntimeError('restored slot routing state changed before resume')
                self._owned(drained=True)
                self.admission.resume_admission()
                self.service.native_lane_gated = False
                self._gated_process = None
                self._resume_inventory = None
                self._resume_held = {}
                self._durable_inventory = None
                self._parked_checkpoint = None
                self._parked_snapshot = None
                self._parked_sessions = ()
                return LifecycleResult('resumed')
            except Exception as exc:
                self._restore_failed = True
                # A rejected resume never creates an ungated escape route.
                self.service.native_lane_gated = True
                try:
                    self.admission.gate_admission()
                except Exception:
                    pass
                return LifecycleResult('blocked', error=f'{type(exc).__name__}: {exc}')

    def gate_admission(self):
        self._owned()
        current = (self.engine.proc, getattr(self.engine, 'gen', None))
        if self._gated_process is not None and self._gated_process != current:
            raise RuntimeError('native process changed between gated lane operations')
        self._gated_process = current
        self.admission.gate_admission()
        self.service.native_lane_gated = True

    def wait_drained(self, timeout_s):
        self._owned()
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or not 0 <= timeout_s <= 60):
            raise ValueError('drain timeout must be finite and between 0 and 60 seconds')
        deadline = time.monotonic() + timeout_s
        if not self.admission.wait_drained(timeout_s):
            return False
        if not self.service.fifo.acquire(timeout=max(0.0, deadline - time.monotonic())):
            return False
        self._fifo_held = True
        self._validate_engine()
        self._same_process()
        if getattr(self.engine, 'batch', 0):
            ctl = getattr(self.engine, 'ctl', None)
            if ctl is None or not callable(getattr(ctl, 'acquire', None)):
                raise RuntimeError('native batch control lock is unavailable')
            if not ctl.acquire(timeout=max(0.0, deadline - time.monotonic())):
                return False
            self._ctl_held = ctl
            self._batch_idle(deadline)
            self._owned(drained=True)
        if not self.engine.alive():
            raise RuntimeError('native lane is not ready')
        # Admission stays gated. All ordinary state/control changes must use FIFO.
        return True

    def _snapshot(self, *, released_slot=None):
        self._owned(drained=True)
        inventory = self.engine.session_inventory()
        self._owned(drained=True)
        generation, ids = self._parse_inventory(inventory)
        batch = getattr(self.engine, 'batch', 0)
        slot_ids = {entry['id'] for entry in inventory['sessions'] if entry['kind'] == 'slot'}
        if any(not batch or int(sid.split('-')[1]) >= batch for sid in slot_ids):
            raise ValueError('native inventory contains a slot outside this lane')
        if batch and hasattr(self.engine, 'slot_held'):
            # This is a lower bound only: native inventory must also include
            # cached states unknown to Python. Never silently omit a known one.
            with self.engine.slot_cv:
                held = self.engine.slot_held
                if not isinstance(held, list) or len(held) != batch:
                    raise ValueError('invalid native cached-slot ownership metadata')
                known = {f'slot-{i}' for i, tokens in enumerate(held) if tokens and i != released_slot}
            if not known <= slot_ids:
                raise ValueError('native inventory omitted a known cached slot')
        detail = (generation, tuple((e['id'], e['kind'], e['tokens']) for e in inventory['sessions']))
        return generation, ids, detail

    def list_sessions(self):
        generation, ids, detail = self._snapshot()
        if self._inventory_detail is not None and self._inventory_detail != detail:
            raise RuntimeError('native inventory changed during checkpoint transaction')
        self._inventory = generation, ids
        self._inventory_detail = detail
        return tuple(SessionMetadata(f'{generation}:{sid}', self.model) for sid in ids)

    @staticmethod
    def _parse_inventory(inventory):
        if not isinstance(inventory, dict) or set(inventory) != {'generation', 'sessions'}:
            raise ValueError('invalid native session inventory')
        generation, entries = inventory['generation'], inventory['sessions']
        if (not isinstance(generation, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', generation)
                or not isinstance(entries, list) or len(entries) > 1024):
            raise ValueError('invalid native session inventory generation/list')
        ids = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {'id', 'kind', 'tokens'}:
                raise ValueError('invalid native inventory entry')
            sid = entry['id']
            if (not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', sid)
                    or entry['kind'] not in ('live', 'parked', 'slot') or type(entry['tokens']) is not int
                    or entry['tokens'] < 0 or sid in ids):
                raise ValueError('invalid or duplicate native session identifier')
            if entry['kind'] == 'slot' and not re.fullmatch(r'slot-(0|[1-9][0-9]*)', sid):
                raise ValueError('invalid native slot identifier')
            ids.append(sid)
        return generation, tuple(ids)

    def export_session(self, session, path: Path):
        self._owned(drained=True)
        if not isinstance(session, SessionMetadata) or session.model != self.model or self._inventory is None:
            raise ValueError('session was not inventoried for this model')
        generation, ids = self._inventory
        candidates = {f'{generation}:{sid}': sid for sid in ids}
        if session.session_id not in candidates:
            raise ValueError('session does not belong to the inventoried generation')
        sid = candidates[session.session_id]
        expected_tokens = next(row[2] for row in self._inventory_detail[1] if row[0] == sid)
        reply = self.engine.session_export(generation, sid, str(path))
        self._owned(drained=True)
        # A valid checksum only proves the bytes we received. Cross-check the
        # native completion against the frozen inventory before publishing or
        # discarding the source's only in-memory state.
        if (not isinstance(reply, dict) or type(reply.get('tokens')) is not int
                or reply['tokens'] != expected_tokens or type(reply.get('bytes')) is not int
                or reply['bytes'] != path.stat().st_size
                or type(reply.get('ms')) not in (int, float)
                or not math.isfinite(reply['ms']) or reply['ms'] < 0):
            raise ValueError('native export acknowledgement does not match inventory or payload')

    def unload(self):
        self._owned(drained=True)
        proc = getattr(self.engine, 'proc', None)
        if proc is None:
            raise RuntimeError('cannot confirm native process identity before unload')
        self.engine.close()
        if proc.poll() is None:
            raise RuntimeError('native process exit was not confirmed')
        if (getattr(self.engine, 'proc', None) not in (None, proc)
                or getattr(self.engine, 'gen', None) != self._process_pin[1]):
            raise RuntimeError('native process restarted during lane shutdown')
        # Fixed residency survives uncertain shutdown. The ledger additionally
        # proves the admission gate is closed and every lease has drained.
        self.admission.ledger.unregister_worker(self.admission.worker)

    def restore_session(self, session, path: Path):
        self._owned(drained=True)
        if not isinstance(session, SessionMetadata) or session.model != self.model:
            raise ValueError('checkpoint session model identity mismatch')
        if getattr(self.engine, 'batch', 0):
            raise ValueError('native RESTORE requires an unbatched destination lane')
        self.engine.session_file('restore', str(path))
        self._owned(drained=True)
