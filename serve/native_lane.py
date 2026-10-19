"""Opt-in native lane checkpoint adapter; no background handlers or automatic unload.

Only the audited, non-batched Flash-Next lane is admitted. The public operations
hold the service FIFO from completed request drain through export/restore/stop.
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
from serve.lane_lifecycle import LaneLifecycle
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
        self._inventory = None
        self._lifecycle = None
        self._store = None
        self._validate_engine()

    def _validate_engine(self):
        if getattr(self.engine, '_session_protocol_failed', False):
            raise RuntimeError('native session protocol is blocked; restore cannot safely use its command stream')
        if self.service.engine is not self.engine or self.service.request_admission is not self.admission:
            raise RuntimeError('lane engine or admission controller changed')
        if getattr(self.engine, 'batch', 0):
            raise ValueError('native lane lifecycle does not support batch sessions')
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
        files = audit.get('components', {}).get('target', {}).get('files', [])
        if not files or any(f.get('architecture') != self.model.architecture for f in files):
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
        try:
            yield
        finally:
            if self._fifo_held:
                self.service.fifo.release()
                self._fifo_held = False
            self._owner = None
            self._operation.release()

    def _owned(self, *, drained=False):
        if self._owner != threading.get_ident():
            raise RuntimeError('use the native lane public lifecycle operations')
        if drained and not self._fifo_held:
            raise RuntimeError('native lane has not drained under the service FIFO')
        self._validate_engine()

    def _controller(self, store):
        if self._lifecycle is None:
            self._store = store
            self._lifecycle = LaneLifecycle(self, store, self.model)
        elif store is not self._store:
            raise ValueError('a lane lifecycle cannot switch checkpoint stores')
        return self._lifecycle

    def persist_and_stop(self, store, timeout_s=30.0):
        with self._transaction():
            return self._controller(store).persist_and_stop(timeout_s)

    def lazy_restore(self, store, checkpoint_id, session_id, timeout_s=30.0):
        with self._transaction():
            return self._controller(store).lazy_restore(checkpoint_id, session_id, timeout_s)

    def gate_admission(self):
        self._owned()
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
        if not self.engine.alive():
            raise RuntimeError('native lane is not ready')
        # Admission stays gated. All ordinary state/control changes must use FIFO.
        return True

    def list_sessions(self):
        self._owned(drained=True)
        inventory = self.engine.session_inventory()
        generation, ids = self._parse_inventory(inventory)
        if self._inventory is not None and self._inventory != (generation, ids):
            raise RuntimeError('native inventory changed during checkpoint transaction')
        self._inventory = generation, ids
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
                    or entry['kind'] not in ('live', 'parked') or type(entry['tokens']) is not int
                    or entry['tokens'] < 0 or sid in ids):
                raise ValueError('invalid or duplicate native session identifier')
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
        self.engine.session_export(generation, candidates[session.session_id], str(path))

    def unload(self):
        self._owned(drained=True)
        proc = getattr(self.engine, 'proc', None)
        if proc is None:
            raise RuntimeError('cannot confirm native process identity before unload')
        self.engine.close()
        if proc.poll() is None:
            raise RuntimeError('native process exit was not confirmed')
        # Fixed residency survives uncertain shutdown. The ledger additionally
        # proves the admission gate is closed and every lease has drained.
        self.admission.ledger.unregister_worker(self.admission.worker)

    def restore_session(self, session, path: Path):
        self._owned(drained=True)
        if not isinstance(session, SessionMetadata) or session.model != self.model:
            raise ValueError('checkpoint session model identity mismatch')
        self.engine.session_file('restore', str(path))
