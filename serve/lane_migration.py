"""Opt-in owner API for durable lane consolidation and on-demand slot rotation.

All requests for these lanes must use generate(), or the caller must provide an
exclusive routing gate. No HTTP route, automatic lane discovery, or deployment.
The journal records durable checkpoint references; it is not a per-token WAL.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import math
import os
import threading
import time
import uuid

from serve.checkpoint_store import CheckpointStore, _sync_dir
from serve.drain_receipt import DrainReceipt
from serve.native_lane import NativeLaneWorker


class MigrationBusy(RuntimeError):
    """Retryable routing wait; no native state or admission gate was changed."""


@dataclass(frozen=True)
class AgentRoute:
    lane: str | None
    slot: int | None
    checkpoint_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class MigrationResult:
    state: str  # consolidated, ready, waiting, blocked
    journal_path: str
    error: str | None = None


class MigrationCoordinator:
    def __init__(self, lanes, store, agents, *, enabled=False):
        if enabled is not True:
            raise ValueError('lane migration requires explicit opt-in')
        if not isinstance(store, CheckpointStore) or not isinstance(lanes, dict) or not lanes:
            raise ValueError('migration requires lanes and a durable checkpoint store')
        if any(not isinstance(k, str) or not k or not isinstance(w, NativeLaneWorker)
               for k, w in lanes.items()):
            raise ValueError('lanes must have unique names and native lane workers')
        if len({id(w.engine) for w in lanes.values()}) != len(lanes):
            raise ValueError('a native engine cannot be registered as multiple lanes')
        if len({(id(w.admission.ledger), w.admission.worker) for w in lanes.values()}) != len(lanes):
            raise ValueError('lanes must have distinct registered admission workers')
        self.model = next(iter(lanes.values())).model
        for w in lanes.values():
            w._validate_engine()  # Same runtime precision/identity audit as every native operation.
            if (w.model != self.model or w.engine.batch <= 0
                    or getattr(w.engine, 'session_batch_restore', False) is not True
                    or getattr(w.engine, 'session_batch_release', False) is not True
                    or getattr(w.engine, 'session_batch_route', False) is not True):
                raise ValueError('migration lanes require matching audited models and batch restore/release/owned routing')
        if not isinstance(agents, dict) or not agents:
            raise ValueError('explicit agent-to-slot bindings are required')
        routes, occupied = {}, set()
        for agent, binding in agents.items():
            if not isinstance(agent, str) or not agent or not isinstance(binding, (tuple, list)) or len(binding) != 2:
                raise ValueError('invalid agent binding')
            lane, slot = binding
            if (lane not in lanes or type(slot) is not int or not 0 <= slot < lanes[lane].engine.batch
                    or (lane, slot) in occupied):
                raise ValueError('agent bindings must identify distinct valid native slots')
            occupied.add((lane, slot))
            routes[agent] = AgentRoute(lane, slot)
        self.lanes, self.store, self._routes = dict(lanes), store, routes
        self._processes = {name: (w.engine.proc, getattr(w.engine, "gen", None)) for name, w in lanes.items()}
        self._operation = threading.Lock()
        self._cv = threading.Condition()
        self._active = set()
        self._blocked = False
        self._destination = None
        self._stopped = set()
        self._snapshots = {}
        self._clock = len(routes)
        self._used = {agent: i for i, agent in enumerate(routes)}
        self.journal_path = store.root / ('migration-' + uuid.uuid4().hex + '.json')
        self._journal('registered')

    @property
    def routes(self):
        with self._cv:
            return dict(self._routes)

    def _journal(self, phase, error=None):
        # Trusted local owner only. Write a new inode, fsync it, replace atomically,
        # then fsync the directory before any destructive step is permitted.
        payload = {'version': 1, 'phase': phase, 'model': asdict(self.model),
                   'destination': self._destination, 'stopped_lanes': sorted(self._stopped),
                   'snapshots': dict(self._snapshots),
                   'agents': {k: asdict(v) for k, v in self._routes.items()}, 'error': error,
                   'checkpoint_scope': 'last successful quiescent snapshot, not subsequent generated tokens'}
        temp = self.journal_path.with_name('.' + self.journal_path.name + '.' + uuid.uuid4().hex)
        with temp.open('x', encoding='utf-8') as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.journal_path)
        _sync_dir(self.journal_path.parent)

    @staticmethod
    def _timeout(timeout_s):
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 <= timeout_s <= 60):
            raise ValueError('migration timeout must be between 0 and 60 seconds')

    def _idle(self, timeout_s):
        self._timeout(timeout_s)
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._active:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MigrationBusy('routed agents have not finished their turns')
                self._cv.wait(left)

    def _result(self, state, error=None):
        return MigrationResult(state, str(self.journal_path), error)

    def _fail(self, exc):
        self._blocked = True
        # Do not kill/restart uncertain workers. Keep their leases and fixed costs.
        for lane, worker in self.lanes.items():
            if lane not in self._stopped:
                worker.service.native_lane_gated = True
                try:
                    worker.admission.gate_admission()
                except Exception:
                    pass
        error = f'{type(exc).__name__}: {exc}'
        try:
            self._journal('blocked', error)
        except Exception:
            # The last successfully synced journal and immutable bundles survive.
            pass
        return self._result('blocked', error)

    @staticmethod
    def _require(result, state):
        if result.state != state:
            raise RuntimeError(result.error or f'expected {state}, received {result.state}')
        return result

    def _check_lane(self, lane):
        worker = self.lanes[lane]
        worker._validate_engine()
        proc, generation = self._processes[lane]
        if (worker.engine.proc is not proc or getattr(worker.engine, 'gen', None) != generation
                or not worker.engine.alive()):
            raise RuntimeError('registered native lane process changed or exited')
        return worker

    def _snapshot(self, lane, timeout_s):
        worker = self._check_lane(lane)
        result = self._require(worker.quiesce_snapshot(self.store, timeout_s), 'parked')
        bundle = result.checkpoint_id
        sessions = self.store.sessions(bundle, self.model)
        by_slot = {}
        for session in sessions:
            sid = session.session_id.rsplit(':', 1)[-1]
            if sid.startswith('slot-'):
                by_slot[int(sid[5:])] = session.session_id
        # Persist every completed lane's references before proceeding to the next.
        self._snapshots[lane] = bundle
        with self._cv:
            for agent, route in self._routes.items():
                if route.lane == lane:
                    if route.slot not in by_slot:
                        raise RuntimeError(f'agent {agent} has no retained native slot on {lane}')
                    self._routes[agent] = replace(route, checkpoint_id=bundle, session_id=by_slot[route.slot])
        self._journal('snapshotting')
        return set(by_slot)

    def _verify_parked(self, route):
        if route.checkpoint_id is None or route.session_id is None:
            raise RuntimeError('agent has no durable checkpoint')
        # Checks model binding, exact selected session, bytes and SHA256 before eviction.
        self.store.restore(route.checkpoint_id, route.session_id, self.model, lambda *_: None)

    def consolidate(self, destination, timeout_s=30.0):
        """Snapshot every lane before retiring any; retain overflow on disk."""
        if not self._operation.acquire(blocking=False):
            return self._result('waiting', 'another routing operation is running')
        try:
            if self._blocked or self._destination is not None:
                raise RuntimeError('coordinator is blocked or already consolidated')
            if destination not in self.lanes:
                raise ValueError('unknown destination lane')
            self._idle(timeout_s)
            self._destination = destination
            self._journal('preparing')
            occupied = {}
            for lane in self.lanes:
                occupied[lane] = self._snapshot(lane, timeout_s)
            self._journal('all-snapshots-durable')
            destination_worker = self.lanes[destination]
            free = [slot for slot in range(destination_worker.engine.batch) if slot not in occupied[destination]]
            for agent, route in list(self._routes.items()):
                if route.lane == destination:
                    continue
                self._verify_parked(route)
                if free:
                    slot = free.pop(0)
                    self._require(destination_worker.restore_slot(self.store, route.checkpoint_id,
                                  route.session_id, slot, timeout_s), 'restored')
                    self._routes[agent] = replace(route, lane=destination, slot=slot)
                else:
                    self._routes[agent] = replace(route, lane=None, slot=None)
            # The durable journal now identifies both resident and parked agents,
            # with a fallback bundle for each, before any source is stopped.
            self._journal('routes-durable')
            for lane, worker in self.lanes.items():
                if lane != destination:
                    self._require(worker.stop_parked(self.store, self._snapshots[lane], timeout_s), 'stopped')
                    self._stopped.add(lane)
                    self._journal('retiring-sources')
            self._require(destination_worker.resume(timeout_s), 'resumed')
            self._journal('consolidated')
            return self._result('consolidated')
        except MigrationBusy as exc:
            return self._result('waiting', str(exc))
        except Exception as exc:
            return self._fail(exc)
        finally:
            self._operation.release()

    def _activate(self, agent, timeout_s):
        if self._blocked:
            raise RuntimeError('migration is blocked; explicit owner recovery is required')
        if agent not in self._routes:
            raise ValueError('unknown agent')
        route = self._routes[agent]
        if route.lane is not None:
            self._check_lane(route.lane)
            return
        if self._destination is None:
            raise RuntimeError('no destination lane has been selected')
        self._idle(timeout_s)
        self._verify_parked(route)  # Reject corrupt source before any slot is released.
        lane = self._destination
        worker = self.lanes[lane]
        occupied = self._snapshot(lane, timeout_s)
        free = [slot for slot in range(worker.engine.batch) if slot not in occupied]
        if free:
            slot = free[0]
        else:
            candidates = [a for a, r in self._routes.items() if r.lane == lane]
            if not candidates:
                raise RuntimeError('destination has no managed slot available for rotation')
            victim = min(candidates, key=self._used.__getitem__)
            old = self._routes[victim]
            slot = old.slot
            self._verify_parked(old)
            self._journal('rotation-durable')
            self._require(worker.release_slot(slot, timeout_s), 'released')
            self._routes[victim] = replace(old, lane=None, slot=None)
            self._journal('slot-released')
        self._require(worker.restore_slot(self.store, route.checkpoint_id, route.session_id, slot, timeout_s), 'restored')
        self._routes[agent] = replace(route, lane=lane, slot=slot)
        self._journal('agent-restored')
        self._require(worker.resume(timeout_s), 'resumed')
        self._journal('consolidated')

    def activate(self, agent, timeout_s=30.0):
        """Bring a parked agent into a slot; durable LRU parking makes room."""
        if not self._operation.acquire(blocking=False):
            return self._result('waiting', 'another routing operation is running')
        try:
            self._timeout(timeout_s)
            self._activate(agent, timeout_s)
            self._clock += 1
            self._used[agent] = self._clock
            return self._result('ready')
        except MigrationBusy as exc:
            return self._result('waiting', str(exc))
        except Exception as exc:
            return self._fail(exc)
        finally:
            self._operation.release()

    @contextmanager
    def route(self, agent, timeout_s=30.0):
        """Hold ownership of one agent's slot through an entire request iterator.

        This is an owner primitive, not permission to bypass admission. Prefer
        generate(), which also binds native dispatch and exact drain receipts.
        """
        self._timeout(timeout_s)
        deadline = time.monotonic() + timeout_s
        if not self._operation.acquire(timeout=timeout_s):
            raise MigrationBusy('another routing operation is running')
        try:
            try:
                self._activate(agent, max(0.0, deadline - time.monotonic()))
                with self._cv:
                    if agent in self._active:
                        raise MigrationBusy('agent already has an active turn')
                    self._active.add(agent)
                    self._clock += 1
                    self._used[agent] = self._clock
                    route = self._routes[agent]
            except MigrationBusy:
                raise
            except Exception as exc:
                self._fail(exc)
                raise
        finally:
            self._operation.release()
        try:
            yield route
        finally:
            with self._cv:
                self._active.discard(agent)
                self._cv.notify_all()

    def generate(self, agent, ids, max_new, sampling, cancel, *, timeout_s=30.0):
        """Owner-facing token iterator with explicit slot and admission ownership.

        Callers provide the normal tokenized next-turn prompt. Cached computation
        is reused; a transcript is still needed to construct that prompt. Closing
        this iterator retains the lease until its exact native drain is confirmed.
        """
        with self.route(agent, timeout_s) as route:
            worker = self.lanes[route.lane]
            worker._validate_engine()
            receipt = DrainReceipt()
            def tracked():
                with worker.engine.bound_slot(route.slot), worker.engine.track_drain(receipt):
                    yield from worker.engine.generate(ids, max_new, sampling, cancel)
            yield from worker.admission.run(tracked, len(ids), max_new, drain_receipt=receipt)
