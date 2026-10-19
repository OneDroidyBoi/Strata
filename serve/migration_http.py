"""Explicit HTTP binding for an owner-created MigrationCoordinator.

The request scope must surround the complete logical response, including every
internal tool/reasoning continuation and iterator cleanup. It binds a slot but
creates no admission reservation or drain receipt: Service.run owns those. All
managed Service.run/control paths must enforce this binding in server.py.

This module does not discover lanes, load models, choose checkpoint paths from
HTTP, or start listeners during construction. Agent IDs are names, not API keys;
the HTTP handler must authenticate its frontend BEFORE entering request().
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hmac
import ipaddress
import math
import re
import threading

from serve.lane_migration import MigrationBusy, MigrationCoordinator

_AGENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_SECURITY = frozenset({'api_key', 'cors_origins', 'trusted_origins', 'host_names', 'allowed_hosts'})


class MigrationHTTPError(RuntimeError):
    def __init__(self, status, message, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class MigrationHTTPResult:
    state: str


@dataclass(frozen=True)
class _Binding:
    agent: str
    service: object
    worker: object
    process: object
    generation: object


def _one_header(headers, name):
    if callable(getattr(headers, 'get_all', None)):
        values = headers.get_all(name, [])
    elif callable(getattr(headers, 'items', None)):
        values = [value for key, value in headers.items() if key.lower() == name.lower()]
    else:
        raise MigrationHTTPError(400, 'Invalid HTTP headers')
    if len(values) > 1:
        raise MigrationHTTPError(400, f'{name} must occur exactly once')
    return values[0] if values else None


def _timeout(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 60):
        raise MigrationHTTPError(400, 'Migration timeout must be between 0 and 60 seconds')


class _ServiceProxy:
    def __init__(self, gateway, frontend):
        object.__setattr__(self, '_gateway', gateway)
        object.__setattr__(self, '_frontend', frontend)

    def _target(self, name):
        if name in _SECURITY or name == '_migration_http_started':
            return self._frontend
        binding = getattr(self._gateway._local, 'binding', None)
        if binding is not None:
            return binding.service
        destination = getattr(self._gateway.coordinator, '_destination', None)
        worker = self._gateway.coordinator.lanes.get(destination)
        return worker.service if worker is not None else self._frontend

    def __getattr__(self, name):
        if name == 'migration_gateway':
            return self._gateway
        return getattr(self._target(name), name)

    def __setattr__(self, name, value):
        if name.startswith('_gateway') or name == '_frontend':
            raise AttributeError('gateway proxy identity is immutable')
        if name == 'migration_gateway':
            if value is not self._gateway:
                raise ValueError('cannot replace a migration proxy gateway')
            return
        setattr(self._target(name), name, value)


class MigrationGateway:
    def __init__(self, coordinator, *, enabled=False, admin_token=None):
        if enabled is not True or not isinstance(coordinator, MigrationCoordinator):
            raise ValueError('HTTP migration requires explicit opt-in and an owner-created coordinator')
        if admin_token is not None and (not isinstance(admin_token, str) or not admin_token
                or len(admin_token) > 4096 or not admin_token.isascii()
                or any(c.isspace() or ord(c) < 33 for c in admin_token)):
            raise ValueError('admin_token must be a nonempty ASCII token without whitespace')
        enrolled = frozenset(coordinator.routes)
        if not enrolled or any(not isinstance(a, str) or not _AGENT.fullmatch(a) for a in enrolled):
            raise ValueError('enrolled HTTP agent IDs must be 1..128 ASCII letters/digits/dot/underscore/hyphen')
        services = [worker.service for worker in coordinator.lanes.values()]
        if len({id(s) for s in services}) != len(services):
            raise ValueError('HTTP migration lanes must use distinct services')
        # Validate everything before attaching anything; no partial enrollment.
        for service in services:
            if getattr(service, '_migration_http_started', False):
                raise ValueError('attach migration before any managed service handler starts')
            if getattr(service, 'migration_gateway', None) is not None:
                raise ValueError('a managed service already has a migration gateway')
            if getattr(service, 'vision', None) is not None or getattr(service, 'images', None) is not None or getattr(service, 'mcp', None) is not None:
                raise ValueError('managed HTTP lanes cannot use server-side vision, images or MCP')
        self.coordinator = coordinator
        self._enrolled = enrolled
        self._services = tuple(services)
        self._admin_token = admin_token
        self._local = threading.local()
        self._admin_lock = threading.Lock()
        self._consolidated_destination = None
        for service in services:
            service.migration_gateway = self

    def proxy(self, frontend):
        if isinstance(frontend, _ServiceProxy):
            if frontend._gateway is not self:
                raise ValueError('frontend belongs to another migration gateway')
            return frontend
        return _ServiceProxy(self, frontend)

    def agent_from_headers(self, headers):
        return self._agent(_one_header(headers, 'X-Strata-Agent-ID'))

    @staticmethod
    def admin_from_headers(headers):
        return _one_header(headers, 'X-Strata-Migration-Key')

    def _agent(self, agent):
        if not isinstance(agent, str) or not _AGENT.fullmatch(agent):
            raise MigrationHTTPError(400, 'A valid X-Strata-Agent-ID header is required')
        # Unknown clients must never reach coordinator.route: its owner API
        # treats an unknown binding as an invariant failure and blocks lanes.
        if agent not in self._enrolled or agent not in self.coordinator.routes:
            raise MigrationHTTPError(404, 'Agent is not enrolled for migration')
        return agent

    @contextmanager
    def request(self, agent_id, cancel=None, timeout_s=30.0):
        agent = self._agent(agent_id)
        _timeout(timeout_s)
        if getattr(self._local, 'binding', None) is not None:
            raise MigrationHTTPError(400, 'Nested migration request scopes are not supported')
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise MigrationHTTPError(400, 'Invalid request cancellation signal')
        if cancel is not None and cancel.is_set():
            raise MigrationHTTPError(503, 'Request was cancelled before routing')
        stack = ExitStack()
        try:
            route = stack.enter_context(self.coordinator.route(agent, timeout_s))
            if cancel is not None and cancel.is_set():
                raise MigrationHTTPError(503, 'Request was cancelled before dispatch')
            worker = self.coordinator.lanes[route.lane]
            worker._validate_engine()
            service = worker.service
            if service.migration_gateway is not self or service.engine is not worker.engine or service.request_admission is not worker.admission:
                raise RuntimeError('managed service binding changed')
            stack.enter_context(worker.engine.bound_slot(route.slot))
            binding = _Binding(agent, service, worker, worker.engine.proc, getattr(worker.engine, 'gen', None))
        except Exception as exc:
            stack.close()
            if isinstance(exc, MigrationHTTPError):
                raise
            if isinstance(exc, MigrationBusy):
                raise MigrationHTTPError(503, 'Agent routing is busy; retry later', retry_after=1) from None
            raise MigrationHTTPError(503, 'Migration routing is unavailable; owner recovery may be required') from None
        self._local.binding = binding
        try:
            yield service
        finally:
            self._local.binding = None
            stack.close()

    def require_bound(self, service):
        binding = getattr(self._local, 'binding', None)
        if binding is None or binding.service is not service:
            raise MigrationHTTPError(403, 'Managed lanes require an enrolled agent request')
        worker = binding.worker
        try:
            worker._validate_engine()
            if (service.engine is not worker.engine or service.request_admission is not worker.admission
                    or worker.engine.proc is not binding.process or getattr(worker.engine, 'gen', None) != binding.generation
                    or not worker.engine.alive()):
                raise RuntimeError('managed native process changed')
        except Exception:
            raise MigrationHTTPError(503, 'Managed lane changed or exited; automatic restart is disabled') from None

    def _authorize_admin(self, token):
        if self._admin_token is None:
            raise MigrationHTTPError(404, 'HTTP migration administration is disabled')
        if (not isinstance(token, str) or not token.isascii()
                or not hmac.compare_digest(token, self._admin_token)):
            raise MigrationHTTPError(403, 'Migration operator authentication failed')

    def consolidate(self, destination, token, timeout_s=30.0):
        self._authorize_admin(token)
        _timeout(timeout_s)
        if not isinstance(destination, str) or destination not in self.coordinator.lanes:
            raise MigrationHTTPError(400, 'Unknown destination lane')
        if not self._admin_lock.acquire(blocking=False):
            raise MigrationHTTPError(503, 'Migration administration is busy; retry later', retry_after=1)
        try:
            if getattr(self.coordinator, '_blocked', False):
                raise MigrationHTTPError(503, 'Migration is blocked; explicit owner recovery is required')
            selected = getattr(self.coordinator, '_destination', None)
            if self._consolidated_destination is not None:
                if destination == self._consolidated_destination and selected == destination:
                    return MigrationHTTPResult('consolidated')
                raise MigrationHTTPError(409, 'Migration already selected another destination')
            if selected is not None:
                # An external owner may have completed or started consolidation.
                # Never repeat the destructive owner operation based on HTTP.
                raise MigrationHTTPError(409, 'Migration destination was already selected by its owner')
            try:
                result = self.coordinator.consolidate(destination, timeout_s)
            except Exception:
                raise MigrationHTTPError(503, 'Migration consolidation is unavailable') from None
            if result.state == 'waiting':
                raise MigrationHTTPError(503, 'Migration is waiting for active requests', retry_after=1)
            if result.state != 'consolidated':
                raise MigrationHTTPError(503, 'Migration is blocked; explicit owner recovery is required')
            self._consolidated_destination = destination
            return MigrationHTTPResult('consolidated')
        finally:
            self._admin_lock.release()

    def validate_bind(self, frontend, host):
        """Apply the same bind policy to gateway.serve and raw server.serve."""
        if isinstance(frontend, _ServiceProxy) and frontend._gateway is self:
            frontend = frontend._frontend
        if not any(frontend is service for service in self._services):
            raise ValueError('HTTP frontend must be one of this gateway registered services')
        if not isinstance(host, str) or not host:
            raise ValueError('HTTP bind host must be a nonempty string')
        try:
            loopback = host == 'localhost' or ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback and not getattr(frontend, 'api_key', None):
            raise ValueError('non-loopback HTTP migration requires a frontend API key')

    def serve(self, frontend=None, host='127.0.0.1', port=8095):
        """Start the existing HTTP server over already-created native services."""
        if frontend is None:
            frontend = self._services[0]
        self.validate_bind(frontend, host)
        from serve.server import serve
        return serve(frontend, host=host, port=port)
