"""Gateway scope/auth/proxy tests; no listeners, models, or GPU work."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from email.message import Message
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve.lane_migration import AgentRoute, MigrationBusy, MigrationCoordinator, MigrationResult
from serve.migration_http import MigrationGateway, MigrationHTTPError


class Engine:
    def __init__(self):
        self.proc, self.gen, self.local = object(), 1, threading.local()

    @contextmanager
    def bound_slot(self, slot):
        previous = getattr(self.local, 'slot', None)
        self.local.slot = slot
        try:
            yield slot
        finally:
            self.local.slot = previous

    def alive(self):
        return True


class Coordinator(MigrationCoordinator):
    """Only owner orchestration is fake; production gateway behavior is unchanged."""
    def __init__(self):
        self.lanes = {}
        for name in ('one', 'two'):
            service = SimpleNamespace(name=name, engine=Engine(), request_admission=object(),
                api_key=name + '-key', cors_origins=[name], trusted_origins=[name],
                host_names=[name], allowed_hosts=[name], value=0)
            self.lanes[name] = SimpleNamespace(service=service, engine=service.engine,
                admission=service.request_admission, _validate_engine=lambda: None)
        self._routes = {'agent-a': AgentRoute('one', 0), 'agent-b': AgentRoute('two', 1)}
        self._active = set()
        self._lock = threading.Lock()
        self._destination = None
        self.route_calls = self.consolidate_calls = 0
        self.before_yield = None
        self.failure = None

    @property
    def routes(self):
        return dict(self._routes)

    @contextmanager
    def route(self, agent, timeout_s=30):
        self.route_calls += 1
        with self._lock:
            if agent in self._active:
                raise MigrationBusy('secret owner detail')
            self._active.add(agent)
        try:
            if self.before_yield:
                self.before_yield()
            yield self._routes[agent]
        finally:
            with self._lock:
                self._active.discard(agent)

    def consolidate(self, destination, timeout_s=30):
        self.consolidate_calls += 1
        if self.failure:
            return MigrationResult('blocked', '/private/secret.json', self.failure)
        if self._active:
            return MigrationResult('waiting', '/private/secret.json', 'active agent names')
        self._destination = destination
        self._routes = {a: AgentRoute(destination, i) for i, a in enumerate(self._routes)}
        return MigrationResult('consolidated', '/private/secret.json')


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = Coordinator()
        self.gateway = MigrationGateway(self.coordinator, enabled=True, admin_token='operator-secret')
        self.frontend = self.coordinator.lanes['one'].service
        self.proxy = self.gateway.proxy(self.frontend)

    def error(self, status, operation):
        with self.assertRaises(MigrationHTTPError) as caught:
            operation()
        self.assertEqual(caught.exception.status, status)
        return caught.exception

    def test_header_missing_duplicate_invalid_and_unknown_never_route(self):
        for value, status in [(None, 400), ('', 400), (' agent-a', 400), ('a,b', 400), ('../a', 400), ('unknown', 404)]:
            headers = {} if value is None else {'X-Strata-Agent-ID': value}
            self.error(status, lambda: self.gateway.agent_from_headers(headers))
        headers = Message()
        headers['X-Strata-Agent-ID'] = 'agent-a'
        headers['x-strata-agent-id'] = 'agent-b'
        self.error(400, lambda: self.gateway.agent_from_headers(headers))
        self.assertEqual(self.coordinator.route_calls, 0)
        self.assertEqual(self.gateway.agent_from_headers({'x-strata-agent-id': 'agent-a'}), 'agent-a')

    def test_full_request_pin_spans_multiple_internal_runs_and_blocks_admin(self):
        with self.gateway.request('agent-b') as service:
            self.assertIs(service, self.coordinator.lanes['two'].service)
            self.assertIs(self.proxy.engine, service.engine)
            for _ in range(3):
                self.gateway.require_bound(service)
                self.assertEqual(service.engine.local.slot, 1)
                self.assertIn('agent-b', self.coordinator._active)
            error = self.error(503, lambda: self.gateway.consolidate('one', 'operator-secret', 0))
            self.assertEqual(error.retry_after, 1)
        self.assertFalse(self.coordinator._active)
        self.assertIsNone(service.engine.local.slot)
        self.error(403, lambda: self.gateway.require_bound(service))

    def test_security_proxy_stays_frontend_and_writes_selected_data_only(self):
        with self.gateway.request('agent-b'):
            for name in ('api_key', 'cors_origins', 'trusted_origins', 'host_names', 'allowed_hosts'):
                self.assertEqual(getattr(self.proxy, name), getattr(self.frontend, name))
            self.proxy.value = 9
            self.assertEqual(self.coordinator.lanes['two'].service.value, 9)
            self.assertEqual(self.frontend.value, 0)
            self.proxy.api_key = 'new-front-key'
            self.assertEqual(self.frontend.api_key, 'new-front-key')
            self.assertEqual(self.coordinator.lanes['two'].service.api_key, 'two-key')
        self.gateway.consolidate('two', 'operator-secret')
        self.assertIs(self.proxy.engine, self.coordinator.lanes['two'].engine)
        self.assertEqual(self.proxy.api_key, 'new-front-key')

    def test_thread_local_selected_service_does_not_cross_requests(self):
        barrier = threading.Barrier(2)
        def selected(agent):
            with self.gateway.request(agent) as service:
                barrier.wait(timeout=1)
                self.gateway.require_bound(service)
                return self.proxy.name, service.engine.local.slot, self.proxy.api_key
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(selected, 'agent-a')
            second = pool.submit(selected, 'agent-b')
            self.assertEqual(first.result(timeout=2), ('one', 0, 'one-key'))
            self.assertEqual(second.result(timeout=2), ('two', 1, 'one-key'))

    def test_body_exception_closes_route_and_restores_slot_scope(self):
        with self.assertRaisesRegex(ValueError, 'caller failure'):
            with self.gateway.request('agent-a'):
                raise ValueError('caller failure')
        self.assertFalse(self.coordinator._active)
        self.assertIsNone(self.frontend.engine.local.slot)

    def test_cancelled_request_does_not_dispatch_and_midroute_cancel_unpins(self):
        cancel = threading.Event()
        cancel.set()
        def request():
            with self.gateway.request('agent-a', cancel):
                self.fail('cancelled request was yielded')
        self.error(503, request)
        self.assertEqual(self.coordinator.route_calls, 0)
        cancel.clear()
        self.coordinator.before_yield = cancel.set
        self.error(503, request)
        self.assertFalse(self.coordinator._active)

    def test_unbound_other_service_and_process_replacement_fail_closed(self):
        other = self.coordinator.lanes['two'].service
        with self.gateway.request('agent-a'):
            self.error(403, lambda: self.gateway.require_bound(other))
            self.frontend.engine.proc = object()
            error = self.error(503, lambda: self.gateway.require_bound(self.frontend))
            self.assertNotIn('secret', str(error))

    def test_admin_secret_and_destination_validated_before_owner_operation(self):
        self.error(403, lambda: self.gateway.consolidate('one', 'one-key'))
        self.error(403, lambda: self.gateway.consolidate('one', None))
        self.error(400, lambda: self.gateway.consolidate('/tmp/checkpoint', 'operator-secret'))
        self.assertEqual(self.coordinator.consolidate_calls, 0)
        self.coordinator.failure = 'OSError /private/customer-weights/file.bin'
        error = self.error(503, lambda: self.gateway.consolidate('one', 'operator-secret'))
        self.assertNotIn('/private', str(error))
        self.assertNotIn('OSError', str(error))

    def test_admin_retries_are_idempotent_and_conflicts_do_not_reenter_owner(self):
        self.assertEqual(self.gateway.consolidate('one', 'operator-secret').state, 'consolidated')
        self.assertEqual(self.gateway.consolidate('one', 'operator-secret').state, 'consolidated')
        self.error(409, lambda: self.gateway.consolidate('two', 'operator-secret'))
        self.assertEqual(self.coordinator.consolidate_calls, 1)
        self.coordinator._blocked = True
        self.error(503, lambda: self.gateway.consolidate('one', 'operator-secret'))
        self.assertEqual(self.coordinator.consolidate_calls, 1)

    def test_inflight_admin_and_external_selection_never_repeat_consolidation(self):
        self.gateway._admin_lock.acquire()
        try:
            error = self.error(503, lambda: self.gateway.consolidate('one', 'operator-secret'))
            self.assertEqual(error.retry_after, 1)
        finally:
            self.gateway._admin_lock.release()
        self.assertEqual(self.coordinator.consolidate_calls, 0)
        self.coordinator._destination = 'one'
        self.error(409, lambda: self.gateway.consolidate('one', 'operator-secret'))
        self.assertEqual(self.coordinator.consolidate_calls, 0)

    def test_duplicate_admin_header_rejected(self):
        headers = Message()
        headers['X-Strata-Migration-Key'] = 'operator-secret'
        headers['X-Strata-Migration-Key'] = 'operator-secret'
        self.error(400, lambda: self.gateway.admin_from_headers(headers))

    def test_validation_is_atomic_and_requires_optin(self):
        coordinator = Coordinator()
        second = coordinator.lanes['two'].service
        second._migration_http_started = True
        with self.assertRaises(ValueError):
            MigrationGateway(coordinator, enabled=True)
        self.assertFalse(hasattr(coordinator.lanes['one'].service, 'migration_gateway'))
        with self.assertRaises(ValueError):
            MigrationGateway(Coordinator())
        for attribute in ('vision', 'images', 'mcp'):
            coordinator = Coordinator()
            setattr(coordinator.lanes['two'].service, attribute, object())
            with self.assertRaises(ValueError):
                MigrationGateway(coordinator, enabled=True)

    def test_disabled_admin_and_safe_bind_policy(self):
        gateway = MigrationGateway(Coordinator(), enabled=True)
        self.error(404, lambda: gateway.consolidate('one', 'operator-secret'))
        self.frontend.api_key = None
        for host in ('127.0.0.1', '::1', 'localhost'):
            self.gateway.validate_bind(self.frontend, host)
        for host in ('0.0.0.0', '::', 'server.example', '192.168.1.1'):
            with self.assertRaises(ValueError):
                self.gateway.validate_bind(self.frontend, host)
        self.frontend.api_key = 'front-key'
        self.gateway.validate_bind(self.frontend, '0.0.0.0')
        with self.assertRaises(ValueError):
            self.gateway.validate_bind(SimpleNamespace(api_key='foreign'), '127.0.0.1')

    def test_serve_only_delegates_after_validation_without_launching_models(self):
        with patch('serve.server.serve', return_value='server') as serve:
            self.assertEqual(self.gateway.serve(self.frontend, port=0), 'server')
            serve.assert_called_once_with(self.frontend, host='127.0.0.1', port=0)
        self.assertEqual(self.coordinator.route_calls, 0)
        self.assertEqual(self.coordinator.consolidate_calls, 0)


if __name__ == '__main__':
    unittest.main()
