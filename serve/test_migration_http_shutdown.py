"""Managed listener teardown must join in-flight handlers; no sockets or GPUs."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serve import server


class ManagedShutdownTests(unittest.TestCase):
    def test_managed_close_waits_for_handler_before_owner_cleanup(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        listener = server.Server.__new__(server.Server)
        listener.socket = Mock()
        listener.serve_forever = Mock()
        listener.shutdown_request = Mock()
        def handler(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test handler release timed out')
        listener.finish_request = handler
        svc = SimpleNamespace(migration_gateway=Mock(), allowed_hosts=[], trusted_origins=[],
                              start_telemetry=Mock(), telemetry=None)
        with patch.object(server, 'Server', return_value=listener), patch.object(server, 'make_handler'), \
                patch.object(server, 'host_names_for', return_value=[]):
            self.assertIs(server.serve(svc), listener)
        self.assertFalse(listener.daemon_threads)
        listener.process_request(object(), ('test', 0))
        self.assertTrue(entered.wait(1))
        def close():
            listener.server_close()
            closed.set()
        closer = threading.Thread(target=close)
        closer.start()
        try:
            self.assertFalse(closed.wait(0.05))
        finally:
            release.set()
            closer.join(2)
        self.assertTrue(closed.is_set())
        listener.shutdown_request.assert_called_once()

    def test_unmanaged_listener_keeps_daemon_default(self):
        listener = server.Server.__new__(server.Server)
        listener.serve_forever = Mock()
        svc = SimpleNamespace(migration_gateway=None, allowed_hosts=[], trusted_origins=[],
                              start_telemetry=Mock())
        with patch.object(server, 'Server', return_value=listener), patch.object(server, 'make_handler'), \
                patch.object(server, 'host_names_for', return_value=[]):
            server.serve(svc)
        self.assertTrue(listener.daemon_threads)
