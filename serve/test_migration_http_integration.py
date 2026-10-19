"""Local HTTP integration for stable agent routing, using CPU-only native doubles."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from serve import server
from serve.backends import ModelIdentity
from serve.checkpoint_store import CheckpointStore
from serve.lane_migration import MigrationCoordinator
from serve.native_lane import NativeLaneWorker
from serve.request_admission import RequestAdmission
from serve.test_lane_migration import MigrationWireEngine


class HttpWireEngine(MigrationWireEngine, server.StrataEngine):
    max_context = 16384

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.hold = False
        self.rendezvous = None
        self.slot_answers = {}
        self.tok = server.ByteTokenizer()
        self.last = {}
        self.info = {}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        slot, receipt = self.local.slot, self.local.receipt
        ticket = receipt.begin(self.gen, self.proc, slot)
        pending = (receipt, ticket, self.gen, slot)
        self.pending.append(pending)
        self.dispatched.append((list(ids), slot))
        with self.slot_cv:
            self.slot_busy[slot] = True
        self.started.set()
        output = []
        try:
            if self.hold:
                if not cancel.wait(4):
                    raise RuntimeError('test client disconnect did not reach native cancel')
                self.cancelled.set()
                return
            if self.rendezvous is not None:
                self.rendezvous.wait(timeout=3)
            for token in (self.tok.encode(self.slot_answers.get(slot, 'routed answer')) + self.tok.encode('<|im_end|>', parse_special=True))[:max_new]:
                if cancel.is_set():
                    self.cancelled.set()
                    return
                output.append(token)
                yield token
        finally:
            self.states[slot] = list(ids) + output[:-1]
            self.slot_held[slot] = list(self.states[slot])
            self.epoch += 1
            self.last = dict(prompt_tokens=len(ids), generated=len(output), reused=0,
                             prompt_ms=1.0, decode_ms=1.0,
                             finish='cancel' if cancel.is_set() else 'stop')
            self.acknowledge(pending)
            self.finished.set()


class MigrationHttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.store = CheckpointStore(self.root)
        self.model = ModelIdentity('qwen4exp', 'weights', 'tokenizer')
        self.events, self.lanes, self.workers, agents = [], {}, {}, {}
        for index in range(3):
            name = f'lane-{index}'
            config = dict(host_capacity_bytes=1000000, gpu_capacity_bytes=1000000,
                          gpu_id='0', max_requests=2, fixed_host_bytes=10, fixed_gpu_bytes=10,
                          host_request_bytes=1, gpu_request_bytes=1, host_bytes_per_token=1,
                          gpu_bytes_per_token=1, max_context_tokens=16384, lookahead_tokens=0)
            engine = HttpWireEngine(self.model, threading.Lock(),
                {0: [10 + index, 20], 1: [30 + index, 40]}, name, self.events)
            lane = server.Service(engine, server.ByteTokenizer(),
                server.ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
            engine.fifo = lane.fifo
            lane.request_admission = RequestAdmission(config)
            self.lanes[name] = lane
            self.workers[name] = NativeLaneWorker(lane, self.model, enabled=True)
            for slot in range(2):
                agents[f'agent-{2 * index + slot}'] = (name, slot)
        self.coordinator = MigrationCoordinator(self.workers, self.store, agents, enabled=True)
        self.service = self.lanes['lane-2']
        self.service.api_key = 'client-secret'
        from serve.migration_http import MigrationGateway
        self.gateway = MigrationGateway(self.coordinator, enabled=True, admin_token='admin-secret')
        self.service.migration_gateway = self.gateway
        self.start_http()

    def start_http(self):
        self.http = server.serve(self.service, port=0)
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.base = f'http://127.0.0.1:{self.http.server_address[1]}'

    def post(self, path, body, headers=None):
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer client-secret',
                     **(headers or {})})
        try:
            response = urllib.request.urlopen(request, timeout=8)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status, raw = response.status, response.read().decode()
        if status == 200 and (headers or {}).get('X-Strata-Agent-ID'):
            self.wait_idle()
        return status, raw

    def wait_idle(self):
        # HTTP completion can precede handler finally/accounting. The route is
        # deliberately owned until that cleanup, so synchronize with its condition.
        with self.coordinator._cv:
            self.assertTrue(self.coordinator._cv.wait_for(lambda: not self.coordinator._active, timeout=2),
                            'HTTP handler did not release its agent route')
        for lane in self.lanes.values():
            snapshot = lane.request_admission.ledger.snapshot()
            if snapshot['workers']:
                self.assertTrue(lane.request_admission.wait_drained(2),
                                'HTTP request reservation did not drain')
            else:
                self.assertEqual(snapshot['active_reservations'], 0)

    @staticmethod
    def body(stream=False):
        return {'model': 'm', 'messages': [{'role': 'user', 'content': 'hello'}],
                'max_tokens': 32, 'reasoning_effort': 'none', 'thinking': {'type': 'disabled'}, 'stream': stream}

    def test_openai_and_anthropic_stream_and_json_use_stable_agent_slot(self):
        for path in ('/v1/chat/completions', '/v1/messages'):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    status, raw = self.post(path, self.body(stream), {'X-Strata-Agent-ID': 'agent-1'})
                    self.assertEqual(status, 200, raw)
                    if stream:
                        self.assertIn('data:', raw)
                        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
                        if path.endswith('completions'):
                            content = ''.join(c.get('choices', [{}])[0].get('delta', {}).get('content', '') for c in chunks if c.get('choices'))
                        else:
                            content = ''.join(c.get('delta', {}).get('text', '') for c in chunks)
                        self.assertEqual(content, 'routed answer')
                    else:
                        data = json.loads(raw)
                        content = (data['choices'][0]['message']['content'] if path.endswith('completions')
                                   else ''.join(c.get('text', '') for c in data['content']))
                        self.assertEqual(content, 'routed answer')
                    self.assertEqual(self.lanes['lane-0'].engine.dispatched[-1][1], 1)
                    self.assertEqual(self.coordinator._active, set())
                    self.assertEqual(self.lanes['lane-0'].request_admission.ledger.snapshot()['active_reservations'], 0)
        self.assertEqual(self.lanes['lane-2'].engine.dispatched, [])

    def test_auth_and_agent_validation_do_not_dispatch(self):
        for headers, expected in (
            ({'Authorization': 'Bearer wrong', 'X-Strata-Agent-ID': 'agent-0'}, 401),
            ({}, 400),
            ({'X-Strata-Agent-ID': 'not-enrolled'}, 404),
        ):
            status, raw = self.post('/v1/chat/completions', self.body(), headers)
            self.assertEqual(status, expected, raw)
        self.assertFalse(any(lane.engine.dispatched for lane in self.lanes.values()))
        self.assertEqual(self.events, [], 'auth and enrollment failures must precede native commands')

    def test_operator_consolidation_then_parked_agent_http_restore(self):
        status, raw = self.post('/v1/migration/consolidate', {'destination': 'lane-2'},
                               {'X-Strata-Migration-Key': 'wrong'})
        self.assertEqual(status, 403, raw)
        self.assertFalse(any(lane.engine.close_count for lane in self.lanes.values()))
        status, raw = self.post('/v1/migration/consolidate', {'destination': 'lane-2'},
                               {'X-Strata-Migration-Key': 'admin-secret'})
        self.assertEqual(status, 200, raw)
        parked = next(a for a, route in self.coordinator.routes.items() if route.lane is None)
        status, raw = self.post('/v1/messages', self.body(), {'X-Strata-Agent-ID': parked})
        self.assertEqual(status, 200, raw)
        route = self.coordinator.routes[parked]
        self.assertEqual(route.lane, 'lane-2')
        self.assertEqual(self.lanes['lane-2'].engine.dispatched[-1][1], route.slot)
        self.assertTrue(any(command == 'RESTORE_SLOT' for _, command in self.events))
        self.assertEqual(self.coordinator._active, set())

    def test_disconnected_stream_cancels_and_releases_route_and_reservation(self):
        engine = self.lanes['lane-0'].engine
        engine.hold = True
        data = json.dumps(self.body(True)).encode()
        sock = socket.create_connection(('127.0.0.1', self.http.server_address[1]), timeout=5)
        try:
            head = ('POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n'
                    'Content-Type: application/json\r\nAuthorization: Bearer client-secret\r\n'
                    'X-Strata-Agent-ID: agent-0\r\n'
                    f'Content-Length: {len(data)}\r\n\r\n')
            sock.sendall(head.encode() + data)
            self.assertTrue(engine.started.wait(2))
        finally:
            sock.close()
        self.assertTrue(engine.cancelled.wait(2), 'HTTP disconnect must reach native cancellation')
        self.assertTrue(engine.finished.wait(2))
        self.wait_idle()
        self.assertEqual(self.coordinator._active, set())
        self.assertEqual(self.lanes['lane-0'].request_admission.ledger.snapshot()['active_reservations'], 0)
        engine.hold = False
        status, raw = self.post('/v1/chat/completions', self.body(), {'X-Strata-Agent-ID': 'agent-0'})
        self.assertEqual(status, 200, raw)

    def test_admin_retry_is_idempotent_and_different_destination_preserves_ready_lane(self):
        headers = {'X-Strata-Migration-Key': 'admin-secret'}
        for _ in range(2):
            status, raw = self.post('/v1/migration/consolidate', {'destination': 'lane-2'}, headers)
            self.assertEqual(status, 200, raw)
            self.assertFalse(self.lanes['lane-2'].native_lane_gated)
        self.assertEqual([self.lanes[f'lane-{i}'].engine.close_count for i in range(3)], [1, 1, 0])
        before = list(self.events)
        status, raw = self.post('/v1/migration/consolidate', {'destination': 'lane-1'}, headers)
        self.assertEqual(status, 409, raw)
        self.assertEqual(self.events, before)
        self.assertFalse(self.lanes['lane-2'].native_lane_gated)
        status, raw = self.post('/v1/chat/completions', self.body(), {'X-Strata-Agent-ID': 'agent-4'})
        self.assertEqual(status, 200, raw)

    def test_missing_operator_key_does_not_gate_or_touch_native_lanes(self):
        status, raw = self.post('/v1/migration/consolidate', {'destination': 'lane-2'})
        self.assertEqual(status, 403, raw)
        self.assertEqual(self.events, [])
        for lane in self.lanes.values():
            self.assertFalse(getattr(lane, 'native_lane_gated', False))
            lease = lane.request_admission.acquire(2, 1)
            lane.request_admission.ledger.release(lease, iterator_closed=True, work_drained=True)

    def test_managed_control_endpoints_are_rejected_without_native_mutation(self):
        for path in ('/settings', '/config', '/load', '/unload', '/v1/load', '/v1/unload',
                     '/v1/vram', '/slots/0?action=restore', '/v1/images/generations'):
            with self.subTest(path=path):
                status, raw = self.post(path, {})
                self.assertEqual(status, 409, raw)
        self.assertEqual(self.events, [])

    def test_duplicate_agent_header_rejected_before_dispatch(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.http.server_address[1], timeout=5)
        self.addCleanup(connection.close)
        data = json.dumps(self.body()).encode()
        connection.putrequest('POST', '/v1/chat/completions')
        connection.putheader('Authorization', 'Bearer client-secret')
        connection.putheader('Content-Type', 'application/json')
        connection.putheader('Content-Length', str(len(data)))
        connection.putheader('X-Strata-Agent-ID', 'agent-0')
        connection.putheader('X-Strata-Agent-ID', 'agent-1')
        connection.endheaders(data)
        response = connection.getresponse()
        self.assertEqual(response.status, 400, response.read().decode())
        self.assertEqual(self.events, [])

    def test_responses_api_json_and_sse_use_the_enrolled_slot(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, raw = self.post('/v1/responses',
                    {'model': 'm', 'input': 'hello', 'max_output_tokens': 32,
                     'reasoning': {'effort': 'none'}, 'stream': stream},
                    {'X-Strata-Agent-ID': 'agent-3'})
                self.assertEqual(status, 200, raw)
                if stream:
                    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
                    content = ''.join(e['delta'] for e in events if e['type'] == 'response.output_text.delta')
                else:
                    data = json.loads(raw)
                    content = ''.join(c.get('text', '') for item in data['output']
                                      if item['type'] == 'message' for c in item['content'])
                self.assertEqual(content, 'routed answer')
                self.assertEqual(self.lanes['lane-1'].engine.dispatched[-1][1], 1)
        self.assertEqual(self.coordinator._active, set())

    def test_concurrent_different_agent_streams_remain_isolated(self):
        engine = self.lanes['lane-0'].engine
        engine.rendezvous = threading.Barrier(2)
        engine.slot_answers = {0: 'first agent', 1: 'second agent'}
        def request(agent):
            return self.post('/v1/chat/completions', self.body(True), {'X-Strata-Agent-ID': agent})
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(request, f'agent-{i}') for i in range(2)]
            for i, job in enumerate(jobs):
                status, raw = job.result(timeout=6)
                self.assertEqual(status, 200, raw)
                chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
                content = ''.join(c['choices'][0].get('delta', {}).get('content', '')
                                  for c in chunks if c.get('choices'))
                self.assertEqual(content, engine.slot_answers[i])
        self.assertEqual({slot for _, slot in engine.dispatched}, {0, 1})
        self.assertEqual(engine.pending, [])
        self.assertEqual(self.coordinator._active, set())
        self.assertEqual(self.lanes['lane-0'].request_admission.ledger.snapshot()['active_reservations'], 0)

    def test_unmanaged_http_service_keeps_default_behavior(self):
        # The ordinary server path remains usable without enrollment or agent headers.
        tok = server.ByteTokenizer()
        plain = server.Service(server.MockEngine(tok, 'ordinary answer'), tok,
                               server.ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        http = server.serve(plain, port=0)
        try:
            request = urllib.request.Request(f'http://127.0.0.1:{http.server_address[1]}/v1/chat/completions',
                data=json.dumps(self.body()).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(json.load(response)['choices'][0]['message']['content'], 'ordinary answer')
        finally:
            http.shutdown()
            http.server_close()


if __name__ == '__main__':
    unittest.main()
