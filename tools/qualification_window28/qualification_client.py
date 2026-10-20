"""Bounded HTTP/native continuation probes for the pinned migration owner.

This is a library for the sole isolated runner, not a server launcher. It never
stores credentials in a result and has no model/GPU dependencies of its own.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import http.client
import json
import socket
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit


ENDPOINTS = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
CONSOLIDATION_SOCKET_TIMEOUT_S = 300.0


@dataclass
class Probe:
    name: str
    status: str
    started_at: float
    completed_at: float
    http_status: int | None = None
    detail: dict | None = None


class JsonlResults:
    """Append-only, bounded JSONL; callers decide prompt/output redaction."""
    def __init__(self, path):
        self.path = str(path)

    def write(self, probe: Probe):
        from pathlib import Path
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(probe)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    def event(self, event: str, **fields):
        row = {"name": "owner-event:" + event, "at_wall": time.time(),
               "at_monotonic": time.monotonic(), **fields}
        from pathlib import Path
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


class HttpProbeClient:
    def __init__(self, base_url: str, client_api_key: str, *, timeout_s: float = 120.0,
                 results: JsonlResults | None = None):
        u = urlsplit(base_url)
        if u.scheme != "http" or u.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("qualification client accepts only loopback HTTP")
        if u.path not in ("", "/") or u.query or u.fragment:
            raise ValueError("base_url must be an origin without path/query")
        if not client_api_key or timeout_s <= 0:
            raise ValueError("client API key and positive timeout are required")
        self.host, self.port = u.hostname, u.port or 80
        self.client_api_key, self.timeout_s, self.results = client_api_key, timeout_s, results

    @staticmethod
    def _payload(path: str, model: str, messages: list[dict], *, stream: bool,
                 max_new: int, temperature: float = 0.0) -> dict:
        if path == "/v1/chat/completions":
            return {"model": model, "messages": messages, "max_tokens": max_new,
                    "temperature": temperature, "stream": stream,
                    "chat_template_kwargs": {"enable_thinking": False}}
        if path == "/v1/messages":
            # Anthropic endpoint expects max_tokens; temperature is accepted by
            # this server's compatibility layer and greedy 0 is explicit. These
            # migration probes measure lifecycle/routing, so bounded responses
            # must not spend their whole output cap in hidden reasoning.
            return {"model": model, "messages": messages, "max_tokens": max_new,
                    "temperature": temperature, "stream": stream,
                    "thinking": {"type": "disabled"}}
        if path == "/v1/responses":
            return {"model": model, "input": messages, "max_output_tokens": max_new,
                    "temperature": temperature, "stream": stream,
                    "reasoning": {"effort": "none"}}
        raise ValueError("unsupported endpoint")

    @staticmethod
    def _text(path: str, body: dict) -> str:
        if path == "/v1/chat/completions":
            return body["choices"][0]["message"].get("content") or ""
        if path == "/v1/messages":
            return "".join(x.get("text", "") for x in body.get("content", []) if x.get("type") == "text")
        return "".join(part.get("text", "") for item in body.get("output", [])
                       if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text")

    @staticmethod
    def _response_metadata(path: str, body: dict) -> dict:
        """Capture completion status without persisting any private reasoning text."""
        if path == "/v1/chat/completions":
            choices = body.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            reasoning = message.get("reasoning_content") or ""
            return {"finish_reason": choice.get("finish_reason"),
                    "reasoning_content_chars": len(reasoning),
                    "reasoning_content_sha256": hashlib.sha256(reasoning.encode()).hexdigest() if reasoning else None}
        if path == "/v1/messages":
            return {"stop_reason": body.get("stop_reason"), "stop_sequence": body.get("stop_sequence")}
        return {"response_status": body.get("status"),
                "incomplete_reason": (body.get("incomplete_details") or {}).get("reason")
                if isinstance(body.get("incomplete_details"), dict) else None}

    @staticmethod
    def _sse_text(path: str, data: str) -> str:
        if data.strip() == "[DONE]":
            return ""
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return ""
        if path == "/v1/chat/completions":
            return "".join(choice.get("delta", {}).get("content") or "" for choice in event.get("choices", []))
        if path == "/v1/messages":
            if event.get("type") == "content_block_delta":
                return (event.get("delta") or {}).get("text", "")
            return ""
        if event.get("type") == "response.output_text.delta":
            return event.get("delta", "")
        return ""

    def _call(self, path: str, payload: dict, agent_id: str | None = None,
              extra_headers: Iterable[tuple[str, str]] = (),
              socket_timeout_s: float | None = None) -> tuple[int, dict[str, str], bytes]:
        timeout = self.timeout_s if socket_timeout_s is None else socket_timeout_s
        if (socket_timeout_s is not None and
                (type(socket_timeout_s) not in (int, float) or not 0 < socket_timeout_s <= 300.0)):
            raise ValueError("per-request socket timeout must be in (0, 300] seconds")
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        extra_headers = list(extra_headers)
        supplied = [(k.lower(), v) for k, v in extra_headers]
        if not any(k == "authorization" for k, _ in supplied):
            headers["Authorization"] = "Bearer " + self.client_api_key
        if agent_id is not None and not any(k == "x-strata-agent-id" for k, _ in supplied):
            headers["X-Strata-Agent-ID"] = agent_id
        # http.client requires a mapping here; a sequence of pairs raises
        # AttributeError while it tries to lowercase each header key.
        for key, value in extra_headers:
            for existing in list(headers):
                if existing.lower() == key.lower():
                    del headers[existing]
            headers[key] = value
        try:
            conn.request("POST", path, body=json.dumps(payload).encode("utf-8"), headers=headers)
            response = conn.getresponse()
            body = response.read()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, body
        finally:
            conn.close()

    @staticmethod
    def _duplicate_agent_wire(path: str, payload: dict, client_api_key: str, agent_id: str) -> bytes:
        """Build one deliberate repeated-header request for the negative test.

        ``HTTPConnection.request`` accepts a mapping and therefore cannot
        represent duplicate field names. Keep this one hostile case explicit
        on the wire; ordinary requests continue to use the standard client.
        """
        body = json.dumps(payload).encode("utf-8")
        head = (f"POST {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1\r\n"
                "Content-Type: application/json\r\n"
                f"Authorization: Bearer {client_api_key}\r\n"
                f"X-Strata-Agent-ID: {agent_id}\r\n"
                f"X-Strata-Agent-ID: {agent_id}\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n").encode("ascii")
        return head + body

    def request(self, path: str, *, model: str, messages: list[dict], agent_id: str,
                stream: bool = False, max_new: int = 32, temperature: float = 0.0,
                expect_status: int = 200, extra_headers: Iterable[tuple[str, str]] = ()) -> Probe:
        start = time.time()
        payload = self._payload(path, model, messages, stream=stream, max_new=max_new,
                                temperature=temperature)
        status, headers, body = self._call(path, payload, agent_id, extra_headers)
        detail: dict = {"stream": stream, "content_type": headers.get("content-type", "")}
        text = ""
        if status == 200:
            if stream:
                for raw in body.decode("utf-8", "replace").splitlines():
                    if raw.startswith("data:"):
                        text += self._sse_text(path, raw[5:].strip())
            else:
                decoded = json.loads(body)
                text = self._text(path, decoded)
            detail.update({"nonempty_text": bool(text.strip()), "output_chars": len(text),
                           "output_text": text})
            if not stream and status == 200:
                detail.update(self._response_metadata(path, decoded))
            result = "PASS" if status == expect_status and text.strip() else "FAIL"
        else:
            detail["error_body"] = body.decode("utf-8", "replace")[:500]
            result = "PASS" if status == expect_status else "FAIL"
        p = Probe(f"http:{path}:{'sse' if stream else 'json'}", result, start, time.time(), status, detail)
        if self.results:
            self.results.write(p)
        return p

    def endpoint_matrix(self, *, model: str, messages_by_agent: Mapping[str, list[dict]],
                        max_new: int = 32) -> list[Probe]:
        """Run JSON and SSE for every endpoint and enrolled agent, sequentially bounded."""
        if max_new not in (1, 4, 8, 16, 32):
            raise ValueError("bounded endpoint probes allow max_new in 1,4,8,16,32")
        histories = [json.dumps(messages, sort_keys=True, ensure_ascii=False) for messages in messages_by_agent.values()]
        if len(histories) != len(set(histories)):
            raise ValueError("each agent must use a distinct test fact/history")
        rows = []
        for agent, messages in messages_by_agent.items():
            for path in ENDPOINTS:
                for stream in (False, True):
                    rows.append(self.request(path, model=model, messages=messages, agent_id=agent,
                                             stream=stream, max_new=max_new))
        return rows

    def negative_routing(self, *, model: str, messages: list[dict], known_agent: str,
                         unknown_agent: str, state_snapshot: Callable[[], Any]) -> list[Probe]:
        """Check invalid auth/IDs leave caller-defined trusted state unchanged.

        Caller must provide a correctly locked, trusted native+admission
        snapshot; this method never guesses at live process inventory.
        """
        payload = self._payload(ENDPOINTS[0], model, messages, stream=False, max_new=1)
        cases = [
            ("wrong_auth", [("Authorization", "Bearer invalid-qualification-key"),
                             ("X-Strata-Agent-ID", known_agent)], 401),
            ("missing_agent", [("Authorization", "Bearer " + self.client_api_key)], 400),
            ("malformed_agent", [("Authorization", "Bearer " + self.client_api_key),
                                  ("X-Strata-Agent-ID", "bad id!")], 400),
            ("unknown_agent", [("Authorization", "Bearer " + self.client_api_key),
                                ("X-Strata-Agent-ID", unknown_agent)], 404),
        ]
        out = []
        for name, headers, expected in cases:
            before = state_snapshot()
            started = time.time()
            status, _, body = self._call(ENDPOINTS[0], payload, extra_headers=headers)
            after = state_snapshot()
            unchanged = before == after
            passed = status == expected and unchanged
            p = Probe("negative:" + name, "PASS" if passed else "FAIL", started, time.time(), status,
                      {"expected_status": expected, "state_unchanged": unchanged,
                       "error_body": body.decode("utf-8", "replace")[:300]})
            out.append(p)
            if self.results:
                self.results.write(p)
        # HTTPConnection's headers mapping cannot express duplicate field
        # names. Send this one case as explicit raw HTTP so the server's
        # get_all() duplicate-header guard is actually exercised.
        before = state_snapshot()
        started = time.time()
        wire = self._duplicate_agent_wire(ENDPOINTS[0], payload, self.client_api_key, known_agent)
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        try:
            sock.settimeout(self.timeout_s)
            sock.sendall(wire)
            response = http.client.HTTPResponse(sock)
            response.begin()
            status, body = response.status, response.read()
        finally:
            sock.close()
        unchanged = before == state_snapshot()
        p = Probe("negative:duplicate_agent", "PASS" if status == 400 and unchanged else "FAIL",
                  started, time.time(), status,
                  {"expected_status": 400, "state_unchanged": unchanged,
                   "body": body.decode("utf-8", "replace")[:300]})
        out.append(p)
        if self.results:
            self.results.write(p)
        return out

    def managed_control_negatives(self, *, state_snapshot: Callable[[], Any]) -> list[Probe]:
        """Verify gateway-owned service controls reject client mutation."""
        cases = (("load", "/v1/load"), ("unload", "/v1/unload"),
                 ("vram", "/v1/vram"), ("config", "/config"),
                 ("slot_file", "/slots/0?action=save"))
        rows = []
        for name, path in cases:
            before = state_snapshot()
            start = time.time()
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
            try:
                conn.request("POST", path, body=b"{}", headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.client_api_key})
                response = conn.getresponse()
                body = response.read()
                status = response.status
            finally:
                conn.close()
            unchanged = before == state_snapshot()
            p = Probe("negative:managed-control:" + name,
                      "PASS" if status == 409 and unchanged else "FAIL",
                      start, time.time(), status,
                      {"expected_status": 409, "state_unchanged": unchanged,
                       "body": body.decode("utf-8", "replace")[:250]})
            rows.append(p)
            if self.results:
                self.results.write(p)
        return rows

    def consolidate(self, *, destination: str, operator_token: str,
                    timeout_s: float = 30.0) -> Probe:
        """Post the isolated owner operation; never prints either credential."""
        payload = {"destination": destination, "timeout_s": timeout_s}
        start = time.time()
        status, headers, body = self._call("/v1/migration/consolidate", payload,
                                          extra_headers=[("X-Strata-Migration-Key", operator_token)],
                                          socket_timeout_s=CONSOLIDATION_SOCKET_TIMEOUT_S)
        detail = {"destination": destination, "retry_after": headers.get("retry-after"),
                  "body": body.decode("utf-8", "replace")[:500]}
        p = Probe("migration:consolidate", "PASS" if status == 200 else "FAIL",
                  start, time.time(), status, detail)
        if self.results:
            self.results.write(p)
        return p

    def rotation_rounds(self, *, model: str, messages_by_agent: Mapping[str, list[dict]],
                        resident_pair: Callable[[], tuple[str, str]],
                        next_user: Callable[[str, str], str],
                        rounds: int = 3, max_new: int = 8) -> list[Probe]:
        """Three rounds over all six agents, preserving each full transcript.

        ``next_user(agent, label)`` returns a distinct next user turn. Both
        ordinary rotation turns and the concurrent resident probes append each
        assistant reply and a new user turn before the agent is sent again.
        """
        if rounds != 3 or max_new not in (1, 4, 8):
            raise ValueError("qualification uses exactly three bounded rotation rounds (1/4/8 output tokens)")
        if not callable(next_user):
            raise ValueError("rotation requires next_user to extend each complete conversation")
        histories = {a: [dict(m) for m in rows] for a, rows in messages_by_agent.items()}
        if any(not rows or rows[-1].get("role") != "user" for rows in histories.values()):
            raise ValueError("each agent history must end with its next user turn")
        probes = []
        for round_no in range(1, rounds + 1):
            for agent, messages in messages_by_agent.items():
                p = self.request(ENDPOINTS[0], model=model, messages=histories[agent],
                                 agent_id=agent, stream=(round_no % 2 == 0), max_new=max_new)
                p.name = f"rotation:r{round_no}:{agent}"
                if self.results:
                    self.results.write(p)
                probes.append(p)
                if p.status == "PASS":
                    histories[agent].append({"role": "assistant", "content": p.detail["output_text"]})
                    histories[agent].append({"role": "user", "content": next_user(agent, p.name)})
                # After every routed request (which may have caused rotation),
                # exercise both current resident agents concurrently. The
                # caller derives the pair from the coordinator's current routes.
                pair = resident_pair()
                if len(set(pair)) != 2 or any(x not in messages_by_agent for x in pair):
                    raise RuntimeError("resident_pair must return two distinct enrolled agents")
                concurrent = self.concurrent_pair(model=model, messages_by_agent=histories,
                                                  agents=pair, max_new=4)
                probes.extend(concurrent)
                # concurrent_pair's result order matches the supplied pair.
                for agent, item in zip(pair, concurrent):
                    if item.status == "PASS":
                        histories[agent].append({"role": "assistant", "content": item.detail["output_text"]})
                        histories[agent].append({"role": "user", "content": next_user(agent, f"rotation:r{round_no}:concurrent")})
        return probes

    def concurrent_pair(self, *, model: str, messages_by_agent: Mapping[str, list[dict]],
                        agents: tuple[str, str], max_new: int = 8) -> list[Probe]:
        if len(set(agents)) != 2 or any(a not in messages_by_agent for a in agents):
            raise ValueError("choose two distinct enrolled agents")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.request, ENDPOINTS[0], model=model,
                                   messages=messages_by_agent[a], agent_id=a,
                                   stream=True, max_new=max_new) for a in agents]
            return [f.result() for f in futures]

    def busy_consolidation_then_retry(self, *, model: str,
                                      messages_by_agent: Mapping[str, list[dict]],
                                      agents: tuple[str, str], destination: str,
                                      operator_token: str,
                                      active_turns_ready: Callable[[], bool],
                                      active_wait_s: float = 15.0) -> list[Probe]:
        """Prove a short consolidation wait while two owned HTTP turns are active.

        The trusted callback must observe owner route/receipt state, not infer
        activity from a submitted future. Once both complete, retry once.
        """
        if len(set(agents)) != 2 or any(a not in messages_by_agent for a in agents):
            raise ValueError("busy test requires two distinct enrolled agents")
        rows = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.request, ENDPOINTS[0], model=model,
                                   messages=messages_by_agent[a], agent_id=a,
                                   stream=True, max_new=32) for a in agents]
            deadline = time.monotonic() + active_wait_s
            while time.monotonic() < deadline and not active_turns_ready():
                if any(f.done() and f.exception() for f in futures):
                    break
                time.sleep(0.05)
            active = active_turns_ready()
            if active:
                start = time.time()
                status, headers, body = self._call(
                    "/v1/migration/consolidate", {"destination": destination, "timeout_s": 0.01},
                    extra_headers=[("Authorization", "Bearer " + self.client_api_key),
                                   ("X-Strata-Migration-Key", operator_token)])
                p = Probe("migration:busy-active-turns", "PASS" if status == 503 else "FAIL",
                          start, time.time(), status,
                          {"expected_status": 503, "retry_after": headers.get("retry-after"),
                           "body": body.decode("utf-8", "replace")[:300]})
            else:
                p = Probe("migration:busy-active-turns", "BLOCKED", time.time(), time.time(),
                          None, {"reason": "owner callback did not confirm both native requests active"})
            rows.append(p)
            if self.results:
                self.results.write(p)
            request_rows = [f.result() for f in futures]
            rows.extend(request_rows)
        if rows[0].status == "PASS":
            rows.append(self.consolidate(destination=destination, operator_token=operator_token,
                                         timeout_s=30.0))
        return rows

    def cancel_stream_after_event(self, *, model: str, messages: list[dict], agent_id: str,
                                  max_new: int = 32, drain_check: Callable[[], dict] | None = None,
                                  drain_timeout_s: float = 30.0) -> Probe:
        """Disconnect after first SSE data event, then capture caller's drain proof.

        Drain callback must report exact native request receipt and admission
        status; absent/ambiguous evidence is BLOCKED, never inferred from close.
        """
        if max_new not in (8, 16, 32):
            raise ValueError("cancel case requires bounded max_new 8,16,32")
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        payload = self._payload(ENDPOINTS[0], model, messages, stream=True, max_new=max_new)
        headers = [("Content-Type", "application/json"),
                   ("Authorization", "Bearer " + self.client_api_key),
                   ("X-Strata-Agent-ID", agent_id)]
        start = time.time()
        first_data = False
        status = None
        try:
            conn.request("POST", ENDPOINTS[0], body=json.dumps(payload).encode(), headers=headers)
            response = conn.getresponse()
            status = response.status
            while True:
                line = response.fp.readline()
                if not line:
                    break
                if line.startswith(b"data:") and line[5:].strip() not in (b"", b"[DONE]"):
                    first_data = True
                    break
            conn.close()  # client disconnect is deliberately the cancellation trigger
        except (TimeoutError, socket.timeout):
            conn.close()
        if drain_timeout_s <= 0 or drain_timeout_s > 120:
            raise ValueError("drain wait must be in (0,120] seconds")
        drain = None
        if drain_check is not None:
            deadline = time.monotonic() + drain_timeout_s
            while True:
                drain = drain_check()
                if (isinstance(drain, dict) and drain.get("receipt_confirmed") is True
                        and drain.get("admission_drained") is True):
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
        confirmed = isinstance(drain, dict) and drain.get("receipt_confirmed") is True and drain.get("admission_drained") is True
        outcome = "PASS" if status == 200 and first_data and confirmed else "BLOCKED"
        p = Probe("cancel:sse-disconnect", outcome, start, time.time(), status,
                  {"first_data_event": first_data, "drain_evidence": drain,
                   "transparent_resume_claim": False})
        if self.results:
            self.results.write(p)
        return p

    def migration_negative_controls(self, *, operator_token: str,
                                    valid_destination: str, other_destination: str,
                                    state_snapshot: Callable[[], Any]) -> list[Probe]:
        """Before migration, reject wrong operator key and unknown destination."""
        cases = (("wrong_operator_key", {"destination": valid_destination, "timeout_s": 30},
                  [("Authorization", "Bearer " + self.client_api_key),
                   ("X-Strata-Migration-Key", "invalid-operator-key")], 403),
                 ("unknown_destination", {"destination": "missing-lane", "timeout_s": 30},
                  [("Authorization", "Bearer " + self.client_api_key),
                   ("X-Strata-Migration-Key", operator_token)], 400))
        rows = []
        for name, payload, headers, expected in cases:
            before = state_snapshot()
            start = time.time()
            status, _, body = self._call("/v1/migration/consolidate", payload, extra_headers=headers)
            unchanged = before == state_snapshot()
            p = Probe("negative:" + name, "PASS" if status == expected and unchanged else "FAIL",
                      start, time.time(), status,
                      {"expected_status": expected, "state_unchanged": unchanged,
                       "body": body.decode("utf-8", "replace")[:300]})
            rows.append(p)
            if self.results:
                self.results.write(p)
        return rows

    def migration_retry_controls(self, *, destination: str, operator_token: str,
                                 other_destination: str,
                                 state_snapshot: Callable[[], Any]) -> list[Probe]:
        """After success, same-destination retry is 200 and another destination 409."""
        rows = []
        for name, dest, expected in (("same_destination_retry", destination, 200),
                                     ("different_destination_retry", other_destination, 409)):
            before = state_snapshot()
            start = time.time()
            status, _, body = self._call("/v1/migration/consolidate",
                                        {"destination": dest, "timeout_s": 30},
                                        extra_headers=[("Authorization", "Bearer " + self.client_api_key),
                                                       ("X-Strata-Migration-Key", operator_token)])
            unchanged = before == state_snapshot()
            p = Probe("migration:" + name, "PASS" if status == expected and unchanged else "FAIL",
                      start, time.time(), status,
                      {"expected_status": expected, "state_unchanged": unchanged,
                       "body": body.decode("utf-8", "replace")[:300]})
            rows.append(p)
            if self.results:
                self.results.write(p)
        return rows


def collect_token_ids(coordinator: Any, lane_workers: Mapping[str, Any], *, agent: str,
                      prompt_ids: list[int], max_new: int = 32,
                      sampling: dict | None = None, timeout_s: float = 30.0) -> dict:
    """Run one greedy native-owner branch and capture token IDs plus cache evidence."""
    if max_new not in (1, 4, 8, 16, 32):
        raise ValueError("native token-ID probe is bounded to max_new 1,4,8,16,32")
    from threading import Event
    route = coordinator.routes.get(agent)
    if route is None or route.lane is None:
        raise ValueError("agent must be active before direct token-ID collection")
    worker = lane_workers[route.lane]
    def held_evidence():
        # Capture the Python-side owner prefix under the batch slot lock. This
        # is diagnostic only; the native `reused` counter remains authoritative.
        with worker.engine.slot_cv:
            held = list(worker.engine.slot_held[route.slot])
        lcp = 0
        for left, right in zip(held, prompt_ids):
            if left != right:
                break
            lcp += 1
        return {"held_count": len(held), "prompt_count": len(prompt_ids),
                "sha256": hashlib.sha256(json.dumps(held, separators=(",", ":")).encode()).hexdigest(),
                "prompt_lcp": lcp,
                "held_is_strict_prompt_prefix": len(held) < len(prompt_ids) and lcp == len(held),
                "prompt_is_prefix_of_held": len(prompt_ids) <= len(held) and lcp == len(prompt_ids)}

    held_before = held_evidence()
    before_generation = getattr(worker.engine, "gen", None)
    # The pinned Service/StrataEngine iterator yields ``None`` as a progress
    # heartbeat while a long prompt is being read. Those are transport events,
    # not token IDs, and their count can differ between identical continuations.
    yielded = list(coordinator.generate(agent, list(prompt_ids), max_new,
                                        sampling or {"temperature": 0.0}, Event(), timeout_s=timeout_s))
    ids = [token for token in yielded if token is not None]
    last = dict(getattr(worker.engine, "last", {}) or {})
    held_after = held_evidence()
    info = getattr(worker.engine, "info", {}) or {}
    native_cache_inputs = {"info_slot_cache": info.get("slot_cache"),
                           "info_session_export_batch": info.get("session_export_batch"),
                           "info_session_restore_batch": info.get("session_restore_batch"),
                           "python_slot_held_present": isinstance(getattr(worker.engine, "slot_held", None), list),
                           "python_slot_cache_present": isinstance(getattr(worker.engine, "slot_cache", None), list)}
    return {"agent": agent, "lane": route.lane, "slot": route.slot,
            "process_generation": before_generation, "prompt_token_count": len(prompt_ids),
            "output_token_ids": ids, "heartbeat_count": sum(token is None for token in yielded),
            "yield_event_count": len(yielded), "reused": last.get("reused"),
            "prompt_read": last.get("prompt_read"), "prompt_tokens": last.get("prompt_tokens"),
            "finish": last.get("finish"), "generated": last.get("generated"),
            "held_prefix_before": held_before, "held_prefix_after": held_after,
            "native_cache_inputs": native_cache_inputs}


def seed_owner_prefix_fixture(harness: Any, *, agent: str, prompt_ids: list[int],
                              tail_tokens: int, persist: Callable[[dict], None] | None = None,
                              timeout_s: float = 30.0,
                              require_empty_slot: bool = True) -> dict:
    """Seed C=P[:-k] through the native owner API and verify the exact slot.

    ``prompt_ids`` must be the complete, already-rendered normal HTTP prompt P
    from the deployed Service tokenizer/template. This function never decodes
    native tokens or reconstructs an assistant turn. Its one generated token is
    consumed only to complete the native operation and is intentionally omitted
    from the fixture transcript.

    ``persist`` is called before native generation and again with observed
    state before any assertion, so a real tokenizer/native mismatch leaves the
    exact P/C arrays and the state needed to diagnose it.
    """
    if (not isinstance(prompt_ids, list) or not prompt_ids
            or any(type(token) is not int for token in prompt_ids)):
        raise ValueError("full-history prompt P must be a non-empty integer token list")
    if type(tail_tokens) is not int or not 0 < tail_tokens < len(prompt_ids):
        raise ValueError("tail length k must be a positive integer smaller than P")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
        raise ValueError("native seed timeout must be positive")
    route = harness.coordinator.routes.get(agent)
    if route is None or route.lane is None or type(route.slot) is not int:
        raise ValueError(f"{agent}: prefix fixture requires an active enrolled route")
    worker = harness.workers[route.lane]
    p_ids = list(prompt_ids)
    c_ids = p_ids[:-tail_tokens]
    evidence = {
        "schema": "strata-owner-prefix-fixture-v1", "agent": agent,
        "lane": route.lane, "slot": route.slot, "tail_tokens": tail_tokens,
        "prompt_token_ids_P": p_ids, "consumed_prefix_token_ids_C": c_ids,
        "P_sha256": token_ids_sha256(p_ids), "C_sha256": token_ids_sha256(c_ids),
        "P_count": len(p_ids), "C_count": len(c_ids), "stage": "before-native-seed",
    }
    if persist:
        persist(dict(evidence))

    try:
        generation_before, inventory_before, held_map_before = _public_lane_inventory(
            worker, timeout_s=float(timeout_s))
        held_before = list(held_map_before.get(route.slot, ()))
        # This is intentionally strict for the isolated comparison worker: a
        # non-empty slot means it is not a fresh cache-isolated baseline.
        if require_empty_slot and held_before:
            raise RuntimeError("fresh comparison slot is not empty before C seed")
        if require_empty_slot:
            if any(row[0] == f"slot-{route.slot}" for row in inventory_before):
                raise RuntimeError("fresh comparison native inventory already contains the target slot")
        from threading import Event
        iterator = harness.coordinator.generate(agent, list(c_ids), 1,
                                                {"temperature": 0.0}, Event(),
                                                timeout_s=float(timeout_s))
        try:
            yielded = list(iterator)  # exhausting verifies the owner drain path
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        output_ids = [token for token in yielded if token is not None]
        # The public owner inventory operation gates/drains the lane while it
        # captures native sessions and Python-held IDs, then resume() explicitly
        # reopens it before any assertions or subsequent coordinator requests.
        generation, inventory, held_map_after = _public_lane_inventory(
            worker, timeout_s=float(timeout_s))
        held_after = list(held_map_after.get(route.slot, ()))
        row = next((row for row in inventory if row[0] == f"slot-{route.slot}"), None)
        evidence.update({
            "stage": "native-seed-observed",
            "native_inventory_generation_before_seed": generation_before,
            "native_inventory_generation_after_seed": generation,
            "held_before_count": len(held_before),
            "held_after_token_ids": held_after,
            "held_after_count": len(held_after),
            "held_after_sha256": token_ids_sha256(held_after),
            "inventory_slot_tokens": row[2] if row else None,
            "inventory_slot_kind": row[1] if row else None,
            "native_output_token_ids_discarded": output_ids,
            "output_count": len(output_ids),
            "drain_confirmed": True,
        })
        if persist:
            persist(dict(evidence))
        if len(output_ids) != 1:
            raise RuntimeError("max_new=1 native seed did not yield exactly one discarded token")
        if held_after != c_ids:
            raise RuntimeError("native slot-held IDs do not exactly equal C=P[:-k]")
        if row is None or row[1] != "slot" or row[2] != len(c_ids):
            raise RuntimeError("native inventory slot count does not exactly equal len(C)")
        evidence["status"] = "PASS"
        evidence["stage"] = "verified"
        if persist:
            persist(dict(evidence))
        return evidence
    except Exception as exc:
        evidence["status"] = "FAIL"
        evidence["failure_type"] = type(exc).__name__
        evidence["failure_message"] = str(exc)
        if persist:
            persist(dict(evidence))
        raise


def _public_lane_inventory(worker: Any, *, timeout_s: float) -> tuple[str, tuple, dict[int, tuple[int, ...]]]:
    """Read slot state only through the owner's public gated inventory API.

    ``quiesce_inventory`` leaves admission gated; every successful capture is
    paired with an explicit ``resume`` before returning to the caller. A failed
    inventory or resume is fail closed and never accesses worker-private snapshot
    or engine slot fields.
    """
    capture = getattr(worker, "quiesce_inventory", None)
    resume = getattr(worker, "resume", None)
    if not callable(capture) or not callable(resume):
        raise RuntimeError("worker lacks public native inventory lifecycle API")
    result = capture(timeout_s)
    if getattr(result, "state", None) != "inventoried":
        raise RuntimeError(f"public native inventory failed: {getattr(result, 'error', None)}")
    try:
        generation = getattr(result, "generation", None)
        sessions = getattr(result, "sessions", None)
        held = getattr(result, "held", None)
        if not isinstance(generation, str) or not generation:
            raise RuntimeError("public native inventory omitted process generation")
        if not isinstance(sessions, tuple) or not isinstance(held, tuple):
            raise RuntimeError("public native inventory has malformed session/held tuples")
        inventory = tuple(tuple(row) for row in sessions)
        if any(len(row) != 3 or not isinstance(row[0], str) or not row[0]
               or not isinstance(row[1], str) or not row[1]
               or type(row[2]) is not int or row[2] < 0
               for row in inventory):
            raise RuntimeError("public native inventory contains malformed session rows")
        held_map: dict[int, tuple[int, ...]] = {}
        for row in held:
            if (not isinstance(row, tuple) or len(row) != 2 or type(row[0]) is not int
                    or row[0] < 0 or row[0] in held_map or not isinstance(row[1], tuple)
                    or any(type(token) is not int for token in row[1])):
                raise RuntimeError("public native inventory contains malformed held-slot rows")
            held_map[row[0]] = row[1]
        return generation, inventory, held_map
    finally:
        resumed = resume(timeout_s)
        if getattr(resumed, "state", None) != "resumed":
            raise RuntimeError(f"public native inventory resume failed: {getattr(resumed, 'error', None)}")


def fresh_isolated_native_baseline(harness_factory: Callable[[], Any], *,
                                   fixtures: Mapping[str, tuple[str, list[dict], int]],
                                   close_harness: Callable[[Any], None],
                                   persist: Callable[[str, dict], None] | None = None,
                                   template_kwargs: Mapping[str, Any] | None = None,
                                   max_new: int = 4) -> dict[str, dict]:
    """Run the uninterrupted comparison in a newly constructed one-worker process.

    The factory owns construction from the same reviewed launch manifest and
    must return a fresh ``single_worker`` harness. Each target slot is seeded
    from C=P[:-k] and immediately continued with identical P. This avoids
    treating a parked session or a prior main-worker cache as baseline proof.
    The caller must close the returned harness and its exact Service in a
    ``finally`` block before constructing the three migration workers.
    """
    if not callable(harness_factory) or not callable(close_harness) or not fixtures:
        raise ValueError("fresh harness construction/close callbacks and fixtures are required")
    if max_new not in (1, 4, 8, 16, 32):
        raise ValueError("baseline generation length is outside the bounded probe set")
    harness = harness_factory()
    try:
        if getattr(harness, "stage", None) != "single_worker" or len(harness.workers) != 1:
            raise RuntimeError("comparison factory did not return a single-worker harness")
        only_worker = next(iter(harness.workers.values()))
        engine = only_worker.engine
        process = getattr(engine, "proc", None)
        generation = getattr(engine, "gen", None)
        if (process is None or type(getattr(process, "pid", None)) is not int or process.pid <= 0
                or not callable(getattr(process, "poll", None)) or process.poll() is not None
                or type(generation) is not int or generation <= 0):
            raise RuntimeError("comparison harness lacks a live, identified fresh native process")
        process_identity = {"pid": process.pid, "generation": generation,
                            "alive_at_start": True, "worker_count": 1}
        fixture_routes = {item[0] for item in fixtures.values() if isinstance(item, tuple) and len(item) == 3}
        if len(fixture_routes) == 0 or fixture_routes - set(harness.coordinator.routes):
            raise RuntimeError("comparison harness is missing one or more fixture routes")
        rows = {}
        used_routes: set[str] = set()
        def check_process():
            # A PID can be reused. Pin the actual Popen object and the READY
            # lifecycle counter, which Strata increments on each ready process.
            if (only_worker.engine is not engine or engine.proc is not process
                    or type(getattr(engine, "gen", None)) is not int
                    or engine.gen != generation or process.pid != process_identity["pid"]
                    or process.poll() is not None):
                raise RuntimeError("comparison native process or generation changed")
        for fixture_name, (agent, messages, tail) in fixtures.items():
            check_process()
            if not isinstance(messages, list) or any(not isinstance(x, dict) for x in messages):
                raise ValueError(f"{fixture_name}: fixture history must be a normal HTTP message list")
            if agent not in harness.coordinator.routes:
                raise RuntimeError(f"{fixture_name}: comparison route {agent} is not registered")
            prompt = tokenize_full_history(harness.frontend, [dict(m) for m in messages],
                                          template_kwargs=template_kwargs)
            seeded = seed_owner_prefix_fixture(harness, agent=agent, prompt_ids=prompt,
                                               tail_tokens=tail,
                                               require_empty_slot=agent not in used_routes,
                                               persist=(lambda row, a=fixture_name: persist(a, {"seed": row})) if persist else None)
            check_process()
            probe = collect_token_ids(harness.coordinator, harness.workers, agent=agent,
                                      prompt_ids=prompt, max_new=max_new,
                                      sampling={"temperature": 0.0})
            check_process()
            if not 0 < len(probe.get("output_token_ids", [])) <= max_new:
                raise RuntimeError(f"{fixture_name}: fresh native baseline returned no bounded token sequence")
            reused, prompt_read = probe.get("reused"), probe.get("prompt_read")
            if type(reused) is not int or not 0 < reused <= seeded["C_count"]:
                raise RuntimeError(f"{fixture_name}: fresh native baseline did not prove bounded positive prefix reuse")
            if type(prompt_read) is not int or prompt_read > tail:
                raise RuntimeError(f"{fixture_name}: fresh native baseline read more than the P suffix")
            if probe["held_prefix_before"].get("sha256") != seeded["C_sha256"]:
                raise RuntimeError(f"{fixture_name}: baseline continuation did not begin from the freshly seeded C slot")
            row = {"seed": seeded, "baseline": probe,
                   "route_agent": agent, "source": "fresh-isolated-single-worker-slot",
                   "comparison_process": process_identity,
                   "reuse_within_seeded_prefix": reused <= seeded["C_count"],
                   "prefill_bounded_by_tail": prompt_read <= tail}
            rows[fixture_name] = row
            used_routes.add(agent)
            if persist:
                persist(fixture_name, row)
        check_process()
        return rows
    finally:
        close_harness(harness)


def tokenize_full_history(service: Any, messages: list[dict], *,
                          template_kwargs: dict | None = None) -> list[int]:
    """Use the ready Service's exact template/tokenizer path for direct tests."""
    encoder = getattr(service, "encode_prompt", None)
    if not callable(encoder):
        raise TypeError("ready service must expose encode_prompt")
    ids = encoder(messages, [], template_kwargs or {}, store=False)
    if not isinstance(ids, list) or any(type(x) is not int for x in ids):
        raise RuntimeError("service prompt encoder returned invalid token IDs")
    return ids


def compare_token_ids(baseline: list[int], candidate: list[int]) -> dict:
    divergence = next((i for i, (a, b) in enumerate(zip(baseline, candidate)) if a != b), None)
    if divergence is None and len(baseline) != len(candidate):
        divergence = min(len(baseline), len(candidate))
    return {"exact_parity": baseline == candidate, "baseline_count": len(baseline),
            "candidate_count": len(candidate), "first_divergence_index": divergence,
            "baseline_divergent_token": baseline[divergence] if divergence is not None and divergence < len(baseline) else None,
            "candidate_divergent_token": candidate[divergence] if divergence is not None and divergence < len(candidate) else None}





def token_ids_sha256(ids: Iterable[int]) -> str:
    """Hash an ID sequence without persisting or logging the IDs themselves."""
    raw = json.dumps(list(ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def prefix_relation(left: Iterable[int], right: Iterable[int]) -> dict:
    """Summarize token prefix alignment without exposing token values."""
    a, b = list(left), list(right)
    lcp = 0
    for x, y in zip(a, b):
        if x != y:
            break
        lcp += 1
    return {"left_count": len(a), "right_count": len(b), "lcp_tokens": lcp,
            "left_is_strict_prefix": len(a) < len(b) and lcp == len(a),
            "right_is_strict_prefix": len(b) < len(a) and lcp == len(b),
            "equal": len(a) == len(b) and lcp == len(a),
            "left_sha256": token_ids_sha256(a), "right_sha256": token_ids_sha256(b)}


def _slot_held_ids(harness: Any, agent: str) -> tuple[int, ...]:
    route = harness.coordinator.routes.get(agent)
    if route is None or route.lane is None or type(route.slot) is not int:
        raise RuntimeError(f"{agent}: no active native slot for prefix inspection")
    worker = harness.workers[route.lane]
    engine = worker.engine
    with engine.slot_cv:
        held = getattr(engine, "slot_held", None)
        if not isinstance(held, list) or not 0 <= route.slot < len(held):
            raise RuntimeError(f"{agent}: native slot-held inventory is unavailable")
        ids = held[route.slot]
        if not isinstance(ids, list) or any(type(x) is not int for x in ids):
            raise RuntimeError(f"{agent}: native slot-held token inventory is malformed")
        return tuple(ids)


def prime_exact_full_history_prompts(client: Any, harness: Any, *, model: str,
                                     messages_by_agent: Mapping[str, list[dict]],
                                     max_new: int = 4) -> tuple[dict[str, list[int]], dict[str, dict]]:
    """Consume each exact native-test prompt once, then verify its slot contains it.

    This creates an actual native prefix for the exact full-history IDs that the
    later baseline and migrated branches will submit. The HTTP request is sent
    with the same messages used for `encode_prompt`; the assistant response is
    deliberately not appended, so no decoded/re-encoded response can shift the
    test prefix. Only hashes/counts are returned as evidence.
    """
    if not isinstance(messages_by_agent, Mapping) or not messages_by_agent:
        raise ValueError("one full-history message list is required per enrolled agent")
    prompts, evidence = {}, {}
    for agent, messages in messages_by_agent.items():
        route = harness.coordinator.routes.get(agent)
        if route is None or route.lane is None:
            raise RuntimeError(f"{agent}: prompt priming requires an active source route")
        # The gateway renders/tokenizes on the registered frontend Service;
        # use that exact renderer for every routed agent.
        service = harness.frontend
        prompt = tokenize_full_history(service, [dict(m) for m in messages])
        if len(prompt) < 32:
            raise RuntimeError(f"{agent}: qualification prompt is unexpectedly short")
        probe = client.request("/v1/chat/completions", model=model,
                               messages=[dict(m) for m in messages], agent_id=agent,
                               stream=False, max_new=max_new, temperature=0.0)
        if probe.status != "PASS":
            raise RuntimeError(f"{agent}: exact full-history priming request failed")
        held = _slot_held_ids(harness, agent)
        relation = prefix_relation(prompt, held)
        # A completed request stores the consumed prompt plus all output except
        # the final unconsumed token. Therefore this exact input must be the
        # beginning of the held sequence; a shorter common prefix is a harness
        # mismatch and must stop before consolidation/model reload work.
        if not relation["left_is_strict_prefix"] and not relation["equal"]:
            raise RuntimeError(f"{agent}: native slot does not hold the exact primed full-history prompt")
        prompts[agent] = prompt
        engine = harness.workers[route.lane].engine
        info = getattr(engine, "info", {}) or {}
        evidence[agent] = {"lane": route.lane, "slot": route.slot,
                           "prompt_tokens": len(prompt), "slot_held_tokens": len(held),
                           "prompt_sha256": token_ids_sha256(prompt),
                           "slot_held_sha256": token_ids_sha256(held),
                           "alignment": relation,
                           "prompt_cache": info.get("prompt_cache"),
                           "slot_cache": info.get("slot_cache"),
                           "session_batch_restore": info.get("session_restore_batch"),
                           "priming_http_status": probe.http_status}
    return prompts, evidence


def checkpoint_prefixes(harness: Any, *, expected_prompts: Mapping[str, list[int]] | None = None) -> dict[str, dict]:
    """Durably checkpoint every enrolled slot, then resume the unchanged lane."""
    prefixes = {}
    for lane, worker in harness.workers.items():
        result = worker.quiesce_snapshot(harness.store, 30.0)
        if getattr(result, "state", None) != "parked":
            raise RuntimeError(f"{lane}: prefix snapshot failed; owner recovery required")
        sessions = harness.store.sessions(result.checkpoint_id, harness.coordinator.model)
        # NativeLaneWorker captures this inventory while holding its lifecycle,
        # service FIFO, engine control lock and admission gate. The lane remains
        # gated after quiesce_snapshot returns, so this immutable captured proof
        # is safer than calling session_inventory() directly from the runner.
        parked_snapshot = getattr(worker, "_parked_snapshot", None)
        if (not isinstance(parked_snapshot, tuple) or len(parked_snapshot) != 2
                or not isinstance(parked_snapshot[1], tuple)):
            raise RuntimeError(f"{lane}: owner did not retain a gated native inventory snapshot")
        token_counts = {sid: count for sid, kind, count in parked_snapshot[1] if kind == "slot"}
        manifest_path = harness.store.root / result.checkpoint_id / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        for entry in manifest.get("sessions", []):
            harness.store.restore(result.checkpoint_id, entry["session"]["session_id"],
                                  harness.coordinator.model, lambda *_: None)
        by_slot = {}
        for entry in sessions:
            suffix = entry.session_id.rsplit(":", 1)[-1]
            if suffix.startswith("slot-"):
                by_slot[int(suffix[5:])] = entry.session_id
        for agent, route in harness.coordinator.routes.items():
            if route.lane == lane:
                if route.slot not in by_slot:
                    raise RuntimeError(f"{lane}: checkpoint omits slot {route.slot}")
                prefix_count = token_counts.get(f"slot-{route.slot}")
                if type(prefix_count) is not int or prefix_count <= 0:
                    raise RuntimeError(f"{lane}: seeded slot has no positive native prefix token count")
                prefixes[agent] = {"lane": lane, "slot": route.slot,
                                   "checkpoint_id": result.checkpoint_id,
                                   "session_id": by_slot[route.slot],
                                   "prefix_token_count": prefix_count,
                                   "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                                   "payload_sha256": next(x["sha256"] for x in manifest["sessions"]
                                                          if x["session"]["session_id"] == by_slot[route.slot])}
                held = worker._resume_held.get(route.slot)
                if not isinstance(held, tuple) or not held:
                    raise RuntimeError(f"{agent}: checkpoint omitted verified held-token routing evidence")
                if prefix_count != len(held):
                    raise RuntimeError(f"{agent}: native inventory token count differs from held-ID routing proof")
                prefixes[agent]["slot_held_token_count"] = len(held)
                prefixes[agent]["slot_held_sha256"] = token_ids_sha256(held)
                if expected_prompts is not None:
                    prompt = expected_prompts.get(agent)
                    if not isinstance(prompt, list):
                        raise RuntimeError(f"{agent}: expected full-history prompt is missing")
                    alignment = prefix_relation(prompt, held)
                    if not (alignment["left_is_strict_prefix"] or alignment["equal"]):
                        raise RuntimeError(f"{agent}: saved checkpoint no longer holds its exact prompt prefix")
                    prefixes[agent]["prompt_alignment"] = alignment
        resumed = worker.resume(30.0)
        if getattr(resumed, "state", None) != "resumed":
            raise RuntimeError(f"{lane}: prefix snapshot could not resume")
    if set(prefixes) != set(harness.coordinator.routes):
        raise RuntimeError("not every enrolled active agent has a durable prefix")
    return prefixes


def assert_current_slots_match_saved(harness: Any, prefixes: Mapping[str, dict]) -> None:
    """Prove slot routing IDs exactly match the saved branch after restoration."""
    routes = harness.coordinator.routes
    if set(routes) != set(prefixes):
        raise RuntimeError("restored state agent inventory differs from saved state")
    for agent, saved in prefixes.items():
        route = routes[agent]
        if (route.lane, route.slot) != (saved["lane"], saved["slot"]):
            raise RuntimeError(f"{agent}: restored route differs from its saved lane/slot")
        held = _slot_held_ids(harness, agent)
        if (len(held) != saved.get("slot_held_token_count")
                or token_ids_sha256(held) != saved.get("slot_held_sha256")):
            raise RuntimeError(f"{agent}: restored native slot IDs differ from the saved branch")


def assert_saved_slot_state_equal(left: Mapping[str, dict], right: Mapping[str, dict]) -> None:
    """Fail if a later pre-consolidation save no longer represents the same slots."""
    if set(left) != set(right):
        raise RuntimeError("saved slot state agent inventories differ")
    for agent in left:
        a, b = left[agent], right[agent]
        for key in ("lane", "slot", "prefix_token_count", "slot_held_token_count",
                    "slot_held_sha256", "payload_sha256"):
            if a.get(key) != b.get(key):
                raise RuntimeError(f"{agent}: saved checkpoint state differs at {key}")


def assert_routes_reference_saved_payloads(harness: Any, prefixes: Mapping[str, dict]) -> None:
    """Bind post-consolidation routes to the exact pre-consolidation slot payloads."""
    routes = harness.coordinator.routes
    if set(routes) != set(prefixes):
        raise RuntimeError("consolidated route inventory differs from saved slot inventory")
    for agent, route in routes.items():
        if not route.checkpoint_id or not route.session_id:
            raise RuntimeError(f"{agent}: consolidated route lacks durable checkpoint reference")
        manifest = json.loads((harness.store.root / route.checkpoint_id / "manifest.json").read_text())
        entries = [x for x in manifest.get("sessions", [])
                   if x.get("session", {}).get("session_id") == route.session_id]
        if len(entries) != 1 or entries[0].get("sha256") != prefixes[agent].get("payload_sha256"):
            raise RuntimeError(f"{agent}: consolidation did not preserve the aligned saved slot payload")


def baseline_once_and_restore(harness: Any, prompts: Mapping[str, list[int]],
                              prefixes: Mapping[str, dict], *, max_new: int = 4,
                              sampling: dict | None = None) -> dict:
    """Collect one baseline branch from the exact saved state, then restore it."""
    if set(prompts) != set(prefixes) or set(prompts) != set(harness.coordinator.routes):
        raise ValueError("baseline inputs, saved prefixes and routes must contain the same agents")
    restore_prefixes(harness, prefixes)
    assert_current_slots_match_saved(harness, prefixes)
    first = {agent: collect_token_ids(harness.coordinator, harness.workers, agent=agent,
                                      prompt_ids=prompts[agent], max_new=max_new, sampling=sampling)
             for agent in prompts}
    for agent, row in first.items():
        if type(row.get("reused")) is not int or row["reused"] <= 0:
            raise RuntimeError(f"{agent}: aligned baseline did not prove native prefix reuse")
    restore_prefixes(harness, prefixes)
    assert_current_slots_match_saved(harness, prefixes)
    return {"prefixes": dict(prefixes), "first": first}


def displace_global_contexts(harness: Any, prompts: Mapping[str, list[int]],
                             prefixes: Mapping[str, dict]) -> dict:
    """Replace each lane's global live prompt cache with an unrelated bounded probe.

    The per-agent checkpoint slots are restored immediately afterward. This
    prevents a repeated same-prompt request in the lane's main session from
    serving as the only source of apparent prefix reuse. The displacer is
    deliberately unique and its LCP against every qualification prompt is
    recorded; only counts/hashes are persisted.
    """
    agents_by_lane: dict[str, list[str]] = {}
    for agent, saved in prefixes.items():
        agents_by_lane.setdefault(saved["lane"], []).append(agent)
    rows = {}
    for lane in sorted(agents_by_lane):
        agent = sorted(agents_by_lane[lane])[0]
        message = [{"role": "user", "content":
                    f"CACHE-ISOLATION-{lane}-NO-QUAL-FACT-20261019. Return one short word."}]
        service = harness.workers[lane].service
        distractor = tokenize_full_history(service, message)
        relations = {a: prefix_relation(distractor, prompts[a]) for a in prompts}
        if any(relation["lcp_tokens"] >= len(prompts[a]) - 16
               for a, relation in relations.items()):
            raise RuntimeError(f"{lane}: cache-displacer prompt is not distinct from qualification histories")
        result = collect_token_ids(harness.coordinator, harness.workers, agent=agent,
                                  prompt_ids=distractor, max_new=4,
                                  sampling={"temperature": 0.0})
        if not result["output_token_ids"]:
            raise RuntimeError(f"{lane}: cache-displacer probe produced no native output")
        rows[lane] = {"agent": agent, "prompt_tokens": len(distractor),
                      "prompt_sha256": token_ids_sha256(distractor),
                      "lcp_tokens_by_agent": {a: r["lcp_tokens"] for a, r in relations.items()},
                      "output_count": len(result["output_token_ids"])}
    restore_prefixes(harness, prefixes)
    assert_current_slots_match_saved(harness, prefixes)
    return rows

def restore_prefixes(harness: Any, prefixes: Mapping[str, dict]) -> None:
    """Restore original checkpoints into original slots with the native proof APIs."""
    for lane, worker in harness.workers.items():
        lane_agents = [(agent, p) for agent, p in prefixes.items() if p["lane"] == lane]
        if not lane_agents:
            continue
        parked = worker.quiesce_snapshot(harness.store, 30.0)
        if getattr(parked, "state", None) != "parked":
            raise RuntimeError(f"{lane}: reset snapshot failed; owner recovery required")
        # First evict every occupied slot against the same verified full-batch
        # snapshot. Then restore the original prefixes into the now-empty slots.
        for agent, prefix in lane_agents:
            route = harness.coordinator.routes[agent]
            if route.lane != lane or route.slot != prefix["slot"]:
                raise RuntimeError(f"{agent}: route changed; cannot restore original slot")
            released = worker.release_slot(prefix["slot"], 30.0)
            if getattr(released, "state", None) != "released":
                raise RuntimeError(f"{lane}: slot release failed; owner recovery required")
        for agent, prefix in lane_agents:
            restored = worker.restore_slot(harness.store, prefix["checkpoint_id"],
                                           prefix["session_id"], prefix["slot"], 30.0)
            if getattr(restored, "state", None) != "restored":
                raise RuntimeError(f"{lane}: prefix restore failed; owner recovery required")
        resumed = worker.resume(30.0)
        if getattr(resumed, "state", None) != "resumed":
            raise RuntimeError(f"{lane}: restored prefix could not resume")


def baseline_repeat_and_restore(harness: Any, prompts: Mapping[str, list[int]], *,
                                max_new: int = 8, sampling: dict | None = None) -> dict:
    """Repeat native greedy continuation and measure same-prefix determinism.

    Prompt IDs must be rendered from each full normal conversation history using
    the ready Service tokenizer/template. The method does not synthesize context.
    """
    routes = harness.coordinator.routes
    if set(prompts) != set(routes):
        raise ValueError("supply one full-history prompt per active agent")
    if len({tuple(ids) for ids in prompts.values()}) != len(prompts):
        raise ValueError("each agent must have a distinct tokenized fact/history")
    prefixes = checkpoint_prefixes(harness)
    first = {a: collect_token_ids(harness.coordinator, harness.workers, agent=a,
                                  prompt_ids=prompts[a], max_new=max_new, sampling=sampling)
             for a in routes}
    restore_prefixes(harness, prefixes)
    repeat = {a: collect_token_ids(harness.coordinator, harness.workers, agent=a,
                                  prompt_ids=prompts[a], max_new=max_new, sampling=sampling)
              for a in routes}
    restore_prefixes(harness, prefixes)
    return {"prefixes": prefixes, "first": first, "repeat": repeat,
            "repeat_comparison": {a: compare_token_ids(first[a]["output_token_ids"],
                                                       repeat[a]["output_token_ids"])
                                  for a in routes}}


def migrated_native_branch(harness: Any, prompts: Mapping[str, list[int]],
                           baseline: Mapping[str, dict], *, max_new: int = 8,
                           sampling: dict | None = None, timeout_s: float = 30.0) -> dict:
    """Continue all agents after consolidation and compare exact native IDs.

    Direct token-ID collection requires an active route. Parked agents are
    activated through the coordinator's durable checkpoint/LRU path first;
    this wrapper never releases a slot or edits route state itself.
    """
    coordinator = harness.coordinator
    initial_routes = coordinator.routes
    if set(prompts) != set(initial_routes) or set(baseline) != set(prompts):
        raise ValueError("migrated branch requires matching prompts and baseline for every enrolled agent")

    def identity(route):
        return (getattr(route, "lane", None), getattr(route, "slot", None))

    expected_parked = {agent for agent, route in initial_routes.items()
                       if getattr(route, "lane", None) is None}
    activated = set()
    trials, comparisons, activation_evidence = {}, {}, []
    for agent in prompts:
        before = coordinator.routes
        prior = before.get(agent)
        if prior is None:
            raise RuntimeError(f"{agent}: enrolled route disappeared before native continuation")
        prior_identity = identity(prior)
        if prior_identity[0] is None:
            activate = getattr(coordinator, "activate", None)
            if not callable(activate):
                raise RuntimeError(f"{agent}: parked route requires coordinator.activate")
            result = activate(agent, timeout_s=timeout_s)
            state = getattr(result, "state", None)
            if state != "ready":
                raise RuntimeError(f"{agent}: durable activation did not return ready")
            after = coordinator.routes
            route = after.get(agent)
            active_identity = identity(route) if route is not None else (None, None)
            if (active_identity[0] is None or type(active_identity[1]) is not int
                    or active_identity[1] < 0):
                raise RuntimeError(f"{agent}: activation reported ready but route is not active")
            changed = {name for name in before
                       if name not in after or identity(before[name]) != identity(after[name])}
            added = set(after) - set(before)
            expected_changes = {agent} | {
                name for name in before if identity(before[name])[0] is not None
                and name in after and identity(after[name])[0] is None
            }
            if added or set(after) != set(before) or changed != expected_changes:
                raise RuntimeError(f"{agent}: activation changed unexpected coordinator routes")
            displaced = sorted(name for name in changed if name != agent)
            if len(displaced) > 1 or any(identity(before[name])[0] is None for name in displaced):
                raise RuntimeError(f"{agent}: activation did not use the bounded destination LRU path")
            activated.add(agent)
            activation = {"required": True, "result_state": state,
                          "before": {"lane": prior_identity[0], "slot": prior_identity[1]},
                          "after": {"lane": active_identity[0], "slot": active_identity[1]},
                          "displaced_agents": displaced}
        else:
            route = prior
            activation = {"required": False, "result_state": "already-active",
                          "before": {"lane": prior_identity[0], "slot": prior_identity[1]},
                          "after": {"lane": prior_identity[0], "slot": prior_identity[1]},
                          "displaced_agents": []}

        row = collect_token_ids(coordinator, harness.workers, agent=agent,
                                prompt_ids=prompts[agent], max_new=max_new,
                                sampling=sampling, timeout_s=timeout_s)
        after_generate = coordinator.routes.get(agent)
        if identity(after_generate) != (row.get("lane"), row.get("slot")):
            raise RuntimeError(f"{agent}: route changed during native token-ID collection")
        row["activation"] = activation
        trials[agent] = row
        activation_evidence.append({"agent": agent, **activation})
        comparisons[agent] = compare_token_ids(baseline[agent]["output_token_ids"],
                                               row["output_token_ids"])
    if not expected_parked.issubset(activated):
        raise RuntimeError("migrated branch did not explicitly activate every initially parked agent")
    if not activated.issubset(set(initial_routes)):
        raise RuntimeError("migrated branch activation evidence names an unenrolled agent")
    return {"trials": trials, "comparison": comparisons,
            "activation_evidence": activation_evidence,
            "initially_parked_agents": sorted(expected_parked),
            "activated_agents": sorted(activated),
            "reactivated_initial_residents": sorted(activated - expected_parked)}


def install_durable_event_recorder(harness: Any, results: JsonlResults) -> None:
    """Instrument only this isolated coordinator's durable journals and exits.

    The wrapper calls each original operation unchanged, then records its
    result and hashes. The journal record is emitted after the native API's
    fsync/atomic replace returns. No credentials or prompt contents are logged.
    """
    import types
    coordinator = harness.coordinator
    original_journal = coordinator._journal

    def journal(phase, error=None):
        result = original_journal(phase, error)
        path = coordinator.journal_path
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            raw = json.loads(path.read_text(encoding="utf-8"))
            checkpoint_refs = dict(raw.get("snapshots", {}))
        except Exception as exc:
            results.event("journal-read-failed", phase=phase, exception=type(exc).__name__)
            raise
        manifests = {}
        for lane, checkpoint_id in checkpoint_refs.items():
            manifest = harness.store.root / checkpoint_id / "manifest.json"
            manifests[lane] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        results.event("journal-durable", phase=phase, journal_sha256=digest,
                      checkpoint_refs=checkpoint_refs, manifest_sha256=manifests)
        return result

    coordinator._journal = journal
    for lane, worker in harness.workers.items():
        original_stop = worker.stop_parked

        def make_stop(lane_name, worker_obj, method):
            def stop_parked(store, checkpoint_id, timeout_s=30.0):
                engine = worker_obj.engine
                proc = getattr(engine, "proc", None)
                identity = {"lane": lane_name, "pid": getattr(proc, "pid", None),
                            "generation": getattr(engine, "gen", None),
                            "poll_before": proc.poll() if proc is not None else None}
                results.event("source-stop-begin", checkpoint_id=checkpoint_id, **identity)
                outcome = method(store, checkpoint_id, timeout_s)
                identity["poll_after"] = proc.poll() if proc is not None else None
                identity["engine_alive_after"] = bool(engine.alive())
                results.event("source-stop-return", checkpoint_id=checkpoint_id,
                              result_state=getattr(outcome, "state", None), **identity)
                return outcome
            return stop_parked
        worker.stop_parked = make_stop(lane, worker, original_stop)


def validate_durable_exit_order(jsonl_path: str, *, source_lanes: set[str],
                                expected_lanes: set[str]) -> dict:
    """Audit event ordering from an actual recorder JSONL; fail closed."""
    from pathlib import Path
    try:
        rows = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError) as exc:
        return {"pass": False, "reason": f"event log unreadable: {type(exc).__name__}"}
    events = [row for row in rows if str(row.get("name", "")).startswith("owner-event:")]
    snapshots = [e for e in events if e.get("name") == "owner-event:journal-durable"
                 and e.get("phase") == "all-snapshots-durable"]
    routes = [e for e in events if e.get("name") == "owner-event:journal-durable"
              and e.get("phase") == "routes-durable"]
    if not snapshots or not routes:
        return {"pass": False, "reason": "durable full-snapshot and routes journal events are required"}
    route_event = routes[-1]
    snapshots_event = snapshots[-1]
    if not isinstance(snapshots_event.get("at_monotonic"), (int, float)) or snapshots_event["at_monotonic"] >= route_event.get("at_monotonic", 0):
        return {"pass": False, "reason": "route references were not journaled after all snapshots"}
    refs, hashes = route_event.get("checkpoint_refs"), route_event.get("manifest_sha256")
    if (not isinstance(refs, dict) or set(refs) != expected_lanes
            or not isinstance(hashes, dict) or set(hashes) != expected_lanes
            or any(not isinstance(v, str) or len(v) != 64 for v in hashes.values())):
        return {"pass": False, "reason": "routes-durable does not bind every lane checkpoint manifest"}
    if snapshots_event.get("checkpoint_refs") != refs:
        return {"pass": False, "reason": "routes journal does not preserve the complete all-snapshots reference set"}
    route_time = route_event.get("at_monotonic")
    if not isinstance(route_time, (float, int)):
        return {"pass": False, "reason": "routes-durable event lacks a monotonic timestamp"}
    stops = [e for e in events if e.get("name") == "owner-event:source-stop-begin"
             and e.get("lane") in source_lanes]
    returns = [e for e in events if e.get("name") == "owner-event:source-stop-return"
               and e.get("lane") in source_lanes]
    exited = {}
    for lane in source_lanes:
        begins = [e for e in stops if e.get("lane") == lane]
        ends = [e for e in returns if e.get("lane") == lane]
        if len(begins) != 1 or len(ends) != 1:
            return {"pass": False, "reason": f"expected one stop begin/return for {lane}"}
        begin, end = begins[0], ends[0]
        if begin.get("checkpoint_id") != refs.get(lane):
            return {"pass": False, "reason": f"{lane} stop does not name its routes-durable checkpoint"}
        if begin.get("at_monotonic", 0) <= route_time:
            return {"pass": False, "reason": f"{lane} stop began before durable route publication"}
        if (end.get("at_monotonic", 0) <= begin.get("at_monotonic", 0)
                or end.get("result_state") != "stopped"
                or end.get("engine_alive_after") is not False
                or not isinstance(end.get("poll_after"), int)):
            return {"pass": False, "reason": f"{lane} native exit was not confirmed after stop"}
        exited[lane] = end.get("poll_after")
    return {"pass": True, "routes_durable_event": route_time,
            "checkpoint_refs": refs, "manifest_sha256": hashes,
            "source_exit_codes": exited}
