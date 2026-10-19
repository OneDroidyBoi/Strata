"""Opt-in native inventory/export framing; the caller owns the service FIFO.

Protocol failures block further checkpoint commands without killing the engine.
The lifecycle owner keeps admission gated until an operator resolves the failure.
"""
import math
import queue
import re
import time


class SessionProtocolError(RuntimeError):
    pass


def _token(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value)


def _count(value):
    if not value.isascii() or not value.isdigit() or len(value) > 20:
        raise SessionProtocolError('invalid native session count')
    return int(value)


class SessionProtocol:
    def __init__(self, engine, timeout_s=30.0):
        if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
            raise ValueError('session protocol timeout must be in (0, 60] seconds')
        self.engine, self.timeout = engine, timeout_s

    def _start(self, command):
        if not self.engine.session_lifecycle or getattr(self.engine, '_session_protocol_failed', False):
            raise SessionProtocolError('native session lifecycle is unavailable or its protocol is blocked')
        if not self.engine.alive():
            raise SessionProtocolError('native engine is not running')
        self.engine._send(command)

    def _read(self, deadline):
        left = deadline - time.monotonic()
        if left <= 0:
            raise SessionProtocolError('native session command timed out')
        try:
            line = self.engine.lines.get(timeout=left)
        except queue.Empty:
            raise SessionProtocolError('native session command timed out') from None
        if line is None:
            raise SessionProtocolError('native session protocol ended')
        if not isinstance(line, str) or len(line) > 8192:
            raise SessionProtocolError('invalid native session response')
        if line.startswith('SERR '):
            from serve.server import SessionRefused
            parts = line.rstrip('\r\n').split(' ', 3)
            if len(parts) != 4 or parts[1] not in SessionRefused.STATUS or parts[2] not in ('0', '1'):
                raise SessionProtocolError('malformed native refusal')
            raise SessionRefused(parts[1], parts[3], parts[2] == '1')
        return line.split()

    def inventory(self):
        from serve.server import SessionRefused
        try:
            self._start('INVENTORY')
            deadline = time.monotonic() + self.timeout
            fields = self._read(deadline)
            if len(fields) != 3 or fields[0] != 'INVENTORY' or not _token(fields[1]):
                raise SessionProtocolError('invalid inventory header')
            generation, count = fields[1], _count(fields[2])
            if count > 1024:
                raise SessionProtocolError('native inventory exceeds supported session count')
            sessions, seen = [], set()
            for _ in range(count):
                fields = self._read(deadline)
                if (len(fields) != 4 or fields[0] != 'SESSION_ENTRY' or not _token(fields[1])
                        or fields[1] in seen or fields[2] not in ('live', 'parked')):
                    raise SessionProtocolError('invalid or duplicate native session entry')
                seen.add(fields[1])
                sessions.append({'id': fields[1], 'kind': fields[2], 'tokens': _count(fields[3])})
            if self._read(deadline) != ['INVENTORY_END', generation]:
                raise SessionProtocolError('incomplete or inconsistent native inventory')
            return {'generation': generation, 'sessions': sessions}
        except SessionRefused:
            raise
        except Exception:
            self.engine._session_protocol_failed = True
            raise

    def export(self, generation, session_id, path):
        from serve.server import SessionRefused
        if not _token(generation) or not _token(session_id):
            raise ValueError('invalid export generation or session ID')
        if not isinstance(path, str) or not path or any(c in path for c in '\r\n\0'):
            raise ValueError('invalid export path')
        try:
            self._start(f'EXPORT {generation} {session_id} {path}')
            deadline = time.monotonic() + self.timeout
            last_done, last_total = -1, None
            while True:
                fields = self._read(deadline)
                if len(fields) == 3 and fields[0] == 'SESSION':
                    done, total = _count(fields[1]), _count(fields[2])
                    if done < last_done or done > total or (last_total is not None and total != last_total):
                        raise SessionProtocolError('invalid native export progress')
                    # Repeated progress cannot keep a stuck writer alive forever.
                    if done > last_done:
                        deadline = time.monotonic() + self.timeout
                    last_done, last_total = done, total
                elif len(fields) == 3 and fields[0] == 'SWAIT':
                    from serve.server import SESSION_WAIT_MAX_S
                    seconds = _count(fields[2])
                    if not fields[1].isascii() or not fields[1].isalpha() or not 0 < seconds <= SESSION_WAIT_MAX_S:
                        raise SessionProtocolError('invalid native blocking phase')
                    deadline = time.monotonic() + max(self.timeout, seconds)
                elif len(fields) == 4 and fields[0] == 'SAVED':
                    tokens, size, ms = _count(fields[1]), _count(fields[2]), float(fields[3])
                    if not math.isfinite(ms) or ms < 0:
                        raise SessionProtocolError('invalid native export timing')
                    return {'tokens': tokens, 'bytes': size, 'ms': ms}
                else:
                    raise SessionProtocolError('unexpected native export response')
        except SessionRefused:
            raise
        except Exception:
            self.engine._session_protocol_failed = True
            raise
