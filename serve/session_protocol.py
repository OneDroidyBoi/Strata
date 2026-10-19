"""Opt-in native inventory/export framing; the caller owns FIFO and batch ctl.

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
                        or fields[1] in seen or fields[2] not in ('live', 'parked', 'slot')):
                    raise SessionProtocolError('invalid or duplicate native session entry')
                if fields[2] == 'slot':
                    slot = re.fullmatch(r'slot-(0|[1-9][0-9]*)', fields[1])
                    if (getattr(self.engine, 'session_batch_export', False) is not True or slot is None
                            or int(slot[1]) >= getattr(self.engine, 'batch', 0)):
                        raise SessionProtocolError('unadvertised or invalid native batch slot')
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

    def restore_slot(self, slot, path):
        """Restore an empty slot and receive its exact routing prefix in chunks."""
        from serve.server import SessionRefused
        if (getattr(self.engine, 'session_batch_restore', False) is not True
                or type(slot) is not int or not 0 <= slot < getattr(self.engine, 'batch', 0)):
            raise ValueError('unadvertised or invalid native restore slot')
        if not isinstance(path, str) or not path or any(c in path for c in '\r\n\0'):
            raise ValueError('invalid restore path')
        limit = getattr(self.engine, 'max_context', 0)
        if type(limit) is not int or limit <= 0:
            raise ValueError('native restore context limit is unavailable')
        try:
            self._start(f'RESTORE_SLOT {slot} {path}')
            deadline = time.monotonic() + self.timeout
            ids, last_done, last_total = [], -1, None
            while True:
                fields = self._read(deadline)
                if len(fields) == 3 and fields[0] == 'SESSION':
                    done, total = _count(fields[1]), _count(fields[2])
                    if ids or done < last_done or done > total or (last_total is not None and total != last_total):
                        raise SessionProtocolError('invalid native restore progress')
                    if done > last_done:
                        deadline = time.monotonic() + self.timeout
                    last_done, last_total = done, total
                elif len(fields) == 3 and fields[0] == 'SWAIT':
                    from serve.server import SESSION_WAIT_MAX_S
                    seconds = _count(fields[2])
                    if (ids or not fields[1].isascii() or not fields[1].isalpha()
                            or not 0 < seconds <= SESSION_WAIT_MAX_S):
                        raise SessionProtocolError('invalid native blocking phase')
                    deadline = time.monotonic() + max(self.timeout, seconds)
                elif len(fields) == 3 and fields[0] == 'SLOT_IDS':
                    if fields[1] != str(slot):
                        raise SessionProtocolError('native restore slot mismatch')
                    chunk = [_count(value) for value in fields[2].split(',')]
                    if len(ids) + len(chunk) > limit or any(value > 2147483647 for value in chunk):
                        raise SessionProtocolError('native restored prefix exceeds limits')
                    ids.extend(chunk)
                    deadline = time.monotonic() + self.timeout
                elif len(fields) == 5 and fields[0] == 'SLOT_RESTORED':
                    tokens, size, ms = _count(fields[2]), _count(fields[3]), float(fields[4])
                    if fields[1] != str(slot) or tokens != len(ids) or not math.isfinite(ms) or ms < 0:
                        raise SessionProtocolError('invalid native slot restore completion')
                    return {'tokens': tokens, 'bytes': size, 'ms': ms, 'ids': ids}
                else:
                    raise SessionProtocolError('unexpected native slot restore response')
        except SessionRefused:
            raise
        except Exception:
            self.engine._session_protocol_failed = True
            raise
