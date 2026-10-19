"""Opt-in per-request drain proof. EOF and iterator closure are not GPU quiescence."""
from dataclasses import dataclass
import threading
import uuid
import math


def matching_bdone(line, slot):
    """Validate a legacy slot terminal record before treating it as evidence."""
    fields = line.split()
    try:
        return (len(fields) == 5 and fields[0] == 'BDONE' and fields[1] == str(slot)
                and int(fields[2]) >= 0 and fields[3] in {'stop', 'length', 'cancel', 'nonfinite'}
                and math.isfinite(float(fields[4])) and float(fields[4]) >= 0)
    except (ValueError, IndexError):
        return False


@dataclass(frozen=True)
class DrainTicket:
    ticket_id: str
    generation: int


class DrainReceipt:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending = {}
        self._sealed = False
        self._failed = False
        self._callbacks = []
        self._watching = False

    def begin(self, generation, process, slot=None):
        with self._lock:
            if self._sealed:
                raise RuntimeError('cannot add work to a sealed drain receipt')
            ticket = DrainTicket(uuid.uuid4().hex, generation)
            self._pending[ticket] = (process, slot)
            return ticket

    def expect_slot(self, ticket, slot):
        with self._lock:
            process, _ = self._pending[ticket]
            self._pending[ticket] = (process, slot)

    def acknowledge(self, ticket, generation, slot=None):
        with self._lock:
            state = self._pending.get(ticket)
            if state is None or ticket.generation != generation or state[1] != slot:
                return False
            del self._pending[ticket]
        self._notify()
        return True

    def confirm_process_exit(self, ticket):
        with self._lock:
            state = self._pending.get(ticket)
            if state is None:
                return False
            process = state[0]
        try:
            exited = process is not None and process.poll() is not None
        except Exception:
            self.mark_failed()
            return False
        if not exited:
            return False
        with self._lock:
            if ticket not in self._pending:
                return False
            del self._pending[ticket]
        self._notify()
        return True

    def confirmed(self):
        with self._lock:
            return self._sealed and not self._pending

    def failed(self):
        with self._lock:
            return self._failed

    def mark_failed(self):
        with self._lock:
            self._failed = True
        self._notify()

    def add_done_callback(self, callback):
        """Called on failure or confirmed drain; may run again after later exit."""
        with self._lock:
            self._callbacks.append(callback)
        self._notify()

    def _notify(self):
        with self._lock:
            callbacks = list(self._callbacks) if self._failed or (self._sealed and not self._pending) else []
        for callback in callbacks:
            callback(self)

    def seal(self):
        with self._lock:
            self._sealed = True
            start = bool(self._pending) and not self._watching
            self._watching = self._watching or start
        self._notify()
        if start:
            threading.Thread(target=self._watch_exit, daemon=True).start()

    def _watch_exit(self):
        # Captured process objects, never the engine's possibly replaced proc.
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return
            for ticket in pending:
                self.confirm_process_exit(ticket)
            threading.Event().wait(0.25)
