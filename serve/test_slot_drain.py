"""Slot terminal evidence and receipt tests. Fake pipes only; no GPU/model."""
import os
import io
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import unittest

from serve.drain_receipt import DrainReceipt, matching_bdone
from serve.server import StrataEngine


class Process:
    code = None
    def __init__(self):
        self.stdin = io.StringIO()
    def poll(self):
        return self.code


def wait_for(predicate, seconds=2):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.005)
    return predicate()


class SlotDrainTest(unittest.TestCase):
    def engine(self):
        engine = object.__new__(StrataEngine)
        engine.gen = 3
        engine.proc = Process()
        engine.slot_q = [queue.Queue(), queue.Queue()]
        engine.slot_busy = [True, True]
        engine.slot_held = [[], []]
        engine.slot_releasing = set()
        engine.slot_claims = set()
        engine._orphan_seen = {}
        engine.slot_cv = threading.Condition()
        engine.wlock = threading.Lock()
        engine.sent = []
        engine._send = engine.sent.append
        return engine

    def start(self, e, slot=0):
        receipt = DrainReceipt()
        ticket = receipt.begin(e.gen, e.proc, slot)
        e._release_slot_when_done(slot, [10, 11], receipt=receipt, ticket=ticket,
                                  process=e.proc, generation=e.gen)
        receipt.seal()
        return receipt, ticket

    def test_delayed_matching_bdone_only_releases_its_slot(self):
        e = self.engine()
        receipt, ticket = self.start(e)
        self.assertFalse(receipt.confirmed())
        e.slot_q[0].put('BDONE 1 2 cancel 1.0')
        self.assertTrue(wait_for(receipt.failed))
        self.assertFalse(receipt.confirmed())
        self.assertTrue(e.slot_busy[0])
        e.slot_q[0].put('BDONE 0 2 cancel 1.0')
        self.assertTrue(wait_for(lambda: receipt.confirmed() and not e.slot_busy[0]))
        self.assertTrue(e.slot_busy[1])
        self.assertFalse(receipt.acknowledge(ticket, 3, 0))

    def test_eof_does_not_release_until_captured_process_exits(self):
        e = self.engine()
        old = e.proc
        receipt, _ = self.start(e)
        e.slot_q[0].put(None)
        self.assertTrue(wait_for(receipt.failed))
        self.assertFalse(receipt.confirmed())
        self.assertTrue(e.slot_busy[0])
        e.proc = Process()  # successor exit is not evidence about the old one
        e.gen += 1
        e.proc.code = 0
        time.sleep(.03)
        self.assertFalse(receipt.confirmed())
        old.code = 0
        self.assertTrue(wait_for(receipt.confirmed))

    def test_timeout_keeps_slot_owned_and_later_ack_works(self):
        e = self.engine()
        # Advance only the waiter's deadline; never wait ten minutes in a test.
        e.SLOT_DRAIN_TIMEOUT_S = 0
        receipt, _ = self.start(e)
        self.assertTrue(wait_for(receipt.failed))
        self.assertTrue(e.slot_busy[0])
        self.assertIn(0, e.slot_releasing)
        with e.slot_cv:
            self.assertEqual(e._reap_orphan_slots(), [])
        e.slot_q[0].put('BDONE 0 1 cancel 5')
        self.assertTrue(wait_for(receipt.confirmed))

    def test_receipt_all_legs_and_seal_are_required(self):
        receipt, process = DrainReceipt(), Process()
        first, second = receipt.begin(1, process, 0), receipt.begin(1, process, 1)
        callbacks = []
        receipt.add_done_callback(lambda r: callbacks.append(r.confirmed()))
        self.assertFalse(receipt.acknowledge(first, 2, 0))
        self.assertFalse(receipt.acknowledge(first, 1, 1))
        receipt.acknowledge(first, 1, 0)
        receipt.seal()
        self.assertFalse(receipt.confirmed())
        receipt.acknowledge(second, 1, 1)
        self.assertTrue(receipt.confirmed())
        self.assertEqual(callbacks, [True])
        with self.assertRaises(RuntimeError):
            receipt.begin(1, process)

    def test_poll_exception_retains_ticket_and_reports_failure(self):
        class Broken:
            def poll(self):
                raise OSError('unknown process state')
        receipt = DrainReceipt()
        ticket = receipt.begin(1, Broken())
        self.assertFalse(receipt.confirm_process_exit(ticket))
        self.assertTrue(receipt.failed())
        receipt.acknowledge(ticket, 1)
        receipt.seal()
        self.assertTrue(receipt.confirmed())

    def test_bdone_validation(self):
        for text in ('BDONE 0 2 cancel 1', 'BDONE 0 2 nonfinite 0'):
            self.assertTrue(matching_bdone(text, 0))
        for text in ('BDONE 1 2 cancel 1', 'BDONE 0 -1 cancel 1',
                     'BDONE 0 2 cancel nan', 'BDONE 0 2 cancel -1', 'BDONE 0 2 nonsense 1'):
            self.assertFalse(matching_bdone(text, 0))

    def test_late_cleanup_after_restart_never_touches_successor(self):
        e = self.engine()
        old, generation = e.proc, e.gen
        receipt = DrainReceipt()
        ticket = receipt.begin(generation, old, 0)
        e.proc, e.gen = Process(), generation + 1
        e.slot_q[0].put('BDONE 0 1 length 1')
        e._release_slot_when_done(0, [1], receipt=receipt, ticket=ticket,
                                  process=old, generation=generation)
        receipt.seal()
        self.assertTrue(receipt.failed())
        self.assertFalse(receipt.confirmed())
        self.assertEqual(e.proc.stdin.getvalue(), '')
        self.assertEqual(old.stdin.getvalue(), '')
        self.assertEqual(e.slot_q[0].qsize(), 1)
        self.assertTrue(e.slot_busy[0])
        self.assertEqual(e.slot_releasing, set())
        old.code = 0
        self.assertTrue(wait_for(receipt.confirmed))

    def test_restart_during_queue_capture_is_detected(self):
        e = self.engine()
        old, generation = e.proc, e.gen
        receipt = DrainReceipt()
        ticket = receipt.begin(generation, old, 0)
        successor_queue = queue.Queue()
        successor_queue.put('BDONE 0 1 length 1')
        class RestartingQueues(list):
            def __getitem__(self, index):
                e.proc, e.gen = Process(), generation + 1
                e.slot_q = [successor_queue]
                e.slot_busy, e.slot_held, e.slot_releasing = [True], [[99]], set()
                return successor_queue
        e.slot_q = RestartingQueues(e.slot_q)
        e._release_slot_when_done(0, [1], receipt=receipt, ticket=ticket,
                                  process=old, generation=generation)
        receipt.seal()
        self.assertTrue(receipt.failed())
        self.assertFalse(receipt.confirmed())
        self.assertEqual(successor_queue.qsize(), 1)
        self.assertEqual(e.slot_busy, [True])
        self.assertEqual(e.slot_held, [[99]])
        self.assertEqual(e.proc.stdin.getvalue(), '')
        old.code = 0
        self.assertTrue(wait_for(receipt.confirmed))

    def test_restart_during_stop_write_uses_only_captured_process(self):
        e = self.engine()
        old, generation, old_queue = e.proc, e.gen, e.slot_q[0]
        receipt = DrainReceipt()
        ticket = receipt.begin(generation, old, 0)
        successor = Process()
        successor_queue = queue.Queue()
        successor_queue.put('BDONE 0 1 length 1')
        class RestartingPipe(io.StringIO):
            def write(self, text):
                e.proc, e.gen = successor, generation + 1
                e.slot_q, e.slot_busy, e.slot_held = [successor_queue], [True], [[99]]
                e.slot_releasing = set()
                return super().write(text)
        old.stdin = RestartingPipe()
        e._release_slot_when_done(0, [1], receipt=receipt, ticket=ticket,
                                  process=old, generation=generation)
        receipt.seal()
        self.assertEqual(old.stdin.getvalue(), 'BSTOP 0\n')
        self.assertEqual(successor.stdin.getvalue(), '')
        old_queue.put('BDONE 0 1 cancel 1')
        self.assertTrue(wait_for(receipt.confirmed))
        self.assertEqual(successor_queue.qsize(), 1)
        self.assertEqual(e.slot_busy, [True])
        self.assertEqual(e.slot_held, [[99]])


FAKE = r'''
import sys,time
print('INFO batch_slots=2',flush=True)
print('READY 4096 stop',flush=True)
for line in sys.stdin:
 if line.startswith('QUIT'): break
 if line.startswith('BGEN '):
  slot=line.split()[1]
  print('T 17',flush=True)
  print('DONE 1 1 1 1 length 0 0 0',flush=True)
  print('BADM '+slot+' 1',flush=True)
  print('BT '+slot+' 19',flush=True)
  if '--auto' in sys.argv: print('BDONE '+slot+' 2 length 2',flush=True)
 elif line.startswith('BSTOP '):
  time.sleep(.1)
  print('BDONE '+line.split()[1]+' 2 cancel 2',flush=True)
 elif line.startswith('GEN '):
  print('T 23',flush=True)
  print('DONE 1 1 1 1 stop 0 0 0',flush=True)
'''


class FakePipeDrainTest(unittest.TestCase):
    def engine(self, extra=()):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'native.py'
        path.write_text('#!' + str(Path(sys.executable).resolve()) + '\n' + FAKE)
        path.chmod(0o700)
        e = StrataEngine(str(path), ['--batch', '2', *extra])
        self.addCleanup(e.close)
        return e

    @unittest.skipIf(os.name == 'nt', 'test stand-in uses a POSIX shebang')
    def test_real_generator_close_waits_for_delayed_bdone(self):
        e = self.engine()
        e.slot_busy[1] = True  # force BGEN without another model/request
        receipt = DrainReceipt()
        with e.track_drain(receipt):
            gen = e.generate_batched([1], 4, {}, threading.Event())
            self.assertEqual(next(gen), 17)
            self.assertEqual(next(gen), 19)
            gen.close()
        receipt.seal()
        self.assertFalse(receipt.confirmed())
        self.assertTrue(e.slot_busy[0])
        self.assertTrue(wait_for(receipt.confirmed))
        self.assertTrue(wait_for(lambda: not e.slot_busy[0]))

    @unittest.skipIf(os.name == 'nt', 'test stand-in uses a POSIX shebang')
    def test_solo_leg_inside_batch_mode_confirms_only_when_closed(self):
        e = self.engine()
        receipt = DrainReceipt()
        with e.track_drain(receipt):
            self.assertEqual(list(e.generate_batched([1], 4, {}, threading.Event())), [23])
        self.assertFalse(receipt.confirmed())
        receipt.seal()
        self.assertTrue(receipt.confirmed())

    @unittest.skipIf(os.name == 'nt', 'test stand-in uses a POSIX shebang')
    def test_normal_slot_completion_and_multiple_generate_calls(self):
        e = self.engine(['--auto'])
        e.slot_busy[1] = True
        receipt = DrainReceipt()
        with e.track_drain(receipt):
            for _ in range(2):
                self.assertEqual(list(e.generate_batched([1], 2, {}, threading.Event())), [17, 19])
                self.assertFalse(receipt.confirmed())
        receipt.seal()
        self.assertTrue(receipt.confirmed())
        self.assertFalse(e.slot_busy[0])

    @unittest.skipIf(os.name == 'nt', 'test stand-in uses a POSIX shebang')
    def test_cancelled_slot_stays_reserved_until_stop_ack(self):
        e = self.engine()
        e.slot_busy[1] = True
        receipt, cancel = DrainReceipt(), threading.Event()
        with e.track_drain(receipt):
            gen = e.generate_batched([1], 4, {}, cancel)
            self.assertEqual(next(gen), 17)
            cancel.set()
            self.assertEqual(list(gen), [])
        receipt.seal()
        self.assertTrue(receipt.confirmed())


if __name__ == '__main__':
    unittest.main()
