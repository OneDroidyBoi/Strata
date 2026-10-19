"""Owner-only slot routing with CPU protocol stand-ins; no GPU/model/network."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from serve.drain_receipt import DrainReceipt
from serve.server import StrataEngine


FAKE = r'''
import sys
print('INFO batch_slots=2 session_export=1 session_export_batch=1 session_restore_batch=1 session_release_batch=1 session_route_batch=1', flush=True)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
 if line.startswith('QUIT'): break
 if line.startswith('BGEN '):
  fields=line.split()
  slot=fields[1]
  prompt=fields[-1].split(',')
  first=prompt[0]
  if first=='702':
   print('PP 1', flush=True)
   command=sys.stdin.readline()
   print('ERR lane_owned cancelled partial prompt', flush=True)
   continue
  eos=first=='701'
  short=int(fields[2])==1 or eos
  print('T '+('248046' if eos else '17'), flush=True)
  print('DONE 1 1 1 1 '+('stop' if eos else 'length')+' 0 0 0', flush=True)
  if 'lane_owned=1' in fields and first!='704':
   count=len(prompt)-(1 if first=='703' else 0)
   proofslot='0' if first=='705' else slot
   print('BOWN '+proofslot+' '+str(count), flush=True)
  print('BADM '+slot+(' 0' if short else ' 1'), flush=True)
  if not short:
   print('BT '+slot+' 19', flush=True)
   print('BDONE '+slot+' 2 length 2', flush=True)
 elif line.startswith('GEN '):
  print('T 23', flush=True)
  print('DONE 1 1 1 1 stop 0 0 0', flush=True)
'''



def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(.005)
    return predicate()


@pytest.fixture
def engine(tmp_path):
    if os.name == 'nt':
        pytest.skip('stand-in uses a POSIX shebang')
    program = tmp_path / 'native.py'
    program.write_text('#!' + str(Path(sys.executable).resolve()) + '\n' + FAKE)
    program.chmod(0o700)
    instance = StrataEngine(str(program), ['--batch', '2'])
    instance.precision_policy_report = {"enforced": True}
    instance.sent = []
    actual_send = instance._send
    def send(command):
        instance.sent.append(command)
        return actual_send(command)
    instance._send = send
    try:
        yield instance
    finally:
        instance.close()


def test_bound_slot_bypasses_prefix_choice_solo_and_promotion(engine):
    engine.slot_held[0] = [1]  # Unbound prefix matching would favor slot zero.
    engine._may_go_solo = lambda *args: True  # Even if performance heuristic wants solo.
    receipt = DrainReceipt()
    with engine.bound_slot(1), engine.track_drain(receipt):
        for _ in range(2):  # Internal continuation calls retain owner routing.
            assert list(engine.generate([1, 2], 100, {}, threading.Event())) == [17, 19]
    receipt.seal()
    assert receipt.confirmed()
    assert len([cmd for cmd in engine.sent if cmd.startswith('BGEN 1 ')]) == 2
    assert not any(cmd.startswith(('GEN ', 'BGEN 0 ', 'BSTOP ')) for cmd in engine.sent)
    assert engine.slot_held[0] == [1]
    assert engine._bound_slot_local.slot is None


def test_bound_busy_slot_waits_even_with_other_slot_free(engine):
    with engine.slot_cv:
        engine.slot_busy[1] = True
        engine.slot_claims.add(1)
    receipt = DrainReceipt()
    def run():
        with engine.bound_slot(1), engine.track_drain(receipt):
            result = list(engine.generate([10], 2, {}, threading.Event()))
        receipt.seal()
        return result
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        try:
            assert wait_for(lambda: engine.admitting == 1)
            assert not future.done()
            assert engine.sent == []
            assert not engine.slot_busy[0]
            with engine.slot_cv:
                engine.slot_claims.remove(1)
                engine.slot_busy[1] = False
                engine.slot_cv.notify_all()
            assert future.result(timeout=2) == [17, 19]
            assert receipt.confirmed()
            assert engine.sent[0].startswith('BGEN 1 ')
        finally:
            with engine.slot_cv:
                engine.slot_claims.discard(1)
                engine.slot_busy[1] = False
                engine.slot_cv.notify_all()


def test_cancel_while_bound_slot_busy_sends_no_native_work(engine):
    cancel, receipt = threading.Event(), DrainReceipt()
    with engine.slot_cv:
        engine.slot_busy[1] = True
        engine.slot_claims.add(1)
    def run():
        with engine.bound_slot(1), engine.track_drain(receipt):
            result = list(engine.generate([10], 2, {}, cancel))
        receipt.seal()
        return result
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        assert wait_for(lambda: engine.admitting == 1)
        cancel.set()
        with engine.slot_cv:
            engine.slot_cv.notify_all()
        assert future.result(timeout=2) == []
    assert engine.sent == []
    assert receipt.confirmed()
    assert engine.slot_busy[1]  # This request did not own or release the busy slot.


def test_unbound_default_still_takes_solo_path(engine):
    assert list(engine.generate([10], 4, {}, threading.Event())) == [23]
    assert engine.sent[0].startswith('GEN ')


def test_nested_binding_restores_previous_and_thread_scopes_are_isolated():
    engine = object.__new__(StrataEngine)
    engine.batch, engine.slot_groups = 2, 1
    engine.info = dict(session_export=1, session_export_batch=1,
                       session_restore_batch=1, session_release_batch=1, session_route_batch=1)
    engine.precision_policy_report = {"enforced": True}
    with engine.bound_slot(0):
        assert engine._bound_slot_local.slot == 0
        with engine.bound_slot(1):
            assert engine._bound_slot_local.slot == 1
        assert engine._bound_slot_local.slot == 0
        def other():
            assert getattr(engine._bound_slot_local, 'slot', None) is None
            with engine.bound_slot(1):
                return engine._bound_slot_local.slot
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(other).result(timeout=1) == 1
        assert engine._bound_slot_local.slot == 0
    assert engine._bound_slot_local.slot is None
    with pytest.raises(RuntimeError):
        with engine.bound_slot(1):
            raise RuntimeError('scope failed')
    assert engine._bound_slot_local.slot is None


@pytest.mark.parametrize('slot', [True, False, -1, 2, 1.0, None, '0'])
def test_invalid_bound_slot_fails_before_native_work(slot):
    engine = object.__new__(StrataEngine)
    engine.batch, engine.slot_groups = 2, 1
    engine.info = dict(session_export=1, session_export_batch=1,
                       session_restore_batch=1, session_release_batch=1, session_route_batch=1)
    engine.precision_policy_report = {"enforced": True}
    with pytest.raises(ValueError):
        with engine.bound_slot(slot):
            pytest.fail('invalid scope entered')


def test_grouped_or_unbatched_lane_rejects_binding():
    engine = object.__new__(StrataEngine)
    for batch, groups in [(0, 1), (2, 2)]:
        engine.batch, engine.slot_groups = batch, groups
        with pytest.raises(ValueError):
            with engine.bound_slot(0):
                pytest.fail('unsupported scope entered')


def test_binding_cannot_fall_back_to_unbatched_after_lane_change(engine):
    from serve.server import EngineDied
    with engine.bound_slot(1):
        engine.batch = 0
        with pytest.raises(EngineDied, match='no longer batched'):
            list(engine.generate([1], 1, {}, threading.Event()))
    assert engine.sent == []


@pytest.mark.parametrize('first,max_new,token', [(700, 1, 17), (701, 100, 248046)])
@pytest.mark.parametrize('close_early', [False, True])
def test_owned_badm_zero_publishes_latest_prefix_even_on_early_close(engine, first, max_new, token, close_early):
    prompt = [first, 8, 9]
    engine.slot_held[1] = [99, 98]  # A different agent's old state must not survive.
    receipt = DrainReceipt()
    with engine.bound_slot(1), engine.track_drain(receipt):
        iterator = engine.generate(prompt, max_new, {}, threading.Event())
        if close_early:
            assert next(iterator) == token
            iterator.close()  # BOWN/BADM consumed by _drain_control.
        else:
            assert list(iterator) == [token]
    receipt.seal()
    assert receipt.confirmed()
    assert engine.slot_held[1] == prompt
    assert ' lane_owned=1 ' in engine.sent[0]
    assert not engine.slot_busy[1]


@pytest.mark.parametrize('first', [703, 704, 705])
def test_missing_wrong_count_or_wrong_slot_publication_proof_quarantines(engine, first):
    from serve.server import EngineDied
    receipt = DrainReceipt()
    with engine.bound_slot(1), engine.track_drain(receipt):
        with pytest.raises(EngineDied, match='retained prompt prefix'):
            list(engine.generate([first, 8], 1, {}, threading.Event()))
    receipt.seal()
    assert receipt.failed()
    assert not receipt.confirmed()
    assert engine._session_protocol_failed
    assert engine.slot_held[1] == []
    assert engine.slot_busy[1]
    assert 1 in engine.slot_releasing


def test_owned_cancelled_partial_prefill_never_claims_full_prefix(engine):
    cancel, receipt = threading.Event(), DrainReceipt()
    cancel.set()
    with engine.bound_slot(1), engine.track_drain(receipt):
        with pytest.raises(ValueError, match='cancelled partial prompt'):
            list(engine.generate([702, 8, 9], 1, {}, cancel))
    receipt.seal()
    assert receipt.failed()
    assert not receipt.confirmed()
    assert engine._session_protocol_failed
    assert engine.slot_held[1] == []
    assert engine.sent[0].startswith('BGEN 1 ')
    assert 'STOP' in engine.sent


def test_old_native_without_route_capability_rejects_owner_scope(engine):
    engine.info.pop('session_route_batch')
    with pytest.raises(ValueError, match='routing capability'):
        with engine.bound_slot(1):
            pytest.fail('old binary accepted bound routing')
    assert engine.sent == []


def test_http_sampling_field_cannot_enable_owned_native_path(engine):
    assert list(engine.generate([1], 1, {'lane_owned': 1}, threading.Event())) == [17]
    assert not any('lane_owned=' in command for command in engine.sent)
