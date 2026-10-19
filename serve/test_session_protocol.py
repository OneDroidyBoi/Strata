import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from serve.session_protocol import SessionProtocol, SessionProtocolError
from serve.server import EngineDied, SessionRefused, StrataEngine


def engine(*lines):
    result = SimpleNamespace(session_lifecycle=True, alive=lambda: True,
                             _send=Mock(), lines=queue.Queue(), proc=Mock())
    for line in lines:
        result.lines.put(line)
    return result


def test_complete_generation_bound_inventory():
    e = engine('INVENTORY proc1-3 2', 'SESSION_ENTRY live live 40',
               'SESSION_ENTRY parked-8 parked 32', 'INVENTORY_END proc1-3')
    assert SessionProtocol(e).inventory() == {'generation': 'proc1-3', 'sessions': [
        {'id': 'live', 'kind': 'live', 'tokens': 40}, {'id': 'parked-8', 'kind': 'parked', 'tokens': 32}]}
    e._send.assert_called_once_with('INVENTORY')


def test_batch_inventory_contains_live_parked_and_separate_slot_sessions():
    e = engine('INVENTORY g 4', 'SESSION_ENTRY live live 8',
               'SESSION_ENTRY parked-1 parked 9', 'SESSION_ENTRY slot-0 slot 12',
               'SESSION_ENTRY slot-1 slot 20', 'INVENTORY_END g')
    e.batch, e.session_batch_export = 2, True
    result = SessionProtocol(e).inventory()
    assert [(entry['id'], entry['tokens']) for entry in result['sessions']] == [
        ('live', 8), ('parked-1', 9), ('slot-0', 12), ('slot-1', 20)]


def test_batch_capability_requires_native_advertisement_and_precision_guard():
    e = StrataEngine.__new__(StrataEngine)
    e.info = {'session_export': '1', 'session_export_batch': '1'}
    e.precision_policy_report = None
    assert not e.session_batch_export
    e.precision_policy_report = {'enforced': True}
    assert e.session_batch_export
    del e.info['session_export_batch']
    assert not e.session_batch_export


@pytest.mark.parametrize('sid,advertised', [('slot-2', True), ('slot--1', True),
                                         ('slot-01', True), ('slot-0', False)])
def test_batch_slot_requires_capability_and_valid_index(sid, advertised):
    e = engine('INVENTORY g 1', f'SESSION_ENTRY {sid} slot 12', 'INVENTORY_END g')
    e.batch, e.session_batch_export = 2, advertised
    with pytest.raises(SessionProtocolError, match='batch slot'):
        SessionProtocol(e).inventory()
    e.proc.kill.assert_not_called()


@pytest.mark.parametrize('lines', [
    ['INVENTORY g 2', 'SESSION_ENTRY live live 3', 'INVENTORY_END g'],
    ['INVENTORY g 2', 'SESSION_ENTRY live live 3', 'SESSION_ENTRY live live 3'],
    ['INVENTORY g 0', 'INVENTORY_END other'], ['INVENTORY g 1025'],
    ['INVENTORY g 1', 'SESSION_ENTRY live live -1'], [None],
])
def test_inventory_failure_blocks_protocol_without_killing(lines):
    e = engine(*lines)
    with pytest.raises(SessionProtocolError):
        SessionProtocol(e).inventory()
    assert e._session_protocol_failed
    e.proc.kill.assert_not_called()
    with pytest.raises(SessionProtocolError):
        SessionProtocol(e).inventory()
    e._send.assert_called_once()


def test_export_progress_and_path_with_spaces():
    e = engine('SWAIT fingerprint 30', 'SESSION 0 16', 'SESSION 16 16', 'SAVED 8 16 2.5')
    assert SessionProtocol(e).export('g-1', 'parked-2', '/tmp/state one.bin') == {
        'tokens': 8, 'bytes': 16, 'ms': 2.5}
    e._send.assert_called_once_with('EXPORT g-1 parked-2 /tmp/state one.bin')


@pytest.mark.parametrize('line', ['SAVED 1 2 nan', 'SAVED -1 2 3', 'SWAIT flush 999999',
                                 'SESSION 3 2', 'SERR nonsense 0 reason', 'ERR bad'])
def test_export_malformed_response_keeps_engine_running(line):
    e = engine(line)
    with pytest.raises((SessionProtocolError, ValueError)):
        SessionProtocol(e).export('g', 'live', '/tmp/test')
    assert e._session_protocol_failed
    e.proc.kill.assert_not_called()


def test_clean_native_refusal_does_not_poison_framing():
    e = engine('SERR invalid 0 stale inventory', 'INVENTORY next 0', 'INVENTORY_END next')
    with pytest.raises(SessionRefused, match='stale inventory'):
        SessionProtocol(e).export('old', 'live', '/tmp/test')
    assert SessionProtocol(e).inventory()['generation'] == 'next'


def test_timeout_and_command_injection():
    e = engine()
    with pytest.raises(ValueError):
        SessionProtocol(e).export('g', 'live', '/tmp/a\nQUIT')
    e._send.assert_not_called()
    with pytest.raises(SessionProtocolError, match='timed out'):
        SessionProtocol(e, timeout_s=0.001).inventory()
    e.proc.kill.assert_not_called()


def test_failed_protocol_cannot_fall_back_to_legacy_restore():
    e = engine()
    with pytest.raises(SessionProtocolError):
        SessionProtocol(e, timeout_s=0.001).inventory()
    with pytest.raises(EngineDied, match='protocol is blocked'):
        StrataEngine.session_file(e, 'restore', '/tmp/checkpoint')
    e.proc.stdin.write.assert_not_called()
    e.proc.kill.assert_not_called()


def restore_engine(*lines):
    e = engine(*lines)
    e.batch, e.session_batch_restore, e.max_context = 2, True, 16
    return e


def test_restore_slot_collects_bounded_prefix_for_routing():
    e = restore_engine('SWAIT validate 30', 'SESSION 0 16', 'SESSION 16 16',
                       'SLOT_IDS 1 10,20', 'SLOT_IDS 1 30', 'SLOT_RESTORED 1 3 16 1.5')
    assert SessionProtocol(e).restore_slot(1, '/tmp/state one.bin') == {
        'tokens': 3, 'bytes': 16, 'ms': 1.5, 'ids': [10, 20, 30]}
    e._send.assert_called_once_with('RESTORE_SLOT 1 /tmp/state one.bin')


@pytest.mark.parametrize('lines', [
    ['SLOT_IDS 0 1'], ['SLOT_IDS 1 -1'], ['SLOT_IDS 1 2147483648'],
    ['SLOT_IDS 1 ' + ','.join(['1'] * 17)], ['SLOT_IDS 1 1,,2'],
    ['SLOT_IDS 1 1', 'SLOT_RESTORED 1 2 16 1'],
    ['SLOT_RESTORED 0 0 16 1'], ['SLOT_RESTORED 1 0 16 nan'],
    ['SLOT_IDS 1 1', 'SESSION 1 2'], [None], ['FATAL transfer failed'],
])
def test_bad_restore_reply_blocks_without_killing_other_slots(lines):
    e = restore_engine(*lines)
    with pytest.raises((SessionProtocolError, ValueError)):
        SessionProtocol(e).restore_slot(1, '/tmp/state')
    assert e._session_protocol_failed
    e.proc.kill.assert_not_called()


@pytest.mark.parametrize('slot,path', [(True, '/tmp/a'), (-1, '/tmp/a'), (2, '/tmp/a'),
                                      (0, '/tmp/a\nQUIT')])
def test_restore_slot_invalid_arguments_send_nothing(slot, path):
    e = restore_engine()
    with pytest.raises(ValueError):
        SessionProtocol(e).restore_slot(slot, path)
    e._send.assert_not_called()


def test_restore_slot_requires_explicit_capability():
    e = restore_engine()
    e.session_batch_restore = False
    with pytest.raises(ValueError):
        SessionProtocol(e).restore_slot(0, '/tmp/a')
    e._send.assert_not_called()
    e = StrataEngine.__new__(StrataEngine)
    e.info = {'session_export': '1', 'session_export_batch': '1', 'session_restore_batch': '1'}
    e.precision_policy_report = {'enforced': True}
    assert e.session_batch_restore
    del e.info['session_restore_batch']
    assert not e.session_batch_restore


def test_restore_clean_refusal_leaves_protocol_aligned():
    e = restore_engine('SERR invalid 0 occupied', 'INVENTORY g 0', 'INVENTORY_END g')
    with pytest.raises(SessionRefused):
        SessionProtocol(e).restore_slot(0, '/tmp/a')
    assert SessionProtocol(e).inventory()['sessions'] == []
