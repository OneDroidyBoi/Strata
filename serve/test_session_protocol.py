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
