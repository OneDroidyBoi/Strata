"""Admission lifecycle and actual Service.run wrapper, with no native engine."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from serve.request_admission import RequestAdmission, configured_admission
from serve.resource_admission import AdmissionRejected
from serve.server import EngineDied, Service


def config(**changes):
    result = dict(host_capacity_bytes=10000, gpu_capacity_bytes=10000, gpu_id='0',
                  max_requests=1, fixed_host_bytes=10, fixed_gpu_bytes=20,
                  host_request_bytes=30, gpu_request_bytes=40,
                  host_bytes_per_token=2, gpu_bytes_per_token=3,
                  max_context_tokens=100, lookahead_tokens=5)
    result.update(changes)
    return result


def test_reservation_exists_before_factory_and_is_released_after_completion():
    controller = RequestAdmission(config())
    states = []
    def factory():
        states.append(controller.ledger.snapshot())
        yield 'event'
    assert list(controller.run(factory, 10, 20, lambda: True)) == ['event']
    assert states[0]['host_bytes'] == 10 + 30 + 35 * 2
    assert states[0]['gpu_bytes']['0'] == 20 + 40 + 35 * 3
    assert controller.wait_drained()
    assert controller.ledger.snapshot()['host_bytes'] == 10


def test_factory_failure_conservatively_gates_even_if_quiescent_callback_says_true():
    for drained in (True, False):
        controller = RequestAdmission(config())
        def fail():
            raise RuntimeError('factory failed')
        with pytest.raises(RuntimeError, match='factory failed'):
            list(controller.run(fail, 10, 20, lambda: drained))
        assert not controller.wait_drained()
        with pytest.raises(AdmissionRejected, match='draining'):
            controller.acquire(1, 1)


def test_iterator_failure_closes_but_gates_uncertain_native_state():
    controller = RequestAdmission(config())
    closed = []
    def factory():
        try:
            yield 1
            raise RuntimeError('iteration failed')
        finally:
            closed.append(True)
    iterator = controller.run(factory, 10, 20, lambda: bool(closed))
    assert next(iterator) == 1
    with pytest.raises(RuntimeError, match='iteration failed'):
        next(iterator)
    assert closed == [True]
    assert not controller.wait_drained()
    assert controller.ledger.snapshot()['workers']['lane']['draining']


def test_cancellation_closes_inner_before_quiescence_check():
    controller = RequestAdmission(config())
    order = []
    def factory():
        try:
            yield 1
            yield 2
        finally:
            order.append('closed')
    def quiescent():
        order.append('checked')
        return order == ['closed', 'checked']
    iterator = controller.run(factory, 10, 20, quiescent)
    assert next(iterator) == 1
    with pytest.raises(AdmissionRejected):
        controller.acquire(1, 1)
    iterator.close()
    assert order == ['closed', 'checked']
    assert controller.wait_drained()


@pytest.mark.parametrize('quiescent', [lambda: False, lambda: 1])
def test_uncertain_cancellation_keeps_capacity_and_gates_new_requests(quiescent):
    controller = RequestAdmission(config())
    def factory():
        yield 1
        yield 2
    iterator = controller.run(factory, 10, 20, quiescent)
    next(iterator)
    before = controller.ledger.snapshot()['host_bytes']
    iterator.close()
    assert controller.ledger.snapshot()['host_bytes'] == before
    assert not controller.wait_drained()
    with pytest.raises(AdmissionRejected, match='draining'):
        controller.acquire(1, 1)


def test_quiescence_exception_keeps_reservation():
    controller = RequestAdmission(config())
    def factory():
        yield 1
    def fail():
        raise RuntimeError('drain unknown')
    assert list(controller.run(factory, 1, 1, fail)) == [1]
    assert not controller.wait_drained()
    assert controller.ledger.snapshot()['workers']['lane']['draining']


def test_close_failure_retains_reservation():
    controller = RequestAdmission(config())
    def factory():
        try:
            yield 1
            yield 2
        finally:
            raise RuntimeError('close failed')
    iterator = controller.run(factory, 1, 1, lambda: True)
    next(iterator)
    with pytest.raises(RuntimeError, match='close failed'):
        iterator.close()
    assert not controller.wait_drained()
    assert controller.ledger.snapshot()['workers']['lane']['draining']


def test_unlimited_output_reserves_entire_context_and_boundary_rejects():
    for unlimited in (0, -1):
        controller = RequestAdmission(config())
        lease = controller.acquire(10, unlimited)
        assert lease.context_tokens == 100
    controller = RequestAdmission(config())
    with pytest.raises(AdmissionRejected):
        controller.acquire(96, 0)
    with pytest.raises(AdmissionRejected):
        controller.acquire(90, 6)
    assert controller.acquire(90, 5).context_tokens == 100


def service(controller=None):
    instance = object.__new__(Service)
    instance.keep_awake = Mock()
    instance.engine = SimpleNamespace(alive=lambda: True, proc=None)
    instance.request_admission = controller
    return instance


def invoke(instance, ids=(1, 2), max_new=3):
    return instance.run(ids, False, None, max_new, {}, threading.Event())


def test_service_default_path_events_and_arguments_unchanged():
    instance = service()
    events = [('event', 'text'), ('done', {'completion_tokens': 3})]
    instance._run = Mock(return_value=iter(events))
    iterator = invoke(instance)
    assert list(iterator) == events
    assert instance._run.call_count == 1
    args = instance._run.call_args.args
    assert args[:5] == ((1, 2), False, None, 3, {})
    assert args[6] is None
    instance.keep_awake.acquire.assert_called_once()
    instance.keep_awake.release.assert_called_once()


def test_service_overload_rejects_before_run_or_engine_entry():
    controller = RequestAdmission(config())
    held = controller.acquire(1, 1)
    instance = service(controller)
    instance._run = Mock(side_effect=AssertionError('must not execute'))
    with pytest.raises(EngineDied, match='resource admission rejected'):
        list(invoke(instance))
    instance._run.assert_not_called()
    assert controller.ledger.snapshot()['active_reservations'] == 1
    controller.ledger.release(held, iterator_closed=True, work_drained=True)
    instance.keep_awake.release.assert_called_once()


def test_service_context_rejection_never_enters_run():
    instance = service(RequestAdmission(config()))
    instance._run = Mock(side_effect=AssertionError('must not execute'))
    with pytest.raises(EngineDied, match='context budget'):
        list(invoke(instance, ids=tuple(range(96))))
    instance._run.assert_not_called()
    assert instance.request_admission.wait_drained()


def test_service_optin_completion_releases_and_preserves_events():
    controller = RequestAdmission(config())
    instance = service(controller)
    def run(*args):
        assert not controller.wait_drained()
        yield ('event', 'text')
        yield ('done', {'completion_tokens': 1})
    instance._run = run
    assert list(invoke(instance)) == [('event', 'text'), ('done', {'completion_tokens': 1})]
    assert controller.wait_drained()


def test_service_cancel_with_unconfirmed_native_exit_keeps_lease():
    controller = RequestAdmission(config())
    instance = service(controller)
    instance.engine = SimpleNamespace(alive=lambda: False,
                                      proc=SimpleNamespace(poll=lambda: None))
    def run(*args):
        yield ('event', 'text')
        yield ('done', {})
    instance._run = run
    iterator = invoke(instance)
    next(iterator)
    iterator.close()
    assert not controller.wait_drained()
    assert controller.ledger.snapshot()['workers']['lane']['draining']
    instance.keep_awake.release.assert_called_once()


def test_configuration_is_optin_and_rejects_unsupported_topology():
    assert configured_admission({}, [0, 1]) is None
    for cfg, gpus in [({'replicas': 2}, [0]), ({}, [0, 1]),
                      ({'vision': True}, [0]), ({}, [1])]:
        with pytest.raises(ValueError):
            configured_admission(dict(cfg, resource_admission=config()), gpus)
    assert isinstance(configured_admission({'resource_admission': config()}, [0]), RequestAdmission)


@pytest.mark.parametrize('bad', [True, -1, 1.2, None])
def test_invalid_config_does_not_create_controller(bad):
    with pytest.raises(ValueError):
        RequestAdmission(config(host_request_bytes=bad))


def test_rejection_never_calls_factory_or_quiescence_callback():
    controller = RequestAdmission(config(host_capacity_bytes=10))
    factory = Mock(side_effect=AssertionError('must not execute'))
    quiescent = Mock(side_effect=AssertionError('must not execute'))
    with pytest.raises(AdmissionRejected):
        list(controller.run(factory, 1, 1, quiescent))
    factory.assert_not_called()
    quiescent.assert_not_called()
    assert controller.wait_drained()


def test_config_fields_and_snapshot_cannot_be_mutated_through_input():
    cfg = config()
    controller = RequestAdmission(cfg)
    cfg['max_context_tokens'] = 1000
    with pytest.raises(AdmissionRejected):
        controller.acquire(101, 1)
    incomplete = config()
    del incomplete['lookahead_tokens']
    with pytest.raises(ValueError):
        RequestAdmission(incomplete)
    with pytest.raises(ValueError):
        RequestAdmission(config(unknown_field=1))


def test_service_batch_rejected_before_run_and_before_reservation():
    controller = RequestAdmission(config())
    instance = service(controller)
    instance.engine.batch = 2
    instance._run = Mock(side_effect=AssertionError('must not execute'))
    with pytest.raises(EngineDied, match='per-slot drain acknowledgements'):
        list(invoke(instance))
    instance._run.assert_not_called()
    assert controller.wait_drained()
    assert controller.ledger.snapshot()['host_bytes'] == 10
    instance.keep_awake.release.assert_called_once()
