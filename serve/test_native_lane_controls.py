"""Sticky checkpoint ownership protects ordinary server controls."""
import threading
from unittest.mock import Mock

import pytest

from serve.server import GpuBusy, ModelBusy, Service


def gated_service():
    service = Service.__new__(Service)
    service.native_lane_gated = True
    service.fifo = threading.Lock()
    service.engine = Mock()
    service.loaded = lambda: False
    return service


def test_gated_load_cannot_restart_retired_lane():
    service = gated_service()
    with pytest.raises(GpuBusy, match='owner recovery'):
        service.load()
    with service.fifo, pytest.raises(GpuBusy, match='owner recovery'):
        service.ensure_loaded()
    service.engine.restart.assert_not_called()


def test_gated_unload_cannot_discard_state_before_checkpoint():
    service = gated_service()
    assert service.unload() == 'busy'
    assert service.unload(idle_for=0) == 'busy'
    service.engine.unload.assert_not_called()
    assert not service.fifo.locked()


def test_gated_vram_control_cannot_resize_retired_or_checkpointing_lane():
    service = gated_service()
    with pytest.raises(ModelBusy, match='owns resource controls'):
        service.vram(1024)
    service.engine.vram.assert_not_called()


@pytest.mark.parametrize('action', ['save', 'restore'])
def test_gated_slot_commands_cannot_mutate_inventory(action):
    service = gated_service()
    status, _ = service.slot_action('0', action, 'state.bin')
    assert status == 503
    service.engine.session_file.assert_not_called()
