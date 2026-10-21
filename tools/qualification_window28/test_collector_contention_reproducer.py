"""Diagnostic reproducer for frozen feed 2e062510, not a fixed-feed regression.

Execute only selected pure/collector function definitions with simulated clock,
commands and files. Never import the feed's HTTP server/fork startup code.
Set FROZEN_HOST_FEED to the preserved source path. A different hash is rejected.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

SOURCE_SHA = '2e0625101acc8b93dbf48ef96fc57c25089025813db82b203b7696e9db30ffee'
DEFAULT_SOURCE = '/Users/jacob/Projects/harness/ailan/results/strata-migration-20261018/window31-r2-reviewed-candidate/host-feed-window31-template.py'


def fixture(*, fan_fail=False, probe_seconds=0, service_seconds=0):
    raw = Path(os.environ.get('FROZEN_HOST_FEED', DEFAULT_SOURCE)).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('reproducer requires exact frozen feed ' + SOURCE_SHA)
    class Clock:
        now = 100.0
        def monotonic(self):
            return self.now
        def time(self):
            return 1_800_000_000 + self.now
    clock = Clock()
    calls = []
    def service(prop):
        clock.now += service_seconds
        return {'ActiveState': 'active', 'MainPID': '77', 'NRestarts': '0'}[prop]
    def ipmi(*args):
        sensor = args[-1]
        calls.append(sensor)
        clock.now += probe_seconds
        if sensor.startswith('FAN') and fan_fail:
            raise RuntimeError('BMC lock busy; fan controller has priority')
        value = '38 degrees C' if sensor == 'CPU Temp' else '1700 RPM'
        return f'Sensor Reading : {value}\nStatus : ok\nLower Non-Critical : 100\n'
    ns = {'time': clock, 're': re, 'json': json,
          'hashlib': SimpleNamespace(sha256=lambda _: SimpleNamespace(hexdigest=lambda: 'script-sha')),
          'pathlib': SimpleNamespace(Path=lambda _: SimpleNamespace(read_bytes=lambda: b'script')),
          'os': SimpleNamespace(getpid=lambda: 123), 'server': SimpleNamespace(server_port=1234),
          'FAN_IDS': tuple(f'FAN{i}' for i in range(1, 7)),
          'FAN_SOURCE_SHA256': 'script-sha', 'FEED_TEMPLATE_SHA256': SOURCE_SHA,
          '_service_value': service, '_start_ticks': lambda _: '999',
          '_controller_mode': lambda: ('normal', 0), 'ipmi_safety': ipmi,
          'SAFETY_CACHE': {'sample': None, 'sample_mono': None}}
    names = {'_field', '_reading', '_fan_row', '_refresh_safety', 'safety_snapshot'}
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-feed-functions>', 'exec'), ns)
    return ns, clock, calls


class FrozenCollectorDefects(unittest.TestCase):
    def test_missing_optional_fans_break_snapshot_despite_valid_cpu(self):
        ns, _, calls = fixture(fan_fail=True)
        with self.assertRaises(TypeError):
            ns['safety_snapshot']()
        self.assertEqual(calls[-1], 'CPU Temp')
        self.assertEqual(ns['SAFETY_CACHE']['sample']['cpu']['temperatures'], {'CPU Temp': 38.0})
        self.assertIsNone(ns['SAFETY_CACHE']['sample']['fans']['age_s'])

    def test_optional_timeouts_exhaust_budget_before_required_cpu(self):
        ns, _, calls = fixture(fan_fail=True, probe_seconds=2, service_seconds=1)
        with self.assertRaisesRegex(TimeoutError, 'six fan sensors'):
            ns['_refresh_safety']()
        self.assertEqual(calls, [f'FAN{i}' for i in range(1, 7)])
        self.assertNotIn('CPU Temp', calls)

    def test_stale_optional_fan_invalidates_fresh_cpu_snapshot(self):
        ns, clock, _ = fixture()
        cached = ns['_refresh_safety']()
        cached['fans']['age_s'] = 16
        cached['fans']['sensor_age_s']['FAN1'] = 16
        ns['SAFETY_CACHE'].update(sample=cached, sample_mono=clock.now)
        with self.assertRaisesRegex(RuntimeError, 'original BMC measurement age'):
            ns['safety_snapshot']()
        self.assertEqual(cached['cpu']['age_s'], 0)

    def test_refresh_failure_discards_still_fresh_cached_cpu(self):
        ns, clock, _ = fixture()
        cached = ns['safety_snapshot']()
        clock.now += 9
        def busy():
            raise RuntimeError('BMC lock busy; fan controller has priority')
        ns['_refresh_safety'] = busy
        with self.assertRaisesRegex(RuntimeError, 'BMC lock busy'):
            ns['safety_snapshot']()
        self.assertLess(cached['cpu']['age_s'] + 9, 15)


if __name__ == '__main__':
    unittest.main()
