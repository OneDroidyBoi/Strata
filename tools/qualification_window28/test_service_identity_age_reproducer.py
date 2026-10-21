"""Diagnostic for frozen 7ac134c3 service identity age, not a fixed-code test."""
import ast
import hashlib
import os
from pathlib import Path
import unittest

from tools.qualification_window28.test_collector_contention_reproducer import fixture

DEFAULT_SOURCE = '/Users/jacob/Projects/harness/ailan/results/strata-migration-20261018/window34-run-cadence-candidate/host-feed-window31-template.py'
SOURCE_SHA = '7ac134c35f7273ebe2d2fb571c772d6c5cb7118bbd37866a765527f5dd3ace8d'


class ServiceIdentityAgeDiagnostic(unittest.TestCase):
    def test_service_age_can_expire_while_cpu_and_fan_measurements_remain_fresh(self):
        raw = Path(os.environ.get('FROZEN_IDENTITY_HOST_FEED', DEFAULT_SOURCE)).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_SHA)
        # Simulate three 1s service queries and seven 1.5s BMC sensor queries.
        # These are diagnostic timings, not measurements of the incident.
        ns, _, _ = fixture(probe_seconds=1.5, service_seconds=1)
        node = next(n for n in ast.parse(raw).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_refresh_safety')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-identity-collector>', 'exec'), ns)
        result = ns['_refresh_safety']()
        transit = 2
        self.assertEqual(result['fans']['service_state_age_s'], 13.5)
        self.assertGreater(result['fans']['service_state_age_s'] + transit, 15)
        self.assertLessEqual(result['cpu']['sensor_age_s'] + transit, 15)
        self.assertTrue(all(age + transit <= 15 for age in result['fans']['sensor_age_s'].values()))
        self.assertLessEqual(result['fans']['controller_state_age_s'] + transit, 15)


if __name__ == '__main__':
    unittest.main()
