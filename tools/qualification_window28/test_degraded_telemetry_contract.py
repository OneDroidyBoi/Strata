"""Synthetic truth table, not evidence about deployed sensors or safe fan RPM."""
from copy import deepcopy
import unittest

from tools.qualification_window28.degraded_telemetry_contract import fallback_evidence


class DegradedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.policy = {'fan_lower_noncritical_rpm': {'fan-a': 1000, 'fan-b': 1000},
                       'gpu_ids': ('gpu-0', 'gpu-1', 'gpu-2', 'gpu-3'),
                       'cpu_ids': ('cpu-0',), 'max_age_s': 15, 'max_temp_c': 85,
                       'sources': {'fans': 'verified-bmc-reader', 'gpus': 'verified-vm-reader',
                                   'cpu': 'verified-bmc-reader'}}
        self.sample = {
            'fans': {'age_s': 2, 'source': 'verified-bmc-reader', 'rpm': {'fan-a': 2000, 'fan-b': 2000},
                     'controller_state_ok': True, 'fault': False, 'status': {'fan-a': 'ok', 'fan-b': 'ok'}},
            'gpus': {'age_s': 2, 'source': 'verified-vm-reader', 'temperatures': {
                gpu: {'core_c': 60, 'hbm_c': 65} for gpu in self.policy['gpu_ids']}},
            'cpu': {'age_s': 2, 'source': 'verified-bmc-reader', 'temperatures': {'cpu-0': 40}},
        }

    def test_full_independent_evidence_allows_fallback_without_heartbeat(self):
        self.assertTrue(fallback_evidence(self.sample, self.policy)[0])
        before = deepcopy(self.sample)
        fallback_evidence(self.sample, self.policy)
        self.assertEqual(self.sample, before)  # no timestamp rewriting

    def test_each_stale_future_nan_missing_or_replaced_source_stops(self):
        for group in ('fans', 'gpus', 'cpu'):
            for age in (15.01, -1, float('nan'), float('inf'), True, None):
                with self.subTest(group=group, age=age):
                    row = deepcopy(self.sample)
                    row[group]['age_s'] = age
                    self.assertFalse(fallback_evidence(row, self.policy)[0])
            row = deepcopy(self.sample)
            del row[group]
            self.assertFalse(fallback_evidence(row, self.policy)[0])
            row = deepcopy(self.sample)
            row[group]['source'] = 'restarted-or-unbound-reader'
            self.assertFalse(fallback_evidence(row, self.policy)[0])

    def test_commanded_full_speed_or_live_service_is_not_actual_fan_proof(self):
        row = deepcopy(self.sample)
        row['fans'] = {'age_s': 1, 'source': 'verified-bmc-reader',
                       'commanded_percent': 100, 'service_active': True}
        self.assertFalse(fallback_evidence(row, self.policy)[0])
        for key, value in (('controller_state_ok', False), ('fault', True)):
            row = deepcopy(self.sample)
            row['fans'][key] = value
            self.assertFalse(fallback_evidence(row, self.policy)[0])
        for value in (0, 999, 1000, None, float('nan')):
            row = deepcopy(self.sample)
            row['fans']['rpm']['fan-a'] = value
            self.assertFalse(fallback_evidence(row, self.policy)[0])

    def test_each_gpu_core_hbm_and_cpu_overtemp_or_unknown_stops(self):
        for gpu in self.policy['gpu_ids']:
            for field in ('core_c', 'hbm_c'):
                for value in (85, 100, None, 'N/A', float('nan'), float('inf'), -1, True):
                    row = deepcopy(self.sample)
                    row['gpus']['temperatures'][gpu][field] = value
                    self.assertFalse(fallback_evidence(row, self.policy)[0])
        row = deepcopy(self.sample)
        row['cpu']['temperatures']['cpu-0'] = 85
        self.assertFalse(fallback_evidence(row, self.policy)[0])

    def test_incomplete_fan_gpu_cpu_inventory_or_missing_hbm_stops(self):
        for group, field in (('fans', 'rpm'), ('gpus', 'temperatures'), ('cpu', 'temperatures')):
            row = deepcopy(self.sample)
            row[group][field].pop(next(iter(row[group][field])))
            self.assertFalse(fallback_evidence(row, self.policy)[0])
        row = deepcopy(self.sample)
        del row['gpus']['temperatures']['gpu-0']['hbm_c']
        self.assertFalse(fallback_evidence(row, self.policy)[0])
        self.assertFalse(fallback_evidence({}, self.policy)[0])

    def test_unverified_inventory_limits_or_threshold_relaxation_stops(self):
        for key, value in (('fan_lower_noncritical_rpm', {}), ('gpu_ids', ()), ('cpu_ids', ()),
                           ('max_age_s', 16), ('max_temp_c', 86),
                           ('fan_lower_noncritical_rpm', {'fan-a': -1, 'fan-b': -1})):
            policy = deepcopy(self.policy)
            policy[key] = value
            self.assertFalse(fallback_evidence(self.sample, policy)[0])

    def test_actual_bmc_inventory_uses_available_noncritical_boundary(self):
        # Recorded hardware values, with synthetic ages for predicate testing.
        # 100 RPM is BMC's available noncritical boundary, not a validated
        # mechanical cooling minimum. Critical/nonrecoverable are unavailable.
        policy = deepcopy(self.policy)
        policy['fan_lower_noncritical_rpm'] = {f'FAN{i}': 100 for i in range(1, 7)}
        row = deepcopy(self.sample)
        row['fans']['rpm'] = dict(zip(policy['fan_lower_noncritical_rpm'],
                                     (1700, 2200, 3500, 3400, 1700, 1800)))
        row['fans']['status'] = {fan: 'ok' for fan in policy['fan_lower_noncritical_rpm']}
        self.assertTrue(fallback_evidence(row, policy)[0])
        row['gpus']['age_s'] = 20  # actual two historical snapshots were 20s apart
        self.assertFalse(fallback_evidence(row, policy)[0])
        row['gpus']['age_s'] = 2
        row['fans']['status']['FAN3'] = 'critical'
        self.assertFalse(fallback_evidence(row, policy)[0])
        row['fans']['status']['FAN3'] = 'ok'
        row['fans']['rpm']['FAN3'] = 100
        self.assertFalse(fallback_evidence(row, policy)[0])


if __name__ == '__main__':
    unittest.main()
