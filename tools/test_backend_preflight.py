"""Offline integration of real synthetic GGUF audit and backend capabilities."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.backend_preflight import build_preflight
from tools.test_model_manifest import fixture


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = fixture(self.root / 'target.gguf', metadata=[('general.architecture', 8, 'qwen4exp')])
        self.ple = fixture(self.root / 'ple.gguf', [('per_layer_token_embd.weight', [32], 8, 0, 34)])

    def report(self, **kwargs):
        return build_preflight({'target': [self.target], 'ple': [self.ple]},
                               ple_mode='external', draft_mode='disabled', **kwargs)

    def test_q8_inventory_passes_without_claiming_launch_support(self):
        report = self.report()
        self.assertTrue(report['offline_checks_passed'], report)
        self.assertFalse(report['launch_qualified'])
        self.assertFalse(report['capabilities']['durable_checkpoint'])
        self.assertEqual(report['artifact_manifest']['scope']['draft'], 'not_provided')

    def test_low_bit_draft_rejected_even_with_q8_target(self):
        draft = fixture(self.root / 'draft.gguf', [('draft_expert', [64], 42, 0, 18)])
        report = build_preflight({'target': [self.target], 'ple': [self.ple], 'draft': [draft]},
                                 ple_mode='external', draft_mode='external')
        self.assertFalse(report['offline_checks_passed'])
        self.assertIn('below floor', ' '.join(report['artifact_manifest']['errors']))

    def test_embedded_table_must_exist_and_external_must_be_declared(self):
        report = build_preflight({'target': [self.target]}, ple_mode='embedded', draft_mode='disabled')
        self.assertFalse(report['offline_checks_passed'])
        self.assertIn('embedded PLE', ' '.join(report['configuration_errors']))
        fixture(self.target, [('per_layer_token_embd.weight', [32], 8, 0, 34)],
                [('general.architecture', 8, 'qwen4exp')])
        report = build_preflight({'target': [self.target]}, ple_mode='embedded', draft_mode='disabled')
        self.assertTrue(report['offline_checks_passed'])

    def test_model_name_cannot_enable_another_architecture(self):
        fixture(self.target, metadata=[('general.architecture', 8, 'other_model')])
        self.assertFalse(self.report()['offline_checks_passed'])

    def test_requested_capability_fails_closed(self):
        for capability in ('durable_checkpoint', 'warm_migration', 'invented', 'max_concurrent_sessions'):
            with self.subTest(capability=capability):
                report = self.report(required_capabilities=[capability])
                self.assertFalse(report['offline_checks_passed'])

    def test_cli_exit_and_input_overwrite_protection(self):
        cli = Path(__file__).with_name('backend_preflight.py')
        command = [sys.executable, str(cli), '--target', str(self.target), '--ple', str(self.ple),
                   '--ple-mode', 'external', '--draft-mode', 'disabled']
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        original = self.target.read_bytes()
        completed = subprocess.run(command + ['--output', str(self.target)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(self.target.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
