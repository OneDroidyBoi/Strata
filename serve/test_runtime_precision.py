"""Constructor integration tests; every spawn is mocked, no GPU/server is started."""
import os
import unittest
from unittest import mock

from serve.precision_policy import PrecisionPolicyError
from serve.server import StrataEngine
from serve import test_precision_policy as fixtures


class StopBeforeSpawn(Exception):
    pass


class RuntimePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PrecisionLaunchTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def engine(self, **kw):
        f = self.fixture
        options = {'cwd': str(f.root), 'env': dict(f.env), 'lazy': True}
        options.update(kw)
        return StrataEngine('/nonexistent/strata', list(f.args), **options)

    def test_bad_policy_fails_before_lazy_return_log_or_spawn(self):
        f = self.fixture
        (f.pack / 'experts.bin').write_bytes(b'opaque')
        logfile = f.root / 'must-not-create.log'
        for lazy in (True, False):
            with mock.patch('serve.server.popen') as spawn:
                with self.assertRaises(PrecisionPolicyError):
                    self.engine(lazy=lazy, log=str(logfile))
                spawn.assert_not_called()
                self.assertFalse(logfile.exists())

    def test_valid_lazy_instance_has_audit_without_spawn(self):
        with mock.patch('serve.server.popen') as spawn:
            engine = self.engine()
            self.assertTrue(engine.precision_policy_report['enforced'])
            self.assertIsNone(engine.proc)
            spawn.assert_not_called()

    def test_valid_nonlazy_reaches_mock_spawn_after_audit(self):
        with mock.patch('serve.server.popen', side_effect=StopBeforeSpawn) as spawn:
            with self.assertRaises(StopBeforeSpawn):
                self.engine(lazy=False)
            self.assertEqual(spawn.call_count, 1)
            self.assertEqual(spawn.call_args.kwargs['env'], self.fixture.env)

    def test_inherited_environment_is_checked(self):
        f = self.fixture
        with mock.patch.dict(os.environ, f.env, clear=True):
            engine = self.engine(env=None)
            self.assertTrue(engine.precision_policy_report['enforced'])
            self.assertIsNone(engine.spawn[4])
            (f.pack / 'experts.bin').write_bytes(b'opaque')
            with mock.patch('serve.server.popen') as spawn:
                with self.assertRaises(PrecisionPolicyError):
                    self.engine(env=None)
                spawn.assert_not_called()

    def test_optout_preserves_legacy_lazy_and_spawn_environment(self):
        with mock.patch('serve.server.popen') as spawn:
            engine = StrataEngine('/missing/strata', ['--unreviewed-option'], env={}, lazy=True)
            self.assertIsNone(engine.precision_policy_report)
            spawn.assert_not_called()
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch('serve.server.popen', side_effect=StopBeforeSpawn) as spawn:
                with self.assertRaises(StopBeforeSpawn):
                    StrataEngine('/missing/strata', [], env=None)
                self.assertIsNone(spawn.call_args.kwargs['env'])

    def test_restart_revalidates_changed_payload(self):
        engine = self.engine()
        (self.fixture.pack / 'dense.bin').write_bytes(b'x' + bytes(511))
        with mock.patch('serve.server.popen') as spawn:
            with self.assertRaises(PrecisionPolicyError):
                engine.restart(tries=1)
            spawn.assert_not_called()
            self.assertFalse(engine.starting)

    def test_restart_reads_current_inherited_environment(self):
        with mock.patch.dict(os.environ, self.fixture.env, clear=True):
            engine = self.engine(env=None)
            os.environ['STRATA_UNREVIEWED_WEIGHT_OVERRIDE'] = '1'
            with mock.patch('serve.server.popen') as spawn:
                with self.assertRaises(PrecisionPolicyError):
                    engine.restart(tries=1)
                spawn.assert_not_called()

    def test_enforced_spawn_uses_environment_snapshot(self):
        with mock.patch.dict(os.environ, self.fixture.env, clear=True):
            with mock.patch('serve.server.popen', side_effect=StopBeforeSpawn) as spawn:
                with self.assertRaises(StopBeforeSpawn):
                    self.engine(env=None, lazy=False)
                child_env = spawn.call_args.kwargs['env']
                self.assertEqual(child_env, self.fixture.env)
                self.assertIsNot(child_env, os.environ)


if __name__ == '__main__':
    unittest.main()
