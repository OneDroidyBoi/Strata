"""Real-disk lifecycle tests with injected workers; no native engine is loaded."""
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve.backends import ModelIdentity, SessionMetadata
from serve.checkpoint_store import CheckpointStore
from serve.lane_lifecycle import LaneLifecycle


class FakeWorker:
    def __init__(self, model, count=2):
        self.sessions = [SessionMetadata(f"s{i}", model) for i in range(count)]
        self.events = []
        self.gated = False
        self.alive = True
        self.drained = True
        self.fail_export = False
        self.fail_unload = False
        self.restored = []

    def gate_admission(self):
        self.gated = True
        self.events.append("gate")

    def wait_drained(self, timeout_s):
        self.events.append("drain")
        return self.drained

    def list_sessions(self):
        self.events.append("list")
        return tuple(self.sessions)

    def export_session(self, session, path):
        assert self.gated and self.alive
        self.events.append("export:" + session.session_id)
        with path.open("xb") as f:
            # Stream multiple chunks rather than passing state bytes to store.
            for _ in range(20):
                f.write((session.session_id * 1000).encode())
                if self.fail_export:
                    raise OSError("disk full")

    def unload(self):
        self.events.append("unload")
        if self.fail_unload:
            raise RuntimeError("unload unconfirmed")
        self.alive = False

    def restore_session(self, session, path):
        self.events.append("restore:" + session.session_id)
        with path.open("rb") as f:
            size = sum(len(chunk) for chunk in iter(lambda: f.read(1024), b""))
        self.restored.append((session, size))


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = ModelIdentity("architecture", "weights:v1", "tokenizer:v1")
        self.store = CheckpointStore(self.root)
        self.worker = FakeWorker(self.model)
        self.lane = LaneLifecycle(self.worker, self.store, self.model)

    def publish(self):
        result = self.lane.persist_and_stop()
        self.assertEqual(result.state, "stopped", result.error)
        return result.checkpoint_id

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_fifo_checkpoint_payload_is_rejected_without_waiting_for_writer(self):
        fifo = self.root / 'unexpected-pipe'
        os.mkfifo(fifo)
        # Bound the regression externally: the old open() hangs before fstat.
        probe = subprocess.run([sys.executable, '-c',
            "from pathlib import Path; from serve.checkpoint_store import _digest; "
            "import sys; _digest(Path(sys.argv[1]))", str(fifo)],
            capture_output=True, text=True, timeout=2,
            cwd=Path(__file__).resolve().parents[1])
        self.assertNotEqual(probe.returncode, 0)
        self.assertIn('regular non-symlink', probe.stderr)

    def test_durable_publish_then_stop_and_lazy_restore_after_restart(self):
        original_unload = self.worker.unload
        def unload():
            bundles = [p for p in self.root.iterdir() if not p.name.startswith(".")]
            self.assertEqual(len(bundles), 1)
            reopened = CheckpointStore(self.root)
            self.assertEqual(reopened.sessions(bundles[0].name, self.model), tuple(self.worker.sessions))
            original_unload()
        self.worker.unload = unload
        checkpoint = self.publish()
        self.assertEqual(self.worker.events[:2], ["gate", "drain"])
        self.assertEqual(self.worker.events[-1], "unload")
        self.assertFalse(self.worker.alive)
        self.assertEqual(self.lane.persist_and_stop().checkpoint_id, checkpoint)
        ready = FakeWorker(self.model, count=0)
        restarted = LaneLifecycle(ready, CheckpointStore(self.root), self.model)
        result = restarted.lazy_restore(checkpoint, "s1")
        self.assertEqual(result.state, "restored", result.error)
        self.assertEqual([s.session_id for s, _ in ready.restored], ["s1"])
        self.assertTrue(ready.gated)
        self.assertTrue(ready.alive)

    def test_drain_timeout_never_exports_or_stops(self):
        self.worker.drained = False
        result = self.lane.persist_and_stop(0)
        self.assertEqual(result.state, "blocked")
        self.assertEqual(self.worker.events, ["gate", "drain"])
        self.assertTrue(self.worker.gated and self.worker.alive)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_partial_write_failure_keeps_worker_and_only_pending_files(self):
        self.worker.fail_export = True
        result = self.lane.persist_and_stop()
        self.assertEqual(result.state, "blocked")
        self.assertIn("disk full", result.error)
        self.assertTrue(self.worker.gated and self.worker.alive)
        self.assertNotIn("unload", self.worker.events)
        self.assertTrue(all(p.name.startswith(".pending-") for p in self.root.iterdir()))

    def test_fsync_and_rename_failures_never_unload(self):
        for target in ("serve.checkpoint_store.os.fsync", "serve.checkpoint_store.os.rename"):
            with self.subTest(target=target):
                with patch(target, side_effect=OSError("storage failure")):
                    result = self.lane.persist_and_stop()
                self.assertEqual(result.state, "blocked")
                self.assertTrue(self.worker.alive)
                self.assertNotIn("unload", self.worker.events)

    def test_parent_directory_sync_failure_does_not_unload(self):
        from serve import checkpoint_store
        real_sync = checkpoint_store._sync_dir
        def sync(path):
            if path == self.root:
                raise OSError("root sync failed")
            real_sync(path)
        with patch("serve.checkpoint_store._sync_dir", side_effect=sync):
            result = self.lane.persist_and_stop()
        self.assertEqual(result.state, "blocked")
        self.assertTrue(self.worker.alive)
        self.assertNotIn("unload", self.worker.events)

    def test_duplicate_or_changed_session_inventory_blocks_stop(self):
        self.worker.sessions.append(self.worker.sessions[0])
        self.assertEqual(self.lane.persist_and_stop().state, "blocked")
        self.worker.sessions.pop()
        export = self.worker.export_session
        def changing(session, path):
            export(session, path)
            if session.session_id == "s1":
                self.worker.sessions.pop()
        self.worker.export_session = changing
        self.assertEqual(self.lane.persist_and_stop().state, "blocked")
        self.assertTrue(self.worker.alive)
        self.assertNotIn("unload", self.worker.events)

    def test_corruption_or_wrong_identity_never_calls_restorer(self):
        checkpoint = self.publish()
        ready = FakeWorker(self.model, 0)
        restored = LaneLifecycle(ready, self.store, self.model)
        payload = self.root / checkpoint / "session-0000.bin"
        with payload.open("ab") as f:
            f.write(b"corrupt")
        self.assertEqual(restored.lazy_restore(checkpoint, "s0").state, "blocked")
        self.assertEqual(ready.restored, [])
        wrong = ModelIdentity("architecture", "different-weights", "tokenizer:v1")
        result = LaneLifecycle(ready, self.store, wrong).lazy_restore(checkpoint, "s1")
        self.assertEqual(result.state, "blocked")
        self.assertEqual(ready.restored, [])

    def test_version_and_path_traversal_are_rejected(self):
        checkpoint = self.publish()
        manifest_path = self.root / checkpoint / "manifest.json"
        saved = json.loads(manifest_path.read_text())
        for change in ("version", "path"):
            manifest = json.loads(json.dumps(saved))
            if change == "version":
                manifest["version"] = 999
            else:
                manifest["sessions"][0]["file"] = "../../outside"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.store.restore(checkpoint, "s0", self.model, lambda *_: self.fail("restored unsafe file"))
        for bad in ("../outside", ".pending-" + checkpoint, "", "/tmp/anywhere"):
            with self.assertRaises(ValueError):
                self.store.sessions(bad, self.model)

    def test_ambiguous_manifest_is_rejected_before_reader(self):
        checkpoint = self.publish()
        path = self.root / checkpoint / 'manifest.json'
        original = path.read_text()
        manifest = json.loads(original)
        digest = manifest['sessions'][0]['sha256']
        cases = [
            original.replace('"version":1', '"version":true'),
            original.replace('"version":1', '"version":1.0'),
            '{"sessions":[],' + original[1:],
            original.replace('"sha256":"' + digest + '"',
                             '"sha256":"' + '0' * 64 + '","sha256":"' + digest + '"', 1),
            '{"unexpected":NaN,' + original[1:],
        ]
        for text in cases:
            with self.subTest(text=text[:90]):
                path.write_text(text)
                called = []
                with self.assertRaises(ValueError):
                    self.store.restore(checkpoint, 's0', self.model, lambda *_: called.append(True))
                self.assertEqual(called, [])
        path.write_text(original)
        called = []
        self.store.restore(checkpoint, 's0', self.model, lambda *_: called.append(True))
        self.assertEqual(called, [True])

    def test_payload_symlink_is_rejected_on_export_and_restore(self):
        outside = self.root / "outside"
        outside.write_bytes(b"data")
        def unsafe_export(session, path):
            path.symlink_to(outside)
        self.worker.export_session = unsafe_export
        self.assertEqual(self.lane.persist_and_stop().state, "blocked")
        self.assertTrue(self.worker.alive)
        self.worker.export_session = FakeWorker.export_session.__get__(self.worker)
        checkpoint = self.publish()
        payload = self.root / checkpoint / "session-0000.bin"
        payload.unlink()
        payload.symlink_to(outside)
        ready = FakeWorker(self.model, 0)
        self.assertEqual(LaneLifecycle(ready, self.store, self.model).lazy_restore(checkpoint, "s0").state, "blocked")
        self.assertEqual(ready.restored, [])

    def test_unload_failure_preserves_durable_bundle_and_reports_blocked(self):
        self.worker.fail_unload = True
        result = self.lane.persist_and_stop()
        self.assertEqual(result.state, "blocked")
        self.assertIsNotNone(result.checkpoint_id)
        self.assertEqual(self.store.sessions(result.checkpoint_id, self.model), tuple(self.worker.sessions))
        self.assertTrue(self.worker.alive)

    def test_concurrent_operation_is_blocked_without_worker_calls(self):
        self.lane._lock.acquire()
        try:
            self.assertEqual(self.lane.persist_and_stop().state, "blocked")
            self.assertEqual(self.worker.events, [])
        finally:
            self.lane._lock.release()


if __name__ == "__main__":
    unittest.main()
