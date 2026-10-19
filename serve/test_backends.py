"""CPU-only contract checks; no server import, engine startup or model files."""
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve.backends import (  # noqa: E402
    ModelIdentity, SessionMetadata, StrataBackend, UnsupportedBackendOperation,
)


class FakeEngine:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []
        self.closed = False
        self.iterator = None

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.calls.append((ids, max_new, sampling, cancel, kwargs))

        def stream():
            try:
                for token in (None, 17, 19):
                    if cancel.is_set():
                        return
                    yield token
                if self.failure:
                    raise self.failure
            finally:
                self.closed = True

        self.iterator = stream()
        return self.iterator


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.model = ModelIdentity("qwen4exp", "manifest:a", "tokenizer:b")
        self.session = SessionMetadata("agent-one", self.model)

    def test_stream_and_arguments_are_transparent(self):
        engine = FakeEngine()
        backend = StrataBackend(engine, self.model)
        ids, sampling, cancel, embeddings = [1, 2], {"temperature": 0.7}, threading.Event(), object()
        stream = backend.generate(ids, 3, sampling, cancel, embeddings, session=self.session)
        self.assertIs(stream, engine.iterator)
        self.assertEqual(list(stream), [None, 17, 19])
        args = engine.calls[0]
        self.assertIs(args[0], ids)
        self.assertEqual(args[1], 3)
        self.assertIs(args[2], sampling)
        self.assertIs(args[3], cancel)
        self.assertIs(args[4]["embeddings"], embeddings)
        self.assertTrue(engine.closed)

    def test_direct_and_adapter_parity(self):
        direct, wrapped = FakeEngine(), FakeEngine()
        event = threading.Event()
        self.assertEqual(list(direct.generate([1], 2, {}, event)),
                         list(StrataBackend(wrapped, self.model).generate([1], 2, {}, event)))
        self.assertEqual(wrapped.calls[0][4], {})

    def test_cancel_is_observed_by_engine_and_cleanup_runs(self):
        for cancel_before in (False, True):
            with self.subTest(cancel_before=cancel_before):
                engine, event = FakeEngine(), threading.Event()
                stream = StrataBackend(engine, self.model).generate([], 2, {}, event)
                if not cancel_before:
                    self.assertIsNone(next(stream))
                event.set()
                self.assertEqual(list(stream), [])
                self.assertTrue(engine.closed)

    def test_close_and_throw_reach_original_generator(self):
        for action in ("close", "throw"):
            with self.subTest(action=action):
                engine = FakeEngine()
                stream = StrataBackend(engine, self.model).generate([], 2, {}, threading.Event())
                next(stream)
                if action == "close":
                    stream.close()
                else:
                    failure = RuntimeError("consumer failed")
                    with self.assertRaises(RuntimeError) as caught:
                        stream.throw(failure)
                    self.assertIs(caught.exception, failure)
                self.assertTrue(engine.closed)

    def test_engine_exception_identity_is_preserved(self):
        failure = RuntimeError("worker failed")
        engine = FakeEngine(failure)
        with self.assertRaises(RuntimeError) as caught:
            list(StrataBackend(engine, self.model).generate([], 2, {}, threading.Event()))
        self.assertIs(caught.exception, failure)
        self.assertTrue(engine.closed)

    def test_immediate_engine_exception_is_preserved(self):
        failure = ValueError("invalid request")
        class Broken:
            def generate(self, *args):
                raise failure
        with self.assertRaises(ValueError) as caught:
            StrataBackend(Broken(), self.model).generate([], 2, {}, threading.Event())
        self.assertIs(caught.exception, failure)

    def test_model_validation_and_mismatch_before_execution(self):
        for field in ("architecture", "weights_fingerprint", "tokenizer_fingerprint"):
            args = dict(architecture="other-model", weights_fingerprint="w", tokenizer_fingerprint="t")
            args[field] = " "
            with self.assertRaises(ValueError):
                ModelIdentity(**args)
        other = ModelIdentity("dense-test-model", "different", "tokenizer:b")
        engine = FakeEngine()
        backend = StrataBackend(engine, self.model)
        with self.assertRaises(ValueError):
            backend.generate([], 2, {}, threading.Event(), session=SessionMetadata("s", other))
        self.assertEqual(engine.calls, [])
        # Metadata is model-neutral; this first adapter has a narrower scope.
        self.assertEqual(other.architecture, "dense-test-model")
        with self.assertRaisesRegex(ValueError, "does not support architecture"):
            StrataBackend(FakeEngine(), other)
        with self.assertRaises(TypeError):
            SessionMetadata("s", "not an identity")
        with self.assertRaises(ValueError):
            SessionMetadata("", self.model)

    def test_capabilities_do_not_claim_portable_state(self):
        backend = StrataBackend(FakeEngine(), self.model)
        caps = backend.capabilities
        self.assertTrue(caps.token_stream)
        self.assertTrue(caps.cooperative_cancellation)
        self.assertIsNone(caps.max_concurrent_sessions)
        self.assertFalse(any((caps.durable_checkpoint, caps.warm_migration, caps.prefill_step, caps.decode_step)))
        calls = [lambda: backend.checkpoint(self.session), lambda: backend.restore(b"state"),
                 lambda: backend.migrate(self.session, backend), lambda: backend.prefill(self.session, [1]),
                 lambda: backend.decode_step(self.session)]
        for call in calls:
            with self.assertRaisesRegex(UnsupportedBackendOperation, "does not support"):
                call()


if __name__ == "__main__":
    unittest.main()
