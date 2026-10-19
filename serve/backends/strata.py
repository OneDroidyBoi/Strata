"""Opt-in adapter for the existing Strata generate API (Flash-Next first).

No engine is constructed here. The caller owns loading, precision validation,
locks, admission, draining, shutdown, and any model-specific configuration.
"""
from __future__ import annotations

from .base import Backend, BackendCapabilities, ModelIdentity, SessionMetadata


class StrataBackend(Backend):
    capabilities = BackendCapabilities(token_stream=True, cooperative_cancellation=True)
    supported_architectures = frozenset({"qwen4exp"})

    def __init__(self, engine, model: ModelIdentity):
        super().__init__(model)
        if model.architecture not in self.supported_architectures:
            raise ValueError(f"StrataBackend does not support architecture {model.architecture!r}; "
                             "this first adapter requires qwen4exp")
        if not callable(getattr(engine, "generate", None)):
            raise TypeError("engine must provide a callable generate method")
        self.engine = engine

    def generate(self, ids, max_new, sampling, cancel, embeddings=None,
                 *, session: SessionMetadata | None = None):
        self.validate_session(session)
        # Return the exact iterator: close/throw, heartbeats, exceptions and
        # deferred cleanup retain the executor's semantics. Do not take a new
        # lock or intercept cancellation; that could prevent its drain protocol.
        if embeddings is not None:
            return self.engine.generate(ids, max_new, sampling, cancel, embeddings=embeddings)
        return self.engine.generate(ids, max_new, sampling, cancel)
