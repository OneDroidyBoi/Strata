"""Small serving contract; metadata never represents exported model state."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class BackendCapabilities:
    token_stream: bool = False
    cooperative_cancellation: bool = False
    durable_checkpoint: bool = False
    warm_migration: bool = False
    prefill_step: bool = False
    decode_step: bool = False
    # None means unverified, not unlimited. A slot count is not a capacity proof.
    max_concurrent_sessions: int | None = None


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class ModelIdentity:
    """Caller-supplied identity, not a loader attestation or precision check.

    weights_fingerprint must identify the complete weight/configuration manifest,
    including auxiliary/draft weights and PLE when present. Its format is owned by
    the caller; this generic contract does not assume a particular architecture.
    """

    architecture: str
    weights_fingerprint: str
    tokenizer_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("architecture", "weights_fingerprint", "tokenizer_fingerprint"):
            _nonempty(name, getattr(self, name))


@dataclass(frozen=True)
class SessionMetadata:
    """Request association only: no KV, recurrence, RNG, or restoration promise."""

    session_id: str
    model: ModelIdentity

    def __post_init__(self) -> None:
        _nonempty("session_id", self.session_id)
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("session model must be a ModelIdentity")


class UnsupportedBackendOperation(NotImplementedError):
    """The backend cannot safely provide the requested state operation."""


class Backend(ABC):
    capabilities = BackendCapabilities()

    def __init__(self, model: ModelIdentity):
        if not isinstance(model, ModelIdentity):
            raise TypeError("model must be a ModelIdentity")
        self.model = model

    def validate_session(self, session: SessionMetadata | None) -> None:
        if session is None:
            return
        if not isinstance(session, SessionMetadata):
            raise TypeError("session must be SessionMetadata")
        if session.model != self.model:
            raise ValueError("session model identity does not match this backend")

    @abstractmethod
    def generate(self, ids, max_new, sampling, cancel, embeddings=None,
                 *, session: SessionMetadata | None = None) -> Iterator[int | None]:
        """Return tokens/heartbeat None; lifetime and cancellation belong to executor."""
        raise NotImplementedError

    def _unsupported(self, operation: str):
        raise UnsupportedBackendOperation(
            f"{type(self).__name__} does not support {operation}; "
            "session metadata is not a portable checkpoint")

    def checkpoint(self, session: SessionMetadata):
        return self._unsupported("durable checkpoint export")

    def restore(self, checkpoint):
        return self._unsupported("durable checkpoint restore")

    def migrate(self, session: SessionMetadata, destination: Backend):
        return self._unsupported("warm session migration")

    def prefill(self, session: SessionMetadata, ids):
        return self._unsupported("independent prefill steps")

    def decode_step(self, session: SessionMetadata):
        return self._unsupported("independent decode steps")
