"""Opt-in serving adapters. Importing this package never imports a GPU runtime."""
from .base import (
    Backend,
    BackendCapabilities,
    ModelIdentity,
    SessionMetadata,
    UnsupportedBackendOperation,
)
from .strata import StrataBackend

__all__ = [
    "Backend", "BackendCapabilities", "ModelIdentity", "SessionMetadata",
    "UnsupportedBackendOperation", "StrataBackend",
]
