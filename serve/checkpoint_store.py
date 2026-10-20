"""Durable, local checkpoint bundles. Payload writers own the state format.

This is storage infrastructure, not proof that an engine can export all its
sessions. The root and its writers are trusted local code, not HTTP input.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

from serve.backends import ModelIdentity, SessionMetadata

_VERSION = 1
_MAX_MANIFEST = 1 << 20
_MAX_SESSIONS = 1024
_ID = re.compile(r"[0-9a-f]{32}\Z")
_FILE = re.compile(r"session-[0-9]{4}\.bin\Z")


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_regular(path: Path):
    # Opening a FIFO read-only can block before fstat gets to reject it.
    # NONBLOCK has no effect on regular files; inspect the actual opened inode.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or path.is_symlink():
            raise ValueError("checkpoint payload must be a regular non-symlink file")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _digest(path: Path, *, sync: bool = False) -> tuple[str, int]:
    h, size = hashlib.sha256(), 0
    with _open_regular(path) as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            size += len(block)
        if sync:
            os.fsync(f.fileno())
    return h.hexdigest(), size


class CheckpointStore:
    def __init__(self, root):
        path = Path(root).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("checkpoint root must not traverse symlinks")
        if str(root).startswith(("//", "\\\\")):
            raise ValueError("checkpoint root must be local")
        # Require an existing trusted directory: its creation/durability belongs
        # to installation, not a checkpoint transaction.
        if not path.is_dir():
            raise ValueError("checkpoint root must be an existing directory")
        self.root = path

    def publish(self, model: ModelIdentity, sessions, writer) -> str:
        sessions = tuple(sessions)
        if not isinstance(model, ModelIdentity):
            raise TypeError("model must be ModelIdentity")
        if len(sessions) > _MAX_SESSIONS:
            raise ValueError("too many sessions for a checkpoint bundle")
        if any(not isinstance(s, SessionMetadata) or s.model != model for s in sessions):
            raise ValueError("checkpoint session identity mismatch")
        if len({s.session_id for s in sessions}) != len(sessions):
            raise ValueError("duplicate checkpoint session")
        bundle_id = uuid.uuid4().hex
        staging = self.root / (".pending-" + bundle_id)
        staging.mkdir(mode=0o700)
        entries = []
        # Failed staging directories are deliberately retained for diagnosis;
        # readers accept only published UUID names, never .pending directories.
        for i, session in enumerate(sessions):
            name = f"session-{i:04d}.bin"
            path = staging / name
            writer(session, path)
            digest, size = _digest(path, sync=True)
            entries.append({"session": asdict(session), "file": name,
                            "sha256": digest, "bytes": size})
        manifest = {"version": _VERSION, "checkpoint_id": bundle_id,
                    "model": asdict(model), "sessions": entries}
        data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(data) > _MAX_MANIFEST:
            raise ValueError("checkpoint manifest too large")
        temp = staging / "manifest.tmp"
        with temp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, staging / "manifest.json")
        _sync_dir(staging)
        os.rename(staging, self.root / bundle_id)
        _sync_dir(self.root)
        return bundle_id

    def _manifest(self, checkpoint_id: str, model: ModelIdentity):
        if not isinstance(checkpoint_id, str) or not _ID.fullmatch(checkpoint_id):
            raise ValueError("invalid checkpoint id")
        bundle = self.root / checkpoint_id
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError("checkpoint directory is missing or unsafe")
        with _open_regular(bundle / "manifest.json") as f:
            data = f.read(_MAX_MANIFEST + 1)
        if len(data) > _MAX_MANIFEST:
            raise ValueError("checkpoint manifest too large")
        def unique_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate checkpoint manifest field: " + key)
                result[key] = value
            return result

        def invalid_constant(value):
            raise ValueError("invalid checkpoint JSON constant: " + value)

        manifest = json.loads(data, object_pairs_hook=unique_fields, parse_constant=invalid_constant)
        if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
                or manifest["version"] != _VERSION):
            raise ValueError("unsupported checkpoint version")
        if manifest.get("checkpoint_id") != checkpoint_id or manifest.get("model") != asdict(model):
            raise ValueError("checkpoint model or bundle identity mismatch")
        entries = manifest.get("sessions")
        if not isinstance(entries, list) or len(entries) > _MAX_SESSIONS:
            raise ValueError("invalid checkpoint session list")
        seen, files, parsed = set(), set(), []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("session"), dict):
                raise ValueError("invalid checkpoint session")
            sm = entry["session"]
            if sm.get("model") != asdict(model):
                raise ValueError("checkpoint session model mismatch")
            session = SessionMetadata(sm.get("session_id"), model)
            name, digest, size = entry.get("file"), entry.get("sha256"), entry.get("bytes")
            if (not isinstance(name, str) or not _FILE.fullmatch(name) or name in files
                    or session.session_id in seen or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or type(size) is not int or size < 0):
                raise ValueError("invalid checkpoint file metadata")
            files.add(name)
            seen.add(session.session_id)
            parsed.append((session, bundle / name, digest, size))
        return parsed

    def sessions(self, checkpoint_id: str, model: ModelIdentity) -> tuple[SessionMetadata, ...]:
        """Read metadata only; payload integrity is checked when selected for restore."""
        return tuple(row[0] for row in self._manifest(checkpoint_id, model))

    def restore(self, checkpoint_id: str, session_id: str, model: ModelIdentity, reader):
        for session, path, expected, size in self._manifest(checkpoint_id, model):
            if session.session_id == session_id:
                if _digest(path) != (expected, size):
                    raise ValueError("checkpoint checksum or size mismatch")
                # Trusted local writers must not mutate published bundles. The
                # reader receives a path for bounded engine-side streaming.
                reader(session, path)
                return session
        raise ValueError("session is not in this checkpoint")
