#!/usr/bin/env python3
"""Audit every tensor of explicitly supplied local GGUF components, without loading weights.

Example: python tools/model_manifest.py --target model-00001-of-00002.gguf model-00002-of-00002.gguf --ple ple.gguf
Omitted components are NOT audited. This is a precision/storage audit, not runtime compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

try:
    from . import gguf_reader as G
except ImportError:
    import gguf_reader as G

COMPONENTS = ('target', 'ple', 'draft')
# Nominal value precision, deliberately NOT storage bytes / values (scales add overhead).
BITS = {'F32': 32, 'F16': 16, 'BF16': 16, 'F64': 64,
        'I8': 8, 'I16': 16, 'I32': 32, 'I64': 64,
        'MXFP4': 4, 'NVFP4': 4}
for _name in G.BLOCK_GEOMETRY:
    _m = re.match(r'I?Q([1-8])(?:_|$)', _name)
    if _m:
        BITS[_name] = int(_m.group(1))
GEOMETRY = dict(G.BLOCK_GEOMETRY, I8=(1, 1), I16=(1, 2), I32=(1, 4), I64=(1, 8))
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
MAX_HEADER = 256 << 20
MAX_STRING = 1 << 20
MAX_ENTRIES = 100_000


class AuditError(ValueError):
    pass


class Header:
    """Bounded GGUF v3 parser reusing gguf_reader's format inventory and TensorInfo.

    Metadata arrays are scanned/skipped, never materialized. Payload is not read.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = {}
        self.tensors = []
        with self.path.open('rb') as self.f:
            self.size = self.path.stat().st_size
            if self.read(4) != b'GGUF':
                raise AuditError('not a little-endian GGUF file')
            version, nt, nk = self.unpack('IQQ')
            if version != 3:
                raise AuditError(f'unsupported GGUF version {version}')
            if nt > MAX_ENTRIES or nk > MAX_ENTRIES:
                raise AuditError('header entry count exceeds audit limit')
            for _ in range(nk):
                key = self.string()
                if key in self.metadata:
                    raise AuditError(f'duplicate metadata key {key}')
                typ, = self.unpack('I')
                self.metadata[key] = self.value(typ)
            for _ in range(nt):
                name = self.string()
                nd, = self.unpack('I')
                if not name or not 1 <= nd <= 4:
                    raise AuditError('empty tensor name or dimension count outside 1..4')
                shape = list(self.unpack('Q' * nd))
                typ, offset = self.unpack('IQ')
                self.tensors.append(G.TensorInfo(name, shape, typ, G.GGML_TYPES.get(typ, f'type{typ}'), offset))
            alignment = self.metadata.get('general.alignment', 32)
            if type(alignment) is not int or alignment <= 0 or alignment > MAX_STRING or alignment & (alignment - 1):
                raise AuditError('alignment must be a bounded positive power of two')
            self.alignment = alignment
            self.data_start = (self.f.tell() + alignment - 1) // alignment * alignment
            if self.data_start > self.size:
                raise AuditError('truncated header alignment/payload start')

    def read(self, n):
        if n < 0 or self.f.tell() + n > min(self.size, MAX_HEADER):
            raise AuditError('truncated header or header exceeds audit limit')
        data = self.f.read(n)
        if len(data) != n:
            raise AuditError('truncated header')
        return data

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))

    def string(self):
        n, = self.unpack('Q')
        if n > MAX_STRING:
            raise AuditError('string exceeds audit limit')
        try:
            return self.read(n).decode('utf-8', errors='strict')
        except UnicodeDecodeError as e:
            raise AuditError('invalid UTF-8 header string') from e

    def value(self, typ):
        if typ in SCALARS:
            return self.unpack(SCALARS[typ])[0]
        if typ == 8:
            return self.string()
        if typ == 9:
            et, count = self.unpack('IQ')
            if et not in SCALARS and et != 8:
                raise AuditError('unsupported metadata array element type')
            # Enforce byte bounds before iterating hostile string counts.
            minimum = 8 if et == 8 else struct.calcsize('<' + SCALARS[et])
            end = self.f.tell() + count * minimum
            if end > min(self.size, MAX_HEADER):
                raise AuditError('truncated/oversized metadata array')
            if et == 8:
                for _ in range(count):
                    self.string()
            else:
                self.f.seek(count * minimum, 1)
            return {'array_type': et, 'count': count}
        raise AuditError(f'unsupported metadata type {typ}')


def _sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _component(name, paths, minimum, preferred, content_hash):
    out = {'files': [], 'tensors': [], 'errors': [], 'warnings': [], 'payload_bytes': 0,
           'tensor_count': 0, 'below_preferred_count': 0}
    headers, seen, physical = [], set(), set()
    for raw in sorted(map(str, paths)):
        path = Path(raw)
        try:
            before = path.stat()
            if not path.is_file():
                raise AuditError('expected GGUF file; legacy runtime packs/directories are not audited')
            key = (before.st_dev, before.st_ino)
            if key in physical:
                raise AuditError('same file supplied more than once')
            physical.add(key)
            h = Header(path)
            headers.append(h)
            record = {'path': str(path), 'file_bytes': h.size, 'data_offset': h.data_start,
                      'architecture': h.metadata.get('general.architecture') if isinstance(h.metadata.get('general.architecture'), str) else None,
                      'identity_strength': 'full_file_sha256' if content_hash else 'metadata_only_not_content_identity'}
            if content_hash:
                record['sha256'] = _sha256(path)
            out['files'].append(record)
            spans = []
            for t in h.tensors:
                context = f'{path}: {t.name}'
                item = {'name': t.name, 'file': str(path), 'shape': t.shape, 'type_id': t.type_id,
                        'type': t.type_name, 'offset': t.offset, 'payload_bytes': None,
                        'nominal_bits': BITS.get(t.type_name), 'errors': []}
                errors = item['errors']
                if t.name in seen:
                    errors.append('duplicate tensor name within component')
                seen.add(t.name)
                geom = GEOMETRY.get(t.type_name)
                if not geom or item['nominal_bits'] is None:
                    errors.append('unknown/unaccounted tensor format or block geometry')
                if any(d <= 0 for d in t.shape):
                    errors.append('tensor dimensions must be positive')
                elif geom:
                    block, width = geom
                    if t.shape[0] % block:
                        errors.append('first dimension is not whole quantization blocks')
                    else:
                        size = math.prod(t.shape) // block * width
                        item['payload_bytes'] = size
                        if t.offset % h.alignment:
                            errors.append('tensor offset is not aligned')
                        if t.offset + size > h.size - h.data_start:
                            errors.append('tensor payload is truncated/out of bounds')
                        else:
                            spans.append((t.offset, t.offset + size, t.name))
                # Strata FP8 PLE uses I8 storage; do not conflate this with integer Q8.
                if t.name == 'per_layer_token_embd.weight' and (t.type_name == 'I8' or h.metadata.get('strata.ple.format') is not None):
                    scale = h.metadata.get('strata.ple.scale')
                    if (t.type_name != 'I8' or h.metadata.get('strata.ple.format') != 'f8_e4m3'
                            or type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0):
                        errors.append('invalid FP8 PLE format/type/scale metadata')
                    else:
                        item['encoding'] = 'F8_E4M3'
                bits = item['nominal_bits']
                if bits is not None:
                    if bits < minimum:
                        errors.append(f'nominal precision {bits} is below floor {minimum}')
                    if bits < preferred:
                        out['below_preferred_count'] += 1
                for err in errors:
                    out['errors'].append(f'{context}: {err}')
                out['tensors'].append(item)
            spans.sort()
            for a, b in zip(spans, spans[1:]):
                if b[0] < a[1]:
                    out['errors'].append(f'{path}: overlapping tensor payloads {a[2]} and {b[2]}')
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev):
                raise AuditError('file changed during audit')
        except (OSError, ValueError, struct.error, OverflowError) as e:
            out['errors'].append(f'{path}: {e}')
    # A component is one model shard set, not a bag of independently named models.
    split = [h for h in headers if any(k in h.metadata for k in ('split.count', 'split.no', 'split.tensors.count'))]
    if split:
        counts = [h.metadata.get('split.count') for h in headers]
        nums = [h.metadata.get('split.no') for h in headers]
        totals = [h.metadata.get('split.tensors.count') for h in headers]
        if (any(type(x) is not int for x in counts + nums + totals)
                or len(set(counts)) != 1 or counts[0] != len(headers)
                or sorted(nums) != list(range(len(headers)))
                or len(set(totals)) != 1 or totals[0] != sum(len(h.tensors) for h in headers)):
            out['errors'].append(f'{name}: missing/inconsistent shard metadata or tensor count')
    # Also detect missing standard filename shards even when split metadata was omitted.
    filenames = [re.match(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', h.path.name) for h in headers]
    if any(filenames):
        if not all(filenames):
            out['errors'].append(f'{name}: mixed sharded and unsharded filenames')
        else:
            groups = {(m[1], int(m[3])) for m in filenames}
            if (len(groups) != 1 or int(filenames[0][3]) != len(headers)
                    or sorted(int(m[2]) for m in filenames) != list(range(1, len(headers) + 1))):
                out['errors'].append(f'{name}: missing/inconsistent filename shards')
    architectures = {str(h.metadata['general.architecture']) for h in headers if 'general.architecture' in h.metadata}
    out['architecture'] = next(iter(architectures)) if len(architectures) == 1 else None
    if len(architectures) > 1:
        out['errors'].append(f'{name}: inconsistent shard architecture')
    out['tensors'].sort(key=lambda x: (x['name'], x['file']))
    out['tensor_count'] = len(out['tensors'])
    out['payload_bytes'] = sum(t['payload_bytes'] or 0 for t in out['tensors'])
    out['payload_bytes_complete'] = all(t['payload_bytes'] is not None for t in out['tensors'])
    if not out['tensor_count']:
        out['errors'].append(f'{name}: no tensors audited')
    return out


def build_manifest(components, minimum_bits=4, preferred_bits=8, content_hash=False, required_components=('target',)):
    """Return deterministic JSON-compatible results; policy failures are data, not exceptions.

    Paths must explicitly enumerate all shards; no discovery, network, or model execution.
    Missing optional components never become a claim of whole-engine compliance.
    """
    if type(minimum_bits) is not int or type(preferred_bits) is not int or not (4 <= minimum_bits <= preferred_bits <= 64):
        raise ValueError('precision policy requires 4 <= minimum_bits <= preferred_bits <= 64')
    if set(components) - set(COMPONENTS) or set(required_components) - set(COMPONENTS):
        raise ValueError('components must be target, ple, or draft')
    result = {'schema_version': 1, 'audit_kind': 'artifact_precision_only', 'runtime_support': 'not_evaluated',
              'policy': {'minimum_bits': minimum_bits, 'preferred_bits': preferred_bits},
              'scope': {}, 'components': {}, 'errors': [], 'warnings': [], 'policy_passed': False}
    for name in COMPONENTS:
        paths = components.get(name, [])
        if not paths:
            result['scope'][name] = 'not_provided'
            result['warnings'].append(f'{name}: not audited; no precision/completeness claim')
            if name in required_components:
                result['errors'].append(f'{name}: required component not provided')
            continue
        item = _component(name, paths, minimum_bits, preferred_bits, content_hash)
        result['components'][name] = item
        result['scope'][name] = 'invalid' if item['errors'] else 'audited'
        result['errors'].extend(item['errors'])
        if item['below_preferred_count']:
            result['warnings'].append(f'{name}: {item["below_preferred_count"]} tensors below preferred {preferred_bits} bits')
    result['totals'] = {'tensor_count': sum(c['tensor_count'] for c in result['components'].values()),
                        'payload_bytes': sum(c['payload_bytes'] for c in result['components'].values()),
                        'payload_bytes_complete': all(c['payload_bytes_complete'] for c in result['components'].values())}
    result['policy_passed'] = not result['errors']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in COMPONENTS:
        parser.add_argument('--' + name, nargs='+', help='explicit local GGUF files, all shards')
    parser.add_argument('--minimum-bits', type=int, default=4)
    parser.add_argument('--preferred-bits', type=int, default=8)
    parser.add_argument('--require-component', action='append', choices=COMPONENTS, default=[])
    parser.add_argument('--content-hash', action='store_true', help='stream entire files for SHA256 (potentially slow)')
    parser.add_argument('--output', type=Path, help='write JSON here instead of stdout')
    args = parser.parse_args(argv)
    try:
        result = build_manifest({n: getattr(args, n) or [] for n in COMPONENTS}, args.minimum_bits,
                                args.preferred_bits, args.content_hash, tuple({'target', *args.require_component}))
    except ValueError as e:
        parser.error(str(e))
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        try:
            for name in COMPONENTS:
                for raw in getattr(args, name) or []:
                    source = Path(raw)
                    if args.output.resolve() == source.resolve() or (args.output.exists() and source.exists() and args.output.samefile(source)):
                        parser.error('--output must not overwrite an input artifact')
            args.output.write_text(text, encoding='utf-8')
        except OSError as e:
            parser.error(f'cannot write output: {e}')
    else:
        sys.stdout.write(text)
    return 0 if result['policy_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
