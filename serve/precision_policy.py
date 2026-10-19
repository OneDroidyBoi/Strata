"""Opt-in, pre-spawn precision guard for a deliberately narrow native-pack subset.

STRATA_PRECISION_POLICY names a JSON policy file. Paths inside it are relative to
that file; engine argument paths are relative to cwd. Call on EVERY spawn, with
the actual merged child environment. No cached manifest is trusted.

Schema: {"schema_version":1,"target":["model.gguf"],"pack":"pack",
         "ple":"embedded", "draft":"disabled",
         "minimum_bits":4,"preferred_bits":8}
ple can instead name a standalone strata-ple GGUF. Optional sha256 maps artifact
paths to expected full-file digests. This is not a kernel-compatibility test or a
sandbox against files modified concurrently after validation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct

from tools import model_manifest as M


class PrecisionPolicyError(ValueError):
    pass


# Fail closed on future/unknown options, especially alternate weight sources.
VALUE_FLAGS = set('''--native --pack --ple-gguf --resident-budget-gib --expert-profile
--expert-cache --prefill --spec --spec-min-p --max-context --kv --kv-resident
--pool-workers --pool-affinity --batch --batch-groups --prompt-cache
--prompt-cache-every --ple-io --ple-row-cache --ple-inflight --vram-reserve-mib
--pcie-frac --adapt-swaps --suffix-draft --seed --temperature --top-p --top-k'''.split())
VALUE_FLAGS.update('''--conversation-cache-mib --conversation-cache-slots
--conversation-cache-min-tokens --conversation-cache-min-free-mib
--session-min-free-mib'''.split())
BOOL_FLAGS = {'--serve', '--mmap-experts', '--resident-cpu-experts', '--resident-pin', '--resident-experts'}
SAFE_ENV = {'STRATA_PRECISION_POLICY', 'STRATA_EXPERIMENTAL_SM60', 'STRATA_SSD_KEEPALIVE',
            'STRATA_SSD_KEEPALIVE_WINDOW', 'STRATA_POOL_TRACE', 'STRATA_SESSION_EXPORT'}
EXPERT = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
TABLE = 'per_layer_token_embd.weight'
LIMIT = 16 << 20


def _fail(message):
    raise PrecisionPolicyError(message)


def _path(value, root):
    if not isinstance(value, str) or not value:
        _fail('policy and launch paths must be nonempty strings')
    return (root / value).resolve()


def _text(path):
    if path.stat().st_size > LIMIT:
        _fail(f'{path.name}: descriptor exceeds guard limit')
    return path.read_text(encoding='utf-8')


def _json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                _fail(f'duplicate JSON key {k}')
            result[k] = v
        return result
    return json.loads(_text(path), object_pairs_hook=pairs)


def _args(args):
    result = {}
    i = 0
    while i < len(args):
        flag = args[i]
        if flag in result or flag not in VALUE_FLAGS | BOOL_FLAGS:
            _fail(f'unreviewed, duplicate, or unsupported launch option: {flag}')
        if flag in VALUE_FLAGS:
            if i + 1 >= len(args) or not isinstance(args[i + 1], str) or args[i + 1].startswith('--'):
                _fail(f'missing launch value: {flag}')
            result[flag] = args[i + 1]
            i += 2
        else:
            result[flag] = True
            i += 1
    return result


def _shards(first):
    match = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', first.name)
    if not match:
        return [first]
    count = int(match[3])
    if not 1 <= count <= 1000 or not 1 <= int(match[2]) <= count:
        _fail('invalid native shard filename')
    return [first.with_name(f'{match[1]}-{i:05d}-of-{count:05d}.gguf') for i in range(1, count + 1)]


def _slice_digest(path, offset, count):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        while count:
            part = f.read(min(count, 1 << 20))
            if not part:
                _fail(f'{path}: truncated source span')
            count -= len(part)
            digest.update(part)
    return digest.digest()


def _converted_digest(path, offset, count, src, dst):
    """Only reviewed float transforms; never dequantize/relabel low-bit tensors."""
    if src == dst:
        return _slice_digest(path, offset, count)
    if (src, dst) not in {('F32', 'BF16'), ('F16', 'F32'), ('BF16', 'F32')}:
        _fail(f'unreviewed pack conversion {src}->{dst}')
    size = 4 if src == 'F32' else 2
    digest = hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        while count:
            data = f.read(min(count, 65536))
            if not data or len(data) % size:
                _fail('truncated float conversion source')
            count -= len(data)
            out = bytearray()
            for (bits,) in struct.iter_unpack('<I' if size == 4 else '<H', data):
                if src == 'F32':
                    if bits & 0x7f800000 == 0x7f800000:
                        _fail('nonfinite F32->BF16 conversion is not admitted')
                    out.extend(struct.pack('<H', (bits + 0x7fff + ((bits >> 16) & 1)) >> 16))
                elif src == 'BF16':
                    out.extend(struct.pack('<I', bits << 16))
                else:
                    out.extend(struct.pack('<f', struct.unpack('<e', struct.pack('<H', bits))[0]))
            digest.update(out)
    return digest.digest()


def _pack(pack, native_first, tensors, floor):
    # These override the validated GGUF source or introduce unreviewed raw payloads.
    for name in ('experts.bin', 'extra.bin', 'embd.bin'):
        if (pack / name).exists():
            _fail(f'{name} is not admitted: opaque runtime payload overrides audited source')
    desc = _text(pack / 'native_experts.txt')
    if not re.match(r'^# strata native experts v[34]:', desc):
        _fail('only native_experts.txt v3/v4 with explicit source offsets is admitted')
    by_layer = {}
    for name, (h, t) in tensors.items():
        match = EXPERT.fullmatch(name)
        if match:
            by_layer.setdefault(int(match[1]), {})[match[2]] = (h, t)
    if not by_layer or sorted(by_layer) != list(range(len(by_layer))):
        _fail('missing/noncontiguous routed expert layers')
    expert_counts = {t.shape[2] for roles in by_layer.values() for _, t in roles.values() if len(t.shape) == 3}
    header_count = re.search(r'\(n_expert (\d+), total (\d+);', desc.splitlines()[0])
    if len(expert_counts) != 1 or not header_count or int(header_count[1]) not in expert_counts:
        _fail('native expert header count is missing/inconsistent')
    seen, running = set(), 0
    for line in desc.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.split()
        if len(cols) not in (8, 9):
            _fail('native descriptor requires eight fields and optional shard column')
        layer, gu, down, offset, blob, *positions = map(int, cols[:8])
        if layer in seen or layer not in by_layer:
            _fail('duplicate/unknown native expert layer')
        seen.add(layer)
        roles = by_layer[layer]
        if set(roles) != {'gate', 'up', 'down'}:
            _fail('missing native expert role')
        ts = [roles[r] for r in ('gate', 'up', 'down')]
        shapes = [t.shape for _, t in ts]
        if any(len(s) != 3 or s[2] <= 0 for s in shapes) or len({s[2] for s in shapes}) != 1:
            _fail('invalid expert geometry')
        if shapes[0] != shapes[1] or shapes[2][:2] != shapes[0][:2][::-1]:
            _fail('gate/up/down expert dimensions disagree')
        if [t.type_id for _, t in ts] != [gu, gu, down] or offset != running:
            _fail('native expert format/layout mismatch')
        if gu in (39, 40) or down in (39, 40):
            _fail('scale-tail expert formats are not admitted by this guard')
        expected_blob = sum(t.expected_bytes() // t.shape[2] for _, t in ts)
        if blob != expected_blob:
            _fail('native expert blob size mismatch')
        running += blob * shapes[0][2]
        names = cols[8].split(',') if len(cols) == 9 else ['']
        if len(names) == 1:
            names *= 3
        if len(names) != 3:
            _fail('invalid per-role shard descriptor')
        for (h, t), pos, shard in zip(ts, positions, names):
            if shard and (Path(shard).name != shard or shard in ('.', '..')):
                _fail('native shard descriptor must be a basename')
            actual = (native_first.parent / shard).resolve() if shard else native_first
            if actual != h.path or pos != h.data_start + t.offset:
                _fail('native expert descriptor does not address its audited GGUF tensor')
    if int(header_count[2]) != running:
        _fail('native expert header total disagrees with source payload')
    if seen != set(by_layer):
        _fail('native descriptor omits expert layers')
    index = _text(pack / 'index.txt')
    if not index.startswith('# strata pack index v3'):
        _fail('only standalone native pack index v3 is admitted')
    expected = set(tensors) - {n for n in tensors if EXPERT.fullmatch(n)} - {TABLE}
    seen, spans = set(), []
    dense = pack / 'dense.bin'
    dense_size = dense.stat().st_size
    dst_end = 0
    header = re.search(r'^# align (\d+) pool (\d+) tensors (\d+)$', index, re.MULTILINE)
    if not header or int(header[1]) not in (64, 256):
        _fail('index requires an explicit reviewed 64- or 256-byte aligned pool header')
    alignment = int(header[1])
    for line in index.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        c = line.split()
        if len(c) != 19:
            _fail('index requires exactly nineteen fields')
        name = c[0]
        if name in seen or name not in expected:
            _fail(f'unknown/duplicate pack tensor {name}')
        seen.add(name)
        v = list(map(int, c[1:]))
        if any(x < 0 for x in v):
            _fail('negative pack index field')
        fileid, kind, off, size, dstoff, dstsize, ne0, ne1, codebits, bias, group, *tail = v
        h, t = tensors[name]
        if len(t.shape) > 2 or [ne0, ne1] != [t.shape[0], t.shape[1] if len(t.shape) == 2 else 0]:
            _fail(f'pack tensor shape mismatch: {name}')
        if fileid != 0 or dstoff != dst_end:
            _fail('index file id/pool offsets are outside standalone native subset')
        dst_end += (dstsize + alignment - 1) // alignment * alignment
        if kind == 0:
            # A native placeholder must not smuggle actual verbatim quantized bytes.
            if (off, size, dstsize, codebits, bias, group, *tail) != (0, 0, 0, 8, 0, 32, *([0] * 7)):
                _fail(f'unreviewed raw/verbatim index row: {name}')
            if t.type_name in ('F32', 'F16', 'BF16', 'F64'):
                _fail('float tensor cannot masquerade as a native quantized placeholder')
            continue
        if kind not in (1, 2, 3, 4, 5) or (codebits, bias, group, *tail) != (0, 0, 1, *([0] * 7)):
            _fail(f'unreviewed pack kind/quantized planes: {name}')
        storage, runtime = {1: ('F32', 'BF16'), 2: ('F32', 'F32'), 3: ('F32', 'F16'),
                            4: ('BF16', 'BF16'), 5: ('F16', 'F16')}[kind]
        if M.BITS[runtime] < floor:
            _fail(f'pack runtime dtype violates floor: {name}')
        elements = math.prod(t.shape)
        if size != elements * M.GEOMETRY[storage][1] or dstsize != elements * M.GEOMETRY[runtime][1]:
            _fail('packed float payload byte count mismatch')
        if off % alignment or off + size > dense_size:
            _fail('packed float span is unaligned/truncated')
        if _converted_digest(h.path, h.data_start + t.offset, t.expected_bytes(), t.type_name, storage) != _slice_digest(dense, off, size):
            _fail(f'packed payload differs from audited source/conversion: {name}')
        spans.append((off, off + size))
    if seen != expected or int(header[3]) != len(seen) or int(header[2]) != dst_end:
        _fail('pack index is incomplete or pool header disagrees')
    spans.sort()
    if any(b[0] < a[1] for a, b in zip(spans, spans[1:])):
        _fail('packed float spans overlap')
    return {'expert_layers': len(by_layer), 'expert_bytes': running, 'pack_tensors': len(seen),
            'packed_payload_verified_against_source': True}


def enforce_launch_policy(args, cwd, env):
    """Return a fresh report, None when opt-out, or raise PrecisionPolicyError.

    Engine arguments only (no executable); accepts optional --serve. No subprocess,
    GPU or network is used. Model files must be immutable during/after this check.
    """
    policy_name = env.get('STRATA_PRECISION_POLICY')
    if policy_name is None:
        return None
    try:
        root = Path(cwd).resolve()
        policy_path = _path(policy_name, root)
        policy = _json(policy_path)
        if not isinstance(policy, dict) or set(policy) - {'schema_version', 'target', 'pack', 'ple', 'draft', 'minimum_bits', 'preferred_bits', 'sha256'}:
            _fail('unknown precision policy fields/schema')
        if type(policy.get('schema_version')) is not int or policy['schema_version'] != 1 or policy.get('draft') != 'disabled':
            _fail('schema 1 and explicitly disabled learned draft model are required')
        for key in env:
            if key.startswith('STRATA_') and key not in SAFE_ENV:
                _fail(f'unreviewed precision-affecting environment override: {key}')
        opts = _args(args)
        target = policy.get('target')
        if not isinstance(target, list) or not target or len(target) > 1000:
            _fail('target must explicitly enumerate all GGUF shards')
        paths = [_path(p, policy_path.parent) for p in target]
        native = _path(opts.get('--native'), root)
        if set(_shards(native)) != set(paths) or len(paths) != len(set(paths)):
            _fail('--native does not select the policy target shard set')
        pack = _path(policy.get('pack'), policy_path.parent)
        if _path(opts.get('--pack'), root) != pack:
            _fail('--pack is not the policy pack')
        floor, preferred = policy.get('minimum_bits', 4), policy.get('preferred_bits', 8)
        if type(floor) is not int or floor not in (4, 8):
            _fail('initial runtime guard admits only Q4 or Q8 minimum policies; higher floors need transform review')
        inputs = {'target': paths}
        ple = policy.get('ple')
        if ple == 'embedded':
            explicit = opts.get('--ple-gguf')
            if explicit and _path(explicit, root) not in paths:
                _fail('--ple-gguf is not an embedded policy shard')
        else:
            table = _path(ple, policy_path.parent)
            if _path(opts.get('--ple-gguf'), root) != table:
                _fail('external PLE must be explicitly bound by --ple-gguf')
            inputs['ple'] = [table]
        audit = M.build_manifest(inputs, floor, preferred)
        if not audit['policy_passed']:
            _fail('artifact audit failed: ' + '; '.join(audit['errors'][:8]))
        tensors = {}
        for path in paths:
            h = M.Header(path)
            for t in h.tensors:
                tensors[t.name] = (h, t)
        if ple == 'embedded':
            if TABLE not in tensors:
                _fail('embedded PLE table missing')
            if opts.get('--ple-gguf') and _path(opts['--ple-gguf'], root) != tensors[TABLE][0].path:
                _fail('explicit embedded PLE shard does not contain the table')
        else:
            table_header = M.Header(inputs['ple'][0])
            if (table_header.metadata.get('general.architecture') != 'strata-ple'
                    or [t.name for t in table_header.tensors] != [TABLE]):
                _fail('external PLE must be a table-only strata-ple file; alternate dense sources rejected')
        pack_report = _pack(pack, native, tensors, floor)
        hashes = policy.get('sha256', {})
        if not isinstance(hashes, dict):
            _fail('sha256 must map paths to expected full-file digests')
        admitted = set(paths) | {pack / n for n in ('index.txt', 'dense.bin', 'native_experts.txt')}
        admitted.update(inputs.get('ple', []))
        for raw, digest in hashes.items():
            path = _path(raw, policy_path.parent)
            if path not in admitted or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                _fail('invalid or unbound sha256 declaration')
            if M._sha256(path) != digest:
                _fail(f'content identity mismatch: {path}')
        return {'schema_version': 1, 'enforced': True, 'policy_path': str(policy_path),
                'runtime_support': 'not_evaluated', 'draft': 'disabled', 'artifact_audit': audit,
                'pack_audit': pack_report, 'identity_hashes_checked': len(hashes),
                'limitation': 'reviewed native-pack subset only; caller must keep files immutable through spawn'}
    except PrecisionPolicyError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError, struct.error) as e:
        raise PrecisionPolicyError(f'precision launch validation failed: {e}') from e
