"""CPU-only synthetic GGUF fixtures; no model files or GPU dependencies."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    from . import model_manifest as M
except ImportError:
    import model_manifest as M


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def fixture(path, tensors=None, metadata=(), truncate=0):
    # tensors: name, dimensions, ggml type, offset, payload byte count
    if tensors is None:
        tensors = [('unrecognized_role.weight', [32], 8, 0, 34)]
    b = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for key, typ, val in metadata:
        b += string(key) + struct.pack('<I', typ)
        b += string(val) if typ == 8 else struct.pack('<' + M.SCALARS[typ], val)
    for name, shape, typ, offset, _ in tensors:
        b += string(name) + struct.pack('<I', len(shape)) + struct.pack('<' + 'Q' * len(shape), *shape)
        b += struct.pack('<IQ', typ, offset)
    b += b'\0' * (-len(b) % 32)
    b += b'\0' * max((off + size for _, _, _, off, size in tensors), default=0)
    path.write_bytes(b[:-truncate] if truncate else b)
    return path


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def audit(self, tensors=None, metadata=(), **kw):
        path = fixture(self.root / 'model.gguf', tensors, metadata)
        return M.build_manifest({'target': [path]}, **kw)

    def test_all_roles_and_exact_counts(self):
        result = self.audit([('unknown_role', [32], 2, 0, 18), ('norm', [1], 0, 32, 4)])
        self.assertTrue(result['policy_passed'])
        self.assertEqual(result['totals']['payload_bytes'], 22)
        self.assertEqual(result['totals']['tensor_count'], 2)
        self.assertEqual(result['components']['target']['below_preferred_count'], 1)
        self.assertEqual(result['scope']['draft'], 'not_provided')
        self.assertEqual(result['runtime_support'], 'not_evaluated')

    def test_nominal_precision_not_storage_average(self):
        r = self.audit([('q3', [256], 11, 0, 110)])
        self.assertFalse(r['policy_passed'])
        self.assertEqual(r['components']['target']['tensors'][0]['nominal_bits'], 3)
        self.assertEqual(r['totals']['payload_bytes'], 110)

    def test_unknown_type_fails(self):
        r = self.audit([('unknown', [32], 999, 0, 32)])
        self.assertFalse(r['policy_passed'])
        self.assertFalse(r['totals']['payload_bytes_complete'])

    def test_row_geometry_not_only_total(self):
        r = self.audit([('badrow', [16, 2], 8, 0, 34)])
        self.assertFalse(r['policy_passed'])
        self.assertIn('first dimension', ' '.join(r['errors']))

    def test_truncated_payload(self):
        path = fixture(self.root / 'm.gguf', truncate=1)
        r = M.build_manifest({'target': [path]})
        self.assertFalse(r['policy_passed'])
        self.assertIn('truncated', ' '.join(r['errors']))

    def test_overlap_and_alignment(self):
        r = self.audit([('a', [64], 8, 0, 68), ('b', [32], 8, 32, 34)])
        self.assertIn('overlapping', ' '.join(r['errors']))
        r = self.audit([('a', [32], 8, 1, 34)])
        self.assertIn('not aligned', ' '.join(r['errors']))

    def test_duplicate_tensors_in_and_across_files(self):
        r = self.audit([('a', [32], 8, 0, 34), ('a', [32], 8, 64, 34)])
        self.assertFalse(r['policy_passed'])
        a = fixture(self.root / 'a.gguf')
        b = fixture(self.root / 'b.gguf')
        self.assertFalse(M.build_manifest({'target': [a, b]})['policy_passed'])

    def test_complete_shards_and_missing_shards(self):
        paths = []
        for i in range(2):
            paths.append(fixture(self.root / f'm-{i+1:05}-of-00002.gguf',
                [(f't{i}', [32], 8, 0, 34)],
                [('split.count', 2, 2), ('split.no', 2, i), ('split.tensors.count', 10, 2)]))
        self.assertTrue(M.build_manifest({'target': paths})['policy_passed'])
        self.assertFalse(M.build_manifest({'target': paths[:1]})['policy_passed'])
        self.assertEqual(M.build_manifest({'target': paths}), M.build_manifest({'target': list(reversed(paths))}))

    def test_missing_shards_by_filename(self):
        path = fixture(self.root / 'm-00001-of-00002.gguf')
        self.assertFalse(M.build_manifest({'target': [path]})['policy_passed'])

    def test_draft_floor_and_required_scope(self):
        target = fixture(self.root / 't.gguf')
        draft = fixture(self.root / 'd.gguf', [('draft', [64], 42, 0, 18)])
        r = M.build_manifest({'target': [target], 'draft': [draft]})
        self.assertFalse(r['policy_passed'])
        self.assertEqual(r['scope']['draft'], 'invalid')
        r = M.build_manifest({'target': [target]}, required_components=('target', 'draft'))
        self.assertFalse(r['policy_passed'])
        self.assertFalse(M.build_manifest({'target': [target], 'draft': [self.root]})['policy_passed'])

    def test_fp8_ple_metadata_and_nan(self):
        for scale, success in [(1.0, True), (float('nan'), False), (-1.0, False)]:
            r = self.audit([('per_layer_token_embd.weight', [160], 24, 0, 160)],
                           [('strata.ple.format', 8, 'f8_e4m3'), ('strata.ple.scale', 6, scale)])
            self.assertEqual(r['policy_passed'], success)
            json.dumps(r, allow_nan=False)

    def test_i8_ple_missing_marker_is_not_generic_int8(self):
        r = self.audit([('per_layer_token_embd.weight', [160], 24, 0, 160)])
        self.assertFalse(r['policy_passed'])

    def test_architecture_and_integer_policy(self):
        r = self.audit(metadata=[('general.architecture', 8, 'qwen4exp')])
        self.assertEqual(r['components']['target']['architecture'], 'qwen4exp')
        for val in (True, 4.0, '4'):
            with self.assertRaises(ValueError):
                M.build_manifest({}, minimum_bits=val)

    def test_output_cannot_overwrite_artifact(self):
        path = fixture(self.root / 'target.gguf')
        original = path.read_bytes()
        alias = self.root / 'alias.gguf'
        alias.symlink_to(path)
        for dest in (path, alias):
            run = subprocess.run([sys.executable, str(Path(M.__file__)), '--target', str(path), '--output', str(dest)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(path.read_bytes(), original)
            self.assertNotIn('Traceback', run.stderr)

    def test_content_identity_is_optional_and_exact(self):
        path = fixture(self.root / 't.gguf')
        r = M.build_manifest({'target': [path]}, content_hash=True)
        self.assertEqual(r['components']['target']['files'][0]['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        r = M.build_manifest({'target': [path]})
        self.assertNotIn('sha256', r['components']['target']['files'][0])

    def test_hostile_headers_do_not_allocate_payload(self):
        path = self.root / 'bad.gguf'
        variants = [b'GGUF' + struct.pack('<IQQ', 3, 2**63, 0),
                    b'GGUF' + struct.pack('<IQQ', 3, 1, 0) + struct.pack('<Q', 2**63),
                    b'GGUF' + struct.pack('<IQQ', 3, 0, 1) + string('array') + struct.pack('<IIQ', 9, 10, 2**63),
                    b'GGUF']
        for payload in variants:
            path.write_bytes(payload)
            self.assertFalse(M.build_manifest({'target': [path]})['policy_passed'])

    def test_duplicate_metadata_alignment_and_zero_dims(self):
        for meta in [[('key', 4, 1), ('key', 4, 2)], [('general.alignment', 4, 3)]]:
            self.assertFalse(self.audit(metadata=meta)['policy_passed'])
        self.assertFalse(self.audit([('a', [0], 8, 0, 0)])['policy_passed'])

    def test_inconsistent_architectures(self):
        a = fixture(self.root / 'a.gguf', [('a', [32], 8, 0, 34)], [('general.architecture', 8, 'one')])
        b = fixture(self.root / 'b.gguf', [('b', [32], 8, 0, 34)], [('general.architecture', 8, 'two')])
        self.assertFalse(M.build_manifest({'target': [a, b]})['policy_passed'])

    def test_missing_file_and_empty_component_fail(self):
        self.assertFalse(M.build_manifest({'target': [self.root / 'absent.gguf']})['policy_passed'])
        path = fixture(self.root / 'empty.gguf', [])
        self.assertFalse(M.build_manifest({'target': [path]})['policy_passed'])
        self.assertFalse(M.build_manifest({})['policy_passed'])

    def test_floor_cannot_be_lowered(self):
        with self.assertRaises(ValueError):
            M.build_manifest({}, minimum_bits=2)

    def test_cli_json_output_and_exit(self):
        path = fixture(self.root / 't.gguf')
        script = Path(M.__file__)
        run = subprocess.run([sys.executable, str(script), '--target', str(path)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(json.loads(run.stdout)['policy_passed'])
        dest = self.root / 'audit.json'
        run = subprocess.run([sys.executable, str(script), '--target', str(path), '--require-component', 'draft', '--output', str(dest)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        self.assertFalse(json.loads(dest.read_text())['policy_passed'])
        self.assertEqual(run.stdout, '')


if __name__ == '__main__':
    unittest.main()
