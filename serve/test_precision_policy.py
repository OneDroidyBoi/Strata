"""Precision guard tests: small synthetic files; never spawn an engine."""
import hashlib
import json
from pathlib import Path
import tempfile
import struct
import unittest

from serve.precision_policy import enforce_launch_policy, PrecisionPolicyError, _q8_0_bf16_digest
from tools import model_manifest as M
from tools.test_model_manifest import fixture


class PrecisionLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.target = self.root / 'model.gguf'
        tensors, offset = [], 0
        for role in ('gate', 'up', 'down'):
            tensors.append((f'blk.0.ffn_{role}_exps.weight', [32, 32, 2], 2, offset, 1152))
            offset = (offset + 1152 + 31) // 32 * 32
        tensors.extend([('blk.0.ffn_gate_inp.weight', [32, 2], 0, offset, 256),
                        ('norm.weight', [1], 0, offset + 256, 4),
                        ('output.weight', [32, 2], 8, offset + 288, 68),
                        ('per_layer_token_embd.weight', [160, 1], 20, offset + 384, 90)])
        fixture(self.target, tensors, [('general.architecture', 8, 'qwen4exp')])
        h = M.Header(self.target)
        self.pack = self.root / 'pack'
        self.pack.mkdir()
        positions = [h.data_start + t.offset for t in h.tensors[:3]]
        (self.pack / 'native_experts.txt').write_text('# strata native experts v3: test (n_expert 2, total 3456; source)\n0 2 2 0 1728 ' + ' '.join(map(str, positions)) + '\n')
        # Every float payload is zero in this synthetic source.
        (self.pack / 'dense.bin').write_bytes(bytes(512))
        rows = [self.row('blk.0.ffn_gate_inp.weight', 2, 0, 256, 0, 256, 32, 2),
                self.row('norm.weight', 2, 256, 4, 256, 4, 1, 0),
                'output.weight 0 0 0 0 512 0 32 2 8 0 32 ' + ' '.join(['0'] * 7)]
        (self.pack / 'index.txt').write_text('# strata pack index v3 -- standalone\n# align 256 pool 512 tensors 3\n' + '\n'.join(rows) + '\n')
        self.policy = self.root / 'policy.json'
        self.document = {'schema_version': 1, 'target': ['model.gguf'], 'pack': 'pack', 'ple': 'embedded', 'draft': 'disabled'}
        self.write_policy()
        self.env = {'STRATA_PRECISION_POLICY': str(self.policy)}
        self.args = ['--native', str(self.target), '--pack', str(self.pack), '--spec', '4']

    @staticmethod
    def row(name, kind, off, size, dstoff, dstsize, n0, n1):
        return f'{name} 0 {kind} {off} {size} {dstoff} {dstsize} {n0} {n1} 0 0 1 ' + ' '.join(['0'] * 7)

    def write_policy(self):
        self.policy.write_text(json.dumps(self.document))

    def tearDown(self):
        self.temp.cleanup()

    def run_guard(self, args=None, env=None):
        return enforce_launch_policy(self.args if args is None else args, self.root, self.env if env is None else env)

    def install_q8_compat_pack(self):
        """Replace the ordinary fixture with one real Q8_0 source span and audited manifests."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy required for Q8 compatibility fixtures')

        self.target = self.root / 'q8-model.gguf'
        rows, offset = [], 0
        for role in ('gate', 'up', 'down'):
            rows.append((f'blk.0.ffn_{role}_exps.weight', [32, 32, 2], 2, offset, 1152))
            offset += 1152
        rows.extend([
            ('blk.0.ffn_gate_inp.weight', [32, 2], 0, offset, 256),
            ('norm.weight', [1], 0, offset + 256, 4),
            ('output.weight', [32, 2], 8, offset + 288, 68),
            ('per_layer_token_embd.weight', [160, 1], 20, offset + 384, 90),
            ('blk.0.attn_q.weight', [32], 8, offset + 480, 34),
        ])
        fixture(self.target, rows, [('general.architecture', 8, 'qwen4exp')])
        header = M.Header(self.target)
        source = next(t for t in header.tensors if t.name == 'blk.0.attn_q.weight')
        codes = np.array([0, 1, -1, 127, -128] + [0] * 27, dtype=np.int8)
        q8 = struct.pack('<e', 0.3) + codes.tobytes()
        with self.target.open('r+b') as f:
            f.seek(header.data_start + source.offset)
            f.write(q8)
        values = np.float32(np.float16(0.3)) * codes.astype(np.float32)
        bits = values.view(np.uint32)
        converted = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2').tobytes()
        _, error = _q8_0_bf16_digest(self.target, header.data_start + source.offset, len(q8))

        (self.pack / 'native_experts.txt').write_text(
            '# strata native experts v3: test (n_expert 2, total 3456; source)\n0 2 2 0 1728 '
            + ' '.join(str(header.data_start + t.offset) for t in header.tensors[:3]) + '\n')
        (self.pack / 'dense.bin').write_bytes(bytes(512) + converted)
        record = {
            'name': source.name, 'shard': self.target.name, 'src_type': 'Q8_0', 'dst_type': 'BF16',
            'method': 'dequantize->bf16 round-to-nearest-even (--compat-bf16)',
            'exact': False, 'max_abs_err': error, 'source_sha256': hashlib.sha256(q8).hexdigest(),
        }
        (self.pack / 'conversions.json').write_text(json.dumps({
            'schema': 1, 'tool': 'tools/iq_pack.py', 'tensors': [record],
        }))
        (self.pack / 'compat-bf16.json').write_text(json.dumps({
            'source': str(self.target.resolve()), 'rounding': 'nearest-even',
            'tensors': [{'name': source.name, 'source_type': 'Q8_0', 'bytes': len(converted)}],
        }))
        packed_rows = [self.row('blk.0.ffn_gate_inp.weight', 2, 0, 256, 0, 256, 32, 2),
                       self.row('norm.weight', 2, 256, 4, 256, 4, 1, 0),
                       'output.weight 0 0 0 0 512 0 32 2 8 0 32 ' + ' '.join(['0'] * 7),
                       self.row(source.name, 4, 512, len(converted), 512, len(converted), 32, 0)]
        (self.pack / 'index.txt').write_text(
            '# strata pack index v3 -- standalone\n# align 256 pool 768 tensors 4\n'
            + '\n'.join(packed_rows) + '\n')
        self.document['target'] = [self.target.name]
        self.document['sha256'] = {
            'pack/conversions.json': hashlib.sha256((self.pack / 'conversions.json').read_bytes()).hexdigest(),
            'pack/compat-bf16.json': hashlib.sha256((self.pack / 'compat-bf16.json').read_bytes()).hexdigest(),
        }
        self.write_policy()
        self.args = ['--native', str(self.target), '--pack', str(self.pack), '--spec', '4']

    def test_opt_out_does_not_inspect_inputs(self):
        self.assertIsNone(enforce_launch_policy(['--mtp', 'missing'], 'missing', {}))

    def test_q8_rounding_reference_chunking_and_bad_sources(self):
        # 257/256 times +/-1 are BF16 halfway cases; +/-3 rounds upward. Expected
        # words are hand-derived ties-to-even, not produced by the validator.
        try:
            import numpy
        except ImportError:
            self.skipTest('numpy required for Q8 compatibility fixtures')
        path = self.root / 'blocks.bin'
        block = struct.pack('<e', 257 / 256) + struct.pack('32b', 1, 3, -1, -3, *([0] * 28))
        expected = struct.pack('<32H', 0x3f80, 0x4041, 0xbf80, 0xc041, *([0] * 28))
        path.write_bytes(block * 16385)
        digest, error = _q8_0_bf16_digest(path, 0, 34 * 16385)
        self.assertEqual(digest, hashlib.sha256(expected * 16385).digest())
        self.assertEqual(error, 1 / 256)
        for data, count in ((block[:-1], 34), (struct.pack('<e', float('inf')) + bytes(32), 34)):
            path.write_bytes(data)
            with self.assertRaises(PrecisionPolicyError):
                _q8_0_bf16_digest(path, 0, count)

    def test_reviewed_q8_0_to_bf16_conversion_is_deterministic_and_bounded(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('numpy is not installed in this test interpreter')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'q8.bin'
            codes = np.array([0, 1, -1, 127, -128] + [0] * 27, dtype=np.int8)
            path.write_bytes(struct.pack('<e', 0.25) + codes.tobytes())
            digest, err = _q8_0_bf16_digest(path, 0, 34)
            values = np.float32(0.25) * codes.astype(np.float32)
            bits = values.view(np.uint32)
            expected = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2').tobytes()
            self.assertEqual(digest, hashlib.sha256(expected).digest())
            decoded = (np.frombuffer(expected, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
            self.assertAlmostEqual(err, float(np.max(np.abs(decoded.astype(np.float64) - values.astype(np.float64)))))
            with self.assertRaises(PrecisionPolicyError):
                _q8_0_bf16_digest(path, 0, 33)

    def test_q8_compat_pack_manifest_boundary_and_source_payload_binding(self):
        self.install_q8_compat_pack()
        report = self.run_guard()
        self.assertEqual(report['pack_audit']['q8_0_to_bf16_conversions'], 1)
        self.assertGreater(report['pack_audit']['q8_0_to_bf16_max_abs_error'], 0.0)

        conversions = self.pack / 'conversions.json'
        compat = self.pack / 'compat-bf16.json'
        original_conversion = json.loads(conversions.read_text())
        original_compat = json.loads(compat.read_text())
        mutations = [
            (conversions, {**original_conversion, 'tensors': [{**original_conversion['tensors'][0], 'exact': True}]}, 'lacks matching'),
            (conversions, {**original_conversion, 'tensors': [{**original_conversion['tensors'][0], 'shard': 'other.gguf'}]}, 'lacks matching'),
            (compat, {**original_compat, 'tensors': [{**original_compat['tensors'][0], 'source_type': 'Q4_0'}]}, 'lacks matching'),
            (conversions, {**original_conversion, 'tensors': []}, 'lacks matching'),
            (compat, {**original_compat, 'tensors': []}, 'lacks matching'),
            (compat, {**original_compat, 'tensors': original_compat['tensors'] * 2}, 'duplicate'),
            (conversions, {**original_conversion, 'tensors': [{**original_conversion['tensors'][0], 'source_sha256': '0' * 64}]}, 'lacks matching'),
            (conversions, {**original_conversion, 'tensors': [{**original_conversion['tensors'][0], 'max_abs_err': float('nan')}]}, 'lacks matching'),
            (conversions, {**original_conversion, 'tensors': [{**original_conversion['tensors'][0], 'max_abs_err': float('inf')}]}, 'lacks matching'),
            (compat, {**original_compat, 'tensors': original_compat['tensors'] + [{'name': 'other', 'bytes': 64}]}, 'do not exactly match'),
        ]
        for path, document, message in mutations:
            with self.subTest(path=path.name, message=message):
                conversions.write_text(json.dumps(original_conversion))
                compat.write_text(json.dumps(original_compat))
                path.write_text(json.dumps(document))
                self.document['sha256']['pack/conversions.json'] = hashlib.sha256(conversions.read_bytes()).hexdigest()
                self.document['sha256']['pack/compat-bf16.json'] = hashlib.sha256(compat.read_bytes()).hexdigest()
                self.document['sha256'][f'pack/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.write_policy()
                with self.assertRaisesRegex(PrecisionPolicyError, message):
                    self.run_guard()
        conversions.write_text(json.dumps(original_conversion))
        compat.write_text(json.dumps(original_compat))
        self.document['sha256']['pack/conversions.json'] = hashlib.sha256(conversions.read_bytes()).hexdigest()
        self.document['sha256']['pack/compat-bf16.json'] = hashlib.sha256(compat.read_bytes()).hexdigest()
        self.write_policy()
        with (self.pack / 'dense.bin').open('r+b') as f:
            f.seek(512)
            f.write(bytes([1]) + bytes(63))
        with self.assertRaisesRegex(PrecisionPolicyError, 'differs'):
            self.run_guard()

    def test_exact_q8_compat_conversion(self):
        self.install_q8_compat_pack()
        header = M.Header(self.target)
        source = next(t for t in header.tensors if t.name == 'blk.0.attn_q.weight')
        raw = struct.pack('<e', 0.25) + struct.pack('32b', 1, -1, *([0] * 30))
        with self.target.open('r+b') as f:
            f.seek(header.data_start + source.offset)
            f.write(raw)
        with (self.pack / 'dense.bin').open('r+b') as f:
            f.seek(512)
            f.write(struct.pack('<32H', 0x3e80, 0xbe80, *([0] * 30)))
        path = self.pack / 'conversions.json'
        document = json.loads(path.read_text())
        document['tensors'][0].update(exact=True, max_abs_err=0.0,
                                     source_sha256=hashlib.sha256(raw).hexdigest())
        path.write_text(json.dumps(document))
        self.document['sha256']['pack/conversions.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_policy()
        self.assertEqual(self.run_guard()['pack_audit']['q8_0_to_bf16_max_abs_error'], 0.0)

    def test_q8_compat_pack_rejects_malformed_manifest_shape_and_hash_tampering(self):
        self.install_q8_compat_pack()
        compat = self.pack / 'compat-bf16.json'
        compat.write_text('[]')
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard()
        self.install_q8_compat_pack()
        conversions = self.pack / 'conversions.json'
        conversions.write_text('[]')
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard()
        self.install_q8_compat_pack()
        compat = self.pack / 'compat-bf16.json'
        compat.write_text(compat.read_text() + ' ')
        with self.assertRaisesRegex(PrecisionPolicyError, 'identity mismatch'):
            self.run_guard()

    def test_complete_binding_passes(self):
        result = self.run_guard()
        self.assertTrue(result['enforced'])
        self.assertTrue(result['pack_audit']['packed_payload_verified_against_source'])
        self.assertEqual(result['draft'], 'disabled')

    def test_packer_64_alignment_and_pool_accounting(self):
        path = self.pack / 'index.txt'
        original = path.read_text()
        path.write_text(original.replace('align 256 pool 512', 'align 64 pool 320')
                        .replace('output.weight 0 0 0 0 512', 'output.weight 0 0 0 0 320'))
        self.assertTrue(self.run_guard()['enforced'])
        valid = path.read_text()
        for bad in (valid.replace('align 64', 'align 32'),
                    valid.replace('pool 320', 'pool 512'),
                    valid.replace('norm.weight 0 2 256', 'norm.weight 0 2 257')):
            path.write_text(bad)
            with self.assertRaises(PrecisionPolicyError):
                self.run_guard()

    def test_reaudits_on_every_call(self):
        self.run_guard()
        (self.pack / 'dense.bin').write_bytes(b'x' + bytes(511))
        with self.assertRaisesRegex(PrecisionPolicyError, 'differs'):
            self.run_guard()

    def test_legacy_mtp_and_override_options_fail(self):
        for flag in ('--mtp', '--embd-gguf', '--native-head-gguf', '--native-dense-gguf', '--keep-canonical', '--mtp-q4', '--unknown'):
            with self.subTest(flag=flag), self.assertRaises(PrecisionPolicyError):
                self.run_guard(self.args + [flag, 'anything'])

    def test_duplicate_or_equals_syntax_fails(self):
        for extra in (['--native', str(self.target)], ['--native=' + str(self.target)]):
            with self.assertRaises(PrecisionPolicyError):
                self.run_guard(self.args + extra)

    def test_unknown_environment_override_fails_even_zero(self):
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard(env={**self.env, 'STRATA_MTP_Q4': '0'})

    def test_pack_source_override_is_denied(self):
        for name in ('experts.bin', 'extra.bin', 'embd.bin'):
            path = self.pack / name
            path.write_bytes(b'')
            with self.assertRaisesRegex(PrecisionPolicyError, 'opaque'):
                self.run_guard()
            path.unlink()

    def test_native_descriptor_offsets_cannot_redirect(self):
        path = self.pack / 'native_experts.txt'
        text = path.read_text().splitlines()
        cols = text[1].split()
        cols[5] = str(int(cols[5]) + 32)
        path.write_text(text[0] + '\n' + ' '.join(cols) + '\n')
        with self.assertRaisesRegex(PrecisionPolicyError, 'does not address'):
            self.run_guard()

    def test_index_cannot_hide_quantized_or_missing_rows(self):
        path = self.pack / 'index.txt'
        old = path.read_text()
        path.write_text(old.replace('norm.weight 0 2', 'norm.weight 0 0'))
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard()
        path.write_text('\n'.join(x for x in old.splitlines() if not x.startswith('norm.weight')))
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard()

    def test_external_ple_is_table_only_and_bound(self):
        ple = fixture(self.root / 'ple.gguf', [('per_layer_token_embd.weight', [160], 8, 0, 170)],
                      [('general.architecture', 8, 'strata-ple')])
        self.document['ple'] = 'ple.gguf'
        self.write_policy()
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard()
        self.assertTrue(self.run_guard(self.args + ['--ple-gguf', str(ple)])['enforced'])
        fixture(ple, [('per_layer_token_embd.weight', [160], 8, 0, 170), ('override', [32], 8, 192, 34)],
                [('general.architecture', 8, 'strata-ple')])
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard(self.args + ['--ple-gguf', str(ple)])

    def test_optional_hash_binds_actual_payload(self):
        self.document['sha256'] = {'model.gguf': hashlib.sha256(self.target.read_bytes()).hexdigest()}
        self.write_policy()
        self.assertEqual(self.run_guard()['identity_hashes_checked'], 1)
        self.document['sha256']['model.gguf'] = '0' * 64
        self.write_policy()
        with self.assertRaisesRegex(PrecisionPolicyError, 'identity mismatch'):
            self.run_guard()

    def test_policy_cannot_claim_learned_draft_disabled_if_supplied(self):
        self.document['draft'] = 'draft.gguf'
        self.write_policy()
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard()

    def test_empty_policy_is_not_opt_out(self):
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard(env={'STRATA_PRECISION_POLICY': ''})

    def test_bf16_pack_conversion_verified_against_f32(self):
        h = M.Header(self.target)
        norm = next(t for t in h.tensors if t.name == 'norm.weight')
        with self.target.open('r+b') as f:
            f.seek(h.data_start + norm.offset)
            f.write(struct.pack('<f', 1.0))
        with (self.pack / 'dense.bin').open('r+b') as f:
            f.seek(256)
            f.write(struct.pack('<H', 0x3f80))
        index = self.pack / 'index.txt'
        index.write_text(index.read_text().replace('norm.weight 0 2 256 4 256 4', 'norm.weight 0 4 256 2 256 2'))
        self.assertTrue(self.run_guard()['enforced'])
        with (self.pack / 'dense.bin').open('r+b') as f:
            f.seek(256)
            f.write(struct.pack('<H', 0x4000))
        with self.assertRaisesRegex(PrecisionPolicyError, 'differs'):
            self.run_guard()

    def test_expert_header_count_and_total_binding(self):
        path = self.pack / 'native_experts.txt'
        old = path.read_text()
        for text in (old.replace('n_expert 2', 'n_expert 3'), old.replace('total 3456', 'total 3000')):
            path.write_text(text)
            with self.assertRaises(PrecisionPolicyError):
                self.run_guard()

    def test_wrong_pack_binding_and_unreviewed_policy(self):
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard(['--native', str(self.target), '--pack', str(self.root)])
        self.document['allow_q2_draft'] = True
        self.write_policy()
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard()

    def test_float_runtime_narrowing_checked(self):
        self.document['minimum_bits'] = 32
        self.document['preferred_bits'] = 32
        self.write_policy()
        with self.assertRaises(PrecisionPolicyError):
            self.run_guard()


if __name__ == '__main__':
    unittest.main()
