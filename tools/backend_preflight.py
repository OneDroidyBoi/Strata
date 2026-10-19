#!/usr/bin/env python3
"""Offline artifact-policy and backend-capability report. Never starts an engine.

Successful exit validates only the supplied inventory/configuration, not native
kernel support, complete model tensors, memory fit, quality, or safe deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.backends import StrataBackend
from tools.model_manifest import build_manifest


def build_preflight(components, *, ple_mode, draft_mode, minimum_bits=4,
                    preferred_bits=8, content_hash=False, required_capabilities=()):
    if ple_mode not in ('embedded', 'external') or draft_mode not in ('disabled', 'external'):
        raise ValueError('declare PLE embedded/external and draft disabled/external')
    required = ['target']
    if ple_mode == 'external':
        required.append('ple')
    if draft_mode == 'external':
        required.append('draft')
    manifest = build_manifest(components, minimum_bits, preferred_bits, content_hash, tuple(required))
    errors = []
    target = manifest['components'].get('target', {})
    files = target.get('files', [])
    architectures = {f.get('architecture') for f in files}
    if not files or architectures != {'qwen4exp'}:
        errors.append('every target shard must declare general.architecture=qwen4exp for this adapter')
    names = {t['name'] for t in target.get('tensors', [])}
    if ple_mode == 'embedded':
        if components.get('ple'):
            errors.append('embedded PLE mode conflicts with external PLE files')
        if 'per_layer_token_embd.weight' not in names:
            errors.append('embedded PLE table was not found in the supplied target tensors')
    else:
        ple_names = {t['name'] for t in manifest['components'].get('ple', {}).get('tensors', [])}
        if 'per_layer_token_embd.weight' not in ple_names:
            errors.append('external PLE component must contain per_layer_token_embd.weight')
    if draft_mode == 'disabled' and components.get('draft'):
        errors.append('disabled draft mode conflicts with supplied draft files')
    caps = asdict(StrataBackend.capabilities)
    for capability in required_capabilities:
        # Only explicit boolean capabilities can qualify an operation. None is
        # unverified and counts/slot limits are not boolean feature support.
        if caps.get(capability) is not True:
            errors.append(f'backend capability {capability!r} is unavailable or unverified')
    return {
        'schema_version': 1,
        'backend': 'strata',
        'capabilities': caps,
        'declared_configuration': {'ple_mode': ple_mode, 'draft_mode': draft_mode},
        'artifact_manifest': manifest,
        'configuration_errors': errors,
        'offline_checks_passed': manifest['policy_passed'] and not errors,
        'launch_qualified': False,
        'not_evaluated': ['complete_model_inventory', 'runtime_transformations',
                          'native_kernel_support', 'memory_fit', 'numerical_quality',
                          'throughput', 'durable_checkpoint_or_migration'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', nargs='+', required=True)
    parser.add_argument('--ple', nargs='+')
    parser.add_argument('--draft', nargs='+')
    parser.add_argument('--ple-mode', required=True, choices=('embedded', 'external'))
    parser.add_argument('--draft-mode', required=True, choices=('disabled', 'external'))
    parser.add_argument('--minimum-bits', type=int, default=4)
    parser.add_argument('--preferred-bits', type=int, default=8)
    parser.add_argument('--content-hash', action='store_true')
    parser.add_argument('--require-capability', action='append', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    components = {k: getattr(args, k) or [] for k in ('target', 'ple', 'draft')}
    try:
        if args.output:
            for paths in components.values():
                for path in paths:
                    source = Path(path)
                    if (args.output.resolve() == source.resolve()
                            or (args.output.exists() and source.exists() and args.output.samefile(source))):
                        parser.error('output must not overwrite an input artifact')
        result = build_preflight(components, ple_mode=args.ple_mode, draft_mode=args.draft_mode,
                                 minimum_bits=args.minimum_bits, preferred_bits=args.preferred_bits,
                                 content_hash=args.content_hash, required_capabilities=args.require_capability)
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(text, encoding='utf-8')
        else:
            sys.stdout.write(text)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0 if result['offline_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
