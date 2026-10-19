# Extensible engine work: backend and precision foundation

This fork starts from upstream Strata commit
`477ac13cbe8e582c20484871ce459d20d82df09d`. It is an experimental development
branch, not a replacement for an installed Strata server.

## What this first change does

- Adds an opt-in backend interface and a thin adapter around an already-created
  Strata engine. The adapter delegates generation; it does not implement another
  model executor or start a process.
- Adds an offline GGUF tensor inventory and precision-policy auditor. The weight
  floor is four bits; eight bits or higher is preferred. The preference is
  reported separately from the required floor.
- Adds a combined offline preflight report, joining artifact precision results
  with the adapter's declared capabilities. Passing this report is not proof of
  numerical correctness, model completeness, kernel support or memory fit.

No existing server startup, model loader, CUDA/HIP/SYCL kernel, or default
generation path is changed. The normal server does not automatically enforce
the new manifest. Integration into actual model admission is a later change.

## Why start here

The runtime needs to separate model execution from resource management before
adding other architectures or changing GPU assignments. A narrow adapter lets
us test the boundary without replacing working execution. An explicit capability
report prevents unsupported operations from appearing to work.

Precision must be established from actual tensor formats, not a model filename
or average bits per parameter. Mixed artifacts can contain lower-bit tensors;
even a higher-precision target can use a lower-precision drafter. The inspected
upstream MTP pack uses Q2 routed experts. It does not meet this fork's strict
floor, so the compliant initial deployment must disable that drafter.

## Artifact audit

Run from the repository root, with Python 3.10 or newer:

```sh
python tools/model_manifest.py --help
python tools/model_manifest.py --target /path/to/target.gguf --output manifest.json
```

Pass all shards and external PLE/draft GGUF artifacts to the corresponding
arguments. Use `--require-component` when a component is required by the chosen
deployment. A component that was not supplied is not certified. Unsupported
raw runtime packs must not be represented as audited GGUF artifacts.

The default audit reads headers and checks file extents. Content hashing is
optional because hashing a large artifact reads all its bytes. A header-only
report is not a cryptographic content identity and must not serve as the key
for a trusted shared immutable store. Even full-file hashes establish byte
identity, not original quantization provenance or accuracy.

The auditor checks supplied artifacts, not every file an arbitrary launch
command might open. It cannot prove that a Q8 file was produced from a genuine
higher-precision source rather than upcast from a lower-quality artifact.
Record provenance separately. KV cache and activation precision are also
separate from learned-weight precision.

For an external PLE table and no draft model, the combined report is:

```sh
python tools/backend_preflight.py \
  --target /path/to/target-00001-of-00002.gguf /path/to/target-00002-of-00002.gguf \
  --ple /path/to/ple.gguf --ple-mode external --draft-mode disabled \
  --output preflight.json
```

Use `--ple-mode embedded` without `--ple` if the target includes the table.
Use `--draft-mode external --draft /path/to/draft.gguf` only for explicitly
supplied draft GGUF artifacts. Unsupported legacy runtime packs fail the audit.
Both tools return status 1 for failed checks and status 2 for invalid command
arguments or output errors. Success is status 0, but the combined report always
marks `launch_qualified: false`: runtime qualification has not occurred.

`--require-capability durable_checkpoint` currently fails explicitly. It cannot
be enabled by renaming an artifact or providing session metadata.

## Backend scope

The first adapter is for the existing Flash-Next architecture. Its model and
session identity types do not hard-code Flash-Next's QSA/GDN/PLE state into a
generic checkpoint format. Other architectures require their own adapters and
verified capabilities; an arbitrary model label does not add support.

Generation tests use an injected fake engine to compare token streams, request
arguments, cancellation and errors. They do not qualify native execution or
speed. The adapter does not promise concurrent safety beyond its wrapped
engine. Durable checkpoints, live migration and separately scheduled prefill
and decode steps are explicitly unsupported in this first version.

An application can opt into the adapter around an engine it already owns:

```python
from serve.backends import ModelIdentity, SessionMetadata, StrataBackend

# These identities come from the caller's audited artifacts and configuration.
# They are not accepted as proof of precision or runtime compatibility.
identity = ModelIdentity("qwen4exp", weights_fingerprint, tokenizer_fingerprint)
backend = StrataBackend(existing_engine, identity)
session = SessionMetadata("agent-one", identity)
stream = backend.generate(token_ids, max_new, sampling, cancel_event, session=session)
```

The caller still owns engine loading, resource admission, locks and draining.
The adapter does not make a single-session engine concurrent.

## Validation

From the repository root:

```sh
python -m unittest tools.test_model_manifest serve.test_backends tools.test_backend_preflight
```

The initial suite has 35 passing tests on Python 3.12.14, using synthetic GGUF
files and fake engines. Coverage includes low-bit draft rejection, unknown
tensor roles, shard completeness, malformed headers, payload extents, optional
hashing, input overwrite prevention, iterator cleanup, cancellation and explicit
unsupported operations. CI is configured for Python 3.10 and 3.12 on a CPU runner.
No CUDA/HIP/SYCL files were changed or GPU backend builds run.

## Next implementation gates

1. Bind audited artifact identity and runtime transformations to actual model
   admission, with a capability matrix for required operators and formats.
2. Add resource reservations and worker/session generations around the existing
   serving path, without changing numerical execution.
3. Harden shared expert-store publication and attach semantics.
4. Add complete backend-specific checkpoints and durable graceful lane shutdown.
   A transcript is not a warm execution-state checkpoint.
5. Qualify a second architecture early, then test useful two-session-per-GPU
   concurrency and three-lane to one-lane transitions with the benchmark runner.

Keep GPU validation with the runner. CPU-side unit tests here do not establish
SM70 compatibility, Q8 model fit, checkpoint durability or performance gains.

## Repository workflow

`upstream` tracks `https://github.com/Niko1221/Strata.git`; `origin` is the owner's
fork. Keep changes opt-in and small so upstream fixes can be incorporated.
Preserve upstream and dependency notices. Do not rewrite the running benchmark
checkout or install this branch into active lanes.
