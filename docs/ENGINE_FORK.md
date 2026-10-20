# Extensible engine work: backend and precision foundation

This fork starts from upstream Strata commit
`477ac13cbe8e582c20484871ce459d20d82df09d`. It is an experimental development
branch, not a replacement for an installed Strata server.

## What this foundation does

- Adds an opt-in backend interface and a thin adapter around an already-created
  Strata engine. The adapter delegates generation; it does not implement another
  model executor or start a process.
- Adds an offline GGUF tensor inventory and precision-policy auditor. The weight
  floor is four bits; eight bits or higher is preferred. The preference is
  reported separately from the required floor.
- Adds a combined offline preflight report, joining artifact precision results
  with the adapter's declared capabilities. Passing this report is not proof of
  numerical correctness, model completeness, kernel support or memory fit.
- Enforces an optional precision policy before native construction and restart,
  and explicit resource reservations around the supported request path.
- Adds durable checkpoint storage and an injected-worker persist-and-stop
  lifecycle, plus native live/parked/batch-slot mappings for supported layouts.
  GPU qualification remains a separate gate.

The backend adapter, launch precision policy, resource ledger and checkpoint
controller are separate opt-in components. None changes numerical kernels or
qualifies a new model executor. Runtime policy hooks must use the actual launch
arguments and child environment on every spawn, including restarts; an offline
preflight report alone does not enforce a running server.

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

## Launch precision enforcement

`serve.precision_policy.enforce_launch_policy(args, cwd, env)` returns `None`
when `STRATA_PRECISION_POLICY` is absent. When set, it freshly validates a JSON
policy and the actual launch, returning an enforcement report or raising
`PrecisionPolicyError` before process creation. `args` contains engine arguments,
not the executable; `env` must be the merged environment the child will receive.
The caller must invoke it on every spawn, including automatic restart.

A policy file has this shape; paths are relative to the policy file:

```json
{
  "schema_version": 1,
  "target": ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"],
  "pack": "pack",
  "ple": "embedded",
  "draft": "disabled",
  "minimum_bits": 4,
  "preferred_bits": 8
}
```

For external PLE, replace `embedded` with the standalone table GGUF path and bind
it explicitly in the launch using `--ple-gguf`. Optional `sha256` maps admitted
artifact paths to expected full-file digests. A Q8 minimum can be requested with
`minimum_bits: 8`; Q8 preference alone does not prohibit Q4.

This guard admits a deliberately narrow native-pack subset. It binds the native
shard set, pack path, PLE source and explicit disabled draft. It verifies native
expert descriptors and packed float payloads against audited source spans or
reviewed float conversions. Opaque overriding runtime payloads, unreviewed
options/environment overrides and lower-bit weights fail closed. This is why an
otherwise legitimate upstream configuration may be refused: support must be
reviewed before widening the allowlist, not bypassed by ignoring its error.

The pack index may declare 64-byte (the packer default) or 256-byte alignment;
offsets and pool accounting must match that declaration. Explicit Q8_0-to-BF16
`--compat-bf16` conversion is also admitted with NumPy installed, matching
conversion/compatibility manifests and a fresh byte-level source/output check.
The guard recomputes rounding error and exactness; it does not accept arbitrary
quantized-to-float transformations. See [patched GPU qualification](GPU_MIGRATION_VALIDATION.md#patched-precision-qualification-2026-10-19)
for the artifact contract and separate commit references.

The current learned MTP pack is below the strict Q4 floor and is not admitted.
Strict-policy startup also rejects vision/image sidecars before constructing them;
their weight sources need separate auditing before that combination is supported.
The guard does not establish kernel compatibility, Q8 fit or original numerical
provenance. Files must remain immutable throughout validation and execution;
this is not protection against a concurrent local writer replacing files after
checking them. Large pack verification can perform substantial disk reads, so
it must not be run during another lane's timed benchmark without coordination.

## Resource admission

`serve.resource_admission.ResourceLedger(host_bytes, gpu_bytes)` provides
process-local reservations. `gpu_bytes` maps GPU IDs to configured byte budgets.
Its public sequence is:

1. `register_worker(worker_id, gpu_id, max_slots, fixed_host_bytes, fixed_gpu_bytes)`
   returns a generation-specific `WorkerHandle` and charges fixed costs.
2. `acquire(handle, host_bytes, gpu_bytes, context_tokens)` atomically checks slot
   and memory budgets and returns a `Reservation`, or raises `AdmissionRejected`.
3. `release(reservation, iterator_closed=True, work_drained=True)` retires it only
   after both conditions are positively established. Duplicate/stale/altered
   reservations do not free another request's charge.
4. `begin_drain(handle)` blocks new admissions. `wait_quiescent(handle, timeout)`
   waits for outstanding reservations without cancelling work. `unregister_worker`
   removes fixed costs only after draining and quiescence. The caller must also
   confirm native shutdown before removing charges for actual resident memory.

`snapshot()` supplies detached accounting data for diagnostics. The ledger does
not inspect physical RAM/VRAM, constrain native allocations or coordinate other
processes. Configured capacities must already exclude operating headroom and
other workloads. Fixed costs must include retained/cached session state, loaded
weights, preallocated slots and fixed scratch. Request costs cover incremental
peaks, including copies; do not count a preallocated slot twice or release its
resident memory charge when a request finishes.

The server now wires `serve.request_admission.configured_admission` into
`Service.run`. Enable it with the config key **`resource_admission`** (the module
name is `request_admission`), with exactly these required fields:

```text
host_capacity_bytes, gpu_capacity_bytes, gpu_id, max_requests,
fixed_host_bytes, fixed_gpu_bytes, host_request_bytes, gpu_request_bytes,
host_bytes_per_token, gpu_bytes_per_token,
max_context_tokens, lookahead_tokens
```

All byte/token/count fields must be explicit; `gpu_id` identifies the one GPU.
There are no universal safe sample values. Calibrate fixed and per-request peaks
for the exact model, precision, context and engine configuration before enabling
it. The token budget includes prompt, permitted output and lookahead; per-token
estimates multiply that budget in addition to base request costs. A nonpositive
output limit reserves the configured maximum context. `RequestAdmission.run`
closes its iterator and calls the supplied quiescence check before release; an
uncertain drain or execution exception retains the reservation and gates further
admission. Recovery requires explicit inspection; capacity is never silently reused.

This integration is **one GPU, one lane, text-only**, with native batched
admission enabled through per-request drain receipts. Vision/images and replicas
remain rejected. Each receipt follows the captured process generation and slot
through all internal request legs. Iterator closure seals the receipt; the
reservation stays charged until every leg has its terminal acknowledgement or
the captured process has actually exited. EOF, an unrelated slot's BDONE and a
replacement process do not prove completion. Pending cancellations keep their
slots unavailable; failed or timed-out drains gate further admission. Other
requests can continue within the remaining budget while a valid drain is pending.
The native protocol has no request nonce: exclusivity of the captured slot queue
and withholding slot reuse are part of this proof. It is a conservative rejection gate,
not a GPU scheduler, a cross-process shared-memory broker or proof of physical
fit. Holding a reservation until an HTTP iterator closes is insufficient if the
engine still drains a batch slot in the background; native quiescence must be
confirmed separately before release. Refuse unsupported combinations rather than
quietly assuming their resource costs.

## Durable storage and graceful lane lifecycle

`serve.checkpoint_store.CheckpointStore(root)` requires an existing trusted local
directory with no symlink traversal. The installer owns making that root durable.
`publish(model, sessions, writer)` calls `writer(session, path)` for each exact
`SessionMetadata`; the writer streams a backend-specific file. It never passes a
whole model/state byte string through the controller. The store streams SHA256,
fsyncs each payload, publishes a versioned JSON manifest by atomic rename, fsyncs
the bundle directory, renames the staging bundle and fsyncs the root before
reporting success. Failed staging directories remain for diagnosis and are not
restorable published bundles.

`store.sessions(checkpoint_id, model)` reads metadata. `store.restore(checkpoint_id,
session_id, model, reader)` validates version, model identity, safe filenames,
size and checksum before invoking `reader(session, path)`. Only the requested
payload is read for lazy restoration. Published bundles must remain immutable.
Checksums detect accidental corruption; they do not authenticate an untrusted
writer or prove that a backend exported all required state. The store deliberately
uses JSON rather than executable serialization. It caps manifest size and session
count, but disk-capacity reservation and retention cleanup remain caller concerns.

`serve.lane_lifecycle.LaneLifecycle(worker, store, model)` accepts an injected
worker implementing:

```text
gate_admission()
wait_drained(timeout_s) -> bool
list_sessions() -> Sequence[SessionMetadata]
export_session(session, path)
unload()
restore_session(session, path)
```

`persist_and_stop(timeout_s)` gates admission, waits for all device work and
cancellation drains, enumerates every retained session, publishes the complete
bundle, verifies that the inventory stayed unchanged, and only then calls
`unload()`. The result has `state`, `checkpoint_id` and `error`. A timeout or export,
checksum, disk, rename or fsync failure returns `blocked`; it does not force-cancel,
kill or unload the worker. Admission remains gated for explicit operator recovery.
An unloader/exporter may itself fail its worker, so `blocked` is not a guarantee
that the worker is alive. An unload failure retains the durable bundle ID.

`lazy_restore(checkpoint_id, session_id, timeout_s)` operates on a caller-provided
ready worker, gates and drains it, validates the selected checkpoint and calls
its restore method. It does not start a model or reopen admission. A failed native
restore can have mutated state: `blocked` is not permission to resume serving or
retry blindly. After a successful stop, construct a new lifecycle around the new
ready worker rather than pretending the old process has restarted.

### Native inventory, export and shutdown

The opt-in native implementation uses `STRATA_SESSION_EXPORT=1` and advertises
`INFO session_export=1`. `INVENTORY` returns a process/epoch generation and every
live, parked or retained batch-slot session in the supported layout.
`EXPORT <generation> <id> <path>`
rejects stale generations and writes that exact session without evicting others.
Live export retains every checkpoint and disables SAVE's optional reclamation;
parked export reads its existing host image directly. Ordinary commands invalidate
the inventory generation. No transcript or token text appears in the inventory.

Supported native checkpoint layouts are **single-GPU qwen4exp, unbatched or one
batch group, with prompt caching enabled and learned MTP disabled**. A supporting
batch binary advertises `INFO session_export_batch=1`; old binaries are rejected.
Slots have IDs `slot-N` and inventory kind `slot`. The exporter reads each slot's
own session buffers directly, preserving the live session and every other slot.
The target-only serialization path omits the absent draft KV layer consistently
in capture, read limits, validation and restoration. Legacy opt-out calls retain
their existing draft-reference behavior.

Active, partial, tainted, image-bearing, half-written or unsupported slots reject
the entire inventory. Empty unused slots need no file; every usable retained slot
does. Layer splits, peer GPUs and multiple batch groups remain unsupported.
Native CPU tests round-trip stored KV/recurrent/PLE/tail state; the GPU capture
and restore paths have not yet been built or qualified on CUDA, HIP or SYCL here.
The generic backend capability flags remain false pending that qualification.

`NativeLaneWorker(service, model, enabled=True)` connects the implementation to
the durable lifecycle. It requires an enforced Q4-minimum/Q8-preferred source and
pack audit, a matching `engine.model_identity`, and the service's actual admission
controller. The owner supplies an immutable complete weight/configuration and
tokenizer identity after audit and READY; metadata alone cannot attest identity.

```python
from serve.native_lane import NativeLaneWorker
from serve.checkpoint_store import CheckpointStore

# A ready audited engine, calibrated admission controller, and owner-verified
# ModelIdentity must already exist. The store directory must already be durable.
service.engine.model_identity = model_identity
worker = NativeLaneWorker(service, model_identity, enabled=True)
result = worker.persist_and_stop(CheckpointStore(checkpoint_directory), timeout_s=30)
# Only result.state == "stopped" confirms publication and process exit.
```

The worker gates requests, drains leases, holds the service FIFO and (for batch
engines) the batch control lock, waits for slot-cleanup bookkeeping, exports all
sessions, rechecks the generation, publishes the bundle and confirms process exit
before retiring fixed memory charges. Ordinary load/unload, slot-file and VRAM
controls are gated too. Failed export or protocol framing leaves the lane gated;
protocol failure does not kill the engine or permit a legacy restore retry.

Restoration supports a ready compatible unbatched worker through `lazy_restore`,
or an empty slot in a compatible batch worker through
`restore_slot(store, checkpoint_id, session_id, slot, timeout_s=30)`. The latter
requires the separate native advertisement `INFO session_restore_batch=1`.
`RESTORE_SLOT <slot> <path>` reads and validates the file against the destination
slot before transfer. It does not use the live session as scratch space or evict
another cached slot. A failed transfer quarantines the destination and keeps
admission closed; other retained sessions stay loaded.

Restore one or more selected sessions while admission is closed, then explicitly
call `worker.resume(timeout_s=30)`. The worker checks native state, process identity,
slot ownership and precision before reopening the same admission worker. Fixed
memory charges and registration generation remain intact. A failed restore blocks
resume. The native response supplies the exact retained token IDs in bounded
chunks, so ordinary prefix matching can route the next prompt back to its restored
slot. These IDs are internal protocol data and must not be logged as user-facing
status. Published checkpoint files remain available after restoration.

For example, with two selected session IDs and a ready batch destination:

```python
store = CheckpointStore(checkpoint_directory)
worker = NativeLaneWorker(destination_service, model_identity, enabled=True)
for slot, session_id in enumerate(selected_session_ids):
    result = worker.restore_slot(store, checkpoint_id, session_id, slot)
    if result.state != "restored":
        raise RuntimeError(result.error)  # leave gated; do not reopen manually
result = worker.resume()
if result.state != "resumed":
    raise RuntimeError(result.error)
```

The owner chooses the source session and destination slot. This is a Python owner
API. The migration coordinator below adds explicit agent ownership and three-to-one
consolidation; HTTP session routing and automatic signal handlers remain separate. Restore
requires the same model, context/KV layout and other state-compatible configuration.
Existing cached slots are not overwritten; unused slots must already exist in the
configured batch, with their residency included in calibrated fixed memory costs.
The fixed host budget must also cover the peak full checkpoint read image and
retained host checkpoints during restore. The native available-RAM floor provides
a second check; it does not reserve that scratch space in the resource ledger.
Requests after resume still pass the usual per-request resource admission checks.
Checkpoints preserve reusable computational prefixes between requests. They do not
save active-generation sampling/counters, the last emitted but not yet consumed
token, cache pinning policy or a partial prompt's execution plan; they do not resume
a sampled reply midway through it.

## Cross-lane migration (opt-in owner API)

`MigrationCoordinator` binds named agents to explicit slots on already-running,
identity-validated `NativeLaneWorker` instances. All lanes must load the same
state-compatible model/configuration and retain Q4-minimum/Q8-preferred audits.
Native binaries must advertise `session_release_batch=1` and
`session_route_batch=1` in addition to batch export/restore. Older binaries are
rejected rather than silently ignoring slot ownership. No model process is started
by the coordinator. Use the same trusted local
`CheckpointStore` for every worker. The owner must route all traffic through the coordinator or the HTTP gateway
below, without concurrent unbound traffic or control-plane changes.

`consolidate(destination)` waits for owned turns and native drain acknowledgements,
then saves **every lane, including the destination, without unloading**. Each bundle
and the agent-to-checkpoint journal are synced to disk before source shutdown.
Only verified source-process exit retires that lane's fixed memory charges.
The destination retains its resident agents and fills empty slots from sources.
Overflow agents stay parked on disk, with exact model, bundle and session bindings.
Unmanaged live/parked/native slot state is included in the snapshots too; it is
never silently treated as an empty destination slot.

`activate(agent)` brings a parked agent back on demand. With a full destination,
it first saves the latest state of all resident agents and syncs the updated
journal. It then releases the least-recently-used managed slot, restores the
requested checkpoint into that slot, and explicitly resumes admission. Corrupt
incoming checkpoints are rejected before eviction. This is turn-boundary rotation:
other active turns drain before a context switch, and the existing slot allocation
is reused. Slot release does not return GPU memory; stopping a source lane does.

```python
from serve.lane_migration import MigrationCoordinator

coordinator = MigrationCoordinator(
    {"lane-a": worker_a, "lane-b": worker_b, "lane-c": worker_c},
    store,
    {"agent-a": ("lane-a", 0), "agent-b": ("lane-b", 0),
     "agent-c": ("lane-c", 0)},
    enabled=True,
)
result = coordinator.consolidate("lane-c")
if result.state != "consolidated":
    raise RuntimeError(result.error)
# Each iterator must be consumed or closed by its owner.
for token in coordinator.generate("agent-a", prompt_ids, max_new, sampling, cancel):
    consume(token)
```

`generate` combines exact slot binding, per-request resource admission and native
drain receipts. Multiple different resident agents can run concurrently. A bound
request waits for its own slot; it cannot spill into another agent's slot or use
the unbound solo path. Owned admission also publishes completed short replies
into their assigned slot before acknowledging completion. Ambiguous/partial
owned admissions fail closed; a cancelled partial prefill may therefore block
migration until explicit recovery, rather than silently checkpoint stale context.
The caller still constructs the normal next-turn tokenized
prompt; restoration preserves cached computation, not an unfinished sampled reply.
The coordinator itself exposes no HTTP endpoint; the optional gateway below
binds HTTP requests to these owner operations. Passing an agent/slot association is an owner assertion about the running
session; checkpoint metadata cannot infer which external agent owns it.

A failure gates remaining lanes and leaves immutable bundles plus the last synced
`migration-*.json` journal available for owner recovery. Process replacement is
rejected instead of silently accepting empty context. No automatic rollback,
restart or crash-recovery replay is attempted. The journal references the latest
successful snapshot, **not tokens generated since it**. Source lanes remain
stopped after successful consolidation; scaling back out is separate work.

If an owned turn has not finished within the routing wait, `activate` or
`consolidate` returns `waiting` without closing admission; it can be retried.
The token iterator raises retryable `MigrationBusy` in that case. Failures after
native state work begins remain gated. The routing timeout also bounds acquisition of the coordinator operation lock;
lock contention returns retryable busy without changing native state. Time spent
waiting for that lock is deducted from the subsequent active-turn wait. Native
drain waits remain individually bounded; this is not a total disk-transfer deadline.
Storage must accommodate all snapshots, including later rotations; automatic
checkpoint garbage collection and disk quotas are not implemented. Calibrated host
budgets must include retained state and checkpoint read scratch. This coordinator
reuses existing lane allocations; it is not a multi-GPU capacity planner. GPU
continuation parity and measured three-to-one latency still require an isolated
runner. No running benchmark lanes were used for these tests.

## HTTP migration gateway

`MigrationGateway` adds one stable HTTP address for an owner-created coordinator.
Attach it **before starting any managed Service HTTP handler**. It uses already
loaded, audited native workers and never discovers, loads, or restarts model lanes.
The same single-GPU, one-batch-group, text-only restrictions apply. Server-side
vision, image generation and MCP integrations are rejected for these managed lanes.
Ordinary client tool-call messages remain supported.

```python
from serve.migration_http import MigrationGateway

# coordinator was constructed with explicit agent-to-slot bindings as above.
# Read both credentials from your existing secret configuration.
frontend = worker_a.service
frontend.api_key = client_api_key
http = MigrationGateway(coordinator, enabled=True, admin_token=operator_key)
server = http.serve(frontend, host="127.0.0.1", port=8095)
```

Clients use the existing `/v1/chat/completions`, `/v1/messages` or `/v1/responses`
endpoint, with normal authentication and one `X-Strata-Agent-ID` header containing
a pre-enrolled agent name. Both streaming and non-streaming requests are supported.
The gateway selects or restores that agent's slot automatically, and holds ownership
through prompt preparation, the entire response, internal continuations and cleanup.
It uses the destination Service's existing per-request admission and drain receipts;
there is no second reservation layer. Each request still supplies its normal message
history; warm state avoids recomputing cached prefixes. Agent names are routing IDs,
not separate credentials or a multi-tenant security boundary.

For example, an OpenAI-compatible client can set
`extra_headers={"X-Strata-Agent-ID": "agent-a"}` on a request; an Anthropic client
can send the same HTTP header. Reuse the agent ID for that conversation. Missing,
malformed, duplicate or unknown IDs are rejected before native routing. IDs cannot
select checkpoint files, raw slots, model identities or new registry entries.

An operator reduces three lanes to one with:

```http
POST /v1/migration/consolidate
Authorization: Bearer <client API key>
X-Strata-Migration-Key: <separate operator key>
Content-Type: application/json

{"destination":"lane-c","timeout_s":30}
```

Success returns `{"status":"consolidated"}`. Requests to parked agents thereafter
restore them on demand; clients do not send checkpoint or restore commands.
The operator selects the remaining lane explicitly. There is no automatic GPU-demand
monitor or policy that chooses when to shrink. Repeating a successful request for
the same destination is safe; changing the destination after consolidation is
rejected without disturbing the remaining lane. Omitting `admin_token` disables
HTTP consolidation while retaining agent routing.

A busy route or drain returns HTTP 503 with a retry hint when retry is safe. Invalid
client input is rejected without gating unrelated lanes; a failed native migration
keeps its existing fail-closed behavior and requires owner recovery. Client disconnects
close the response iterator and retain reservations until native work drains. Waiting
for migration may complete a safe checkpoint/restore before cancellation is observed;
it does not interrupt a disk or GPU state transfer halfway through it.

Ordinary load/unload, VRAM, slot-file, settings/configuration and image-generation
HTTP controls are disabled on managed listeners. Idle unload and automatic model
restart are also blocked for managed services. Non-migration services retain their
existing behavior. A non-loopback gateway bind requires a frontend API key; the
operator key is separately checked and never returned by the HTTP API. Journal
paths and internal native errors are not exposed by migration administration.

HTTP tests run against local fake lanes and real request handlers. GPU context
continuation and end-to-end performance remain unqualified; no running lanes were
accessed or reconfigured for this integration.

## Validation

From the repository root, run the CPU-only suites relevant to these components:

```sh
python -m pip install jinja2==3.1.6 pytest==8.4.2 numpy==2.2.6
python -m pytest -q tools/test_model_manifest.py serve/test_backends.py tools/test_backend_preflight.py serve/test_precision_policy.py serve/test_runtime_precision.py serve/test_resource_admission.py serve/test_request_admission.py serve/test_lane_lifecycle.py
python -m pytest -q serve/test_slot_drain.py serve/test_session_protocol.py serve/test_native_lane.py serve/test_native_lane_controls.py serve/test_batch_checkpoint.py serve/test_batch_restore.py serve/test_bound_slot.py serve/test_lane_migration.py serve/test_migration_http.py serve/test_migration_http_integration.py
python -m pytest -q serve/test_quiet_cancel.py serve/test_request_accounting.py serve/test_replicas.py serve/test_lifecycle.py
```

Tests use synthetic artifacts, fake engines and real temporary checkpoint files.
They cover low-bit draft rejection, parser/identity validation, iterator cleanup,
cancellation, accounting races and stale generations, partial writes, disk sync
and rename failures, corrupt payloads, unsafe paths and restart/lazy restoration.
Consult the current test output for counts. These tests do not qualify native
checkpoint coverage, power-loss recovery on every filesystem, numerical execution,
SM70 compatibility, Q8 fit or performance. No GPU backend build is implied.

The existing GPU target `conversation_snapshot_test` now also covers target-only
slot streaming, independent unbatched restoration, and unchanged source KV and
recurrent state. It is added for the runner once an isolated GPU is available;
it has not run locally. Qualify real multi-agent checkpoint/restore and resumed
continuation equivalence before enabling this on a serving lane.

## Next implementation gates

1. Qualify the opt-in runtime hooks using actual launch and request inputs; keep
   unsupported modes rejected and expose effective policies in status.
2. Qualify per-slot drain receipts on an isolated runner lane and calibrate peak
   reservations before extending beyond the single-lane text-only scope.
3. Harden shared expert-store publication/attach semantics with immutable identity.
4. Build every affected GPU backend and qualify batch export and destination-slot
   restoration and owned-slot turn completion; then qualify three-to-one migration
   and on-demand rotation with real agent traffic.
5. Qualify another architecture and, with the benchmark runner, useful deep-context
   concurrency and lane-count transitions. A generic metadata type is not support
   for another model's executor or state format.

Keep GPU validation with the runner and leave active benchmark lanes unchanged.

## Repository workflow

`upstream` tracks `https://github.com/Niko1221/Strata.git`; `origin` is the owner's
fork. Keep changes opt-in and small so upstream fixes can be incorporated.
Preserve upstream and dependency notices. Do not rewrite the running benchmark
checkout or install this branch into active lanes.

## Local review fixes (2026-10-20)

- `17c8d96`: bound routing-lock acquisition and preserve the owner's lock on
  timeout. The regression reproduced an expired request waiting indefinitely,
  then verified retryable busy with no gating or native commands.
- `cfc5269`: open checkpoint files nonblocking before validating their actual
  inode type. A FIFO previously hung before the regular-file check; it now fails
  immediately. Regular checkpoint read, checksum and fsync behavior is unchanged.

These are Python-only changes, separately versioned from the runner's current
qualification overlays. They do not establish GPU continuation correctness and
should not be applied to an in-progress qualification run. Precision requirements
remain Q4 minimum and Q8 or higher preferred.

Direct coordinator calls now reject unknown/malformed agent or destination names
and invalid timeouts with ValueError before entering a mutating operation. Such
caller errors do not gate healthy lanes or mark the coordinator blocked. Native
snapshot, restore, process-identity and integrity failures still fail closed.
The HTTP gateway already validates these inputs; its response contract is unchanged.

Native exports now require a completion acknowledgement whose token count matches
that session's frozen inventory, byte count matches the written payload, and
timing is finite and nonnegative. A mismatch blocks publication/source shutdown
and retains fixed residency. Checksums still validate bytes independently; neither
check establishes GPU continuation parity. Injected native-worker implementations
must return the same `tokens`, `bytes`, `ms` dictionary as SessionProtocol.export.

Checkpoint manifests reject duplicate JSON fields at every nesting level,
nonstandard NaN/Infinity constants, and non-integer version values before invoking
any restore reader. Previously the standard JSON parser silently selected the last
duplicate field, and Python equality admitted true/1.0 as version 1. Valid published
version-1 bundles are unchanged. This is malformed-metadata rejection within the
trusted local store contract, not a new authentication or hostile-filesystem claim.
