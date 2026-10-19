# Isolated GPU continuation and HTTP migration qualification

Test engine commit `b61e8fc0fbea1b2954393f17f92af8405f74b271` from
[OneDroidyBoi/Strata](https://github.com/OneDroidyBoi/Strata/tree/b61e8fc0fbea1b2954393f17f92af8405f74b271).
This is a qualification procedure, not a claim that GPU continuation works.
The existing HTTP and migration tests use fake engines. A real-model launcher
for this procedure must still be assembled by the runner from its isolated
worker setup and the owner APIs below; there is no turnkey migration CLI.

## Patched precision qualification (2026-10-19)

Keep the original pinned results unchanged. The runner reported a successful
CUDA 12.9 SM70 build, 4,265 synthetic GPU snapshot checks, 972 host fixture checks
and 347 CPU tests, followed by two strict preflight blockers before real-model
load. These are runner-reported results, not independently reproduced here.
Real-model continuation, HTTP routing and three-to-one remain pending.

Two separate implementation fixes are available on `engine/backend-precision-foundation`:

- `a994c75`: accept the packer's declared 64-byte alignment as well as 256-byte
  packs. Validate dense offsets and destination pool accounting using the header.
  No artifact repack is needed just to satisfy the guard. The existing 256-byte
  runner repack remains eligible for fresh source/payload validation.
- `6547b57`: reproduce explicit Q8_0-to-BF16 compatibility conversion, verify
  source bytes and packed bytes, recompute maximum error and check manifests.
  Q4 minimum and Q8 preference remain unchanged; converting low-bit source weights
  to float does not evade the source precision audit.

Label a run with either fix as PATCHED, recording its exact checkout and patch
hashes. Do not call it an unmodified-pin PASS. Before loading, rerun strict audit
against the actual artifact set; previous candidate verification is not enough.
The conversion check requires NumPy. Its temporary arrays are bounded by chunks,
not the entire tensor. The guard compares conversion method, source shard/hash,
source/destination type, exactness, reported error, output byte count and exact
Q8 conversion coverage in both manifests. Both exact and rounded Q8 conversions
are allowed when their metadata matches the recomputed result.

Use the packer's `--compat-bf16` output and an absolute source path, preserving
`conversions.json` and `compat-bf16.json`. This remains a narrow subset: compatibility
manifest entries for other rounded transforms are not yet admitted. Unknown
quantized transforms must fail rather than silently gain support. Bind both
manifests in the policy's optional `sha256` map for reproducible evidence; hashes
are integrity bindings, not signatures. Keep policy and artifacts immutable
through launch. Do not edit manifests to conceal a validation failure.

The owner harness must derive and bind a verified ModelIdentity after READY and
strict audit. Shared semantic identity must exclude per-worker device IDs, log
paths, admission registrations and other placement details; preserve those in
separate launch evidence. This does not relax model/tokenizer/config/state-layout
compatibility checks. Keep worker admission registrations distinct.

Local validation of the fixes: alignment suite 18 passed; final precision suite
23 passed with NumPy, including tampered manifests/payloads, wrong shard/type,
nonfinite/truncated sources, independently specified rounding values, chunk
boundaries and exact conversions. A broader targeted suite before the final
exact-conversion test passed 102 tests with four NumPy-dependent tests skipped in
the lightweight interpreter; those conversion tests were separately run with
NumPy. The CI environment now installs NumPy to run them. No GPU qualification
or live 195-tensor audit was performed in the implementation checkout.

### Split-GGUF lane identity correction

A further Python-only correction admits an audited target shard set whose
`components.target.architecture` matches the lane and whose explicit per-file
architecture values all agree. Secondary shards may omit architecture metadata;
at least one shard must explicitly name the matching architecture. The existing
complete-shard and precision audit must still pass. This fixes the previous
requirement that every shard repeat `general.architecture`.

Regression coverage constructs real four-shard GGUF audit reports, including
first-shard-only metadata, conflicting values, absent metadata and incomplete
shard sets. Lane/checkpoint/restore/migration/manifest and control/protocol tests:
165 passed locally. Record this Python overlay separately from the unchanged C++
binary; it does not establish GPU continuation or migration success.

### Managed HTTP teardown after a client timeout

Managed listeners now use non-daemon request threads. After stopping acceptance
with `httpd.shutdown()`, `httpd.server_close()` joins outstanding handlers before
returning. Only then may the owner close native workers. A client-side timeout
does not cancel consolidation or prove that its disk transfers and source exits
finished. Unmanaged listeners retain their existing daemon behavior.

This closes a confirmed lifecycle gap: previously listener closure could return
while consolidation still ran, allowing owner cleanup to close the destination's
pipes concurrently with resume/inventory. It is a plausible explanation for the
reported window20 closed-file failure, not a confirmed diagnosis without the
exception traceback and timeout/cleanup/native-operation timestamps. Preserve
that failure independently. Do not classify a source-stop-return event as success
without checking its result_state and confirmed process exit.

Joining handlers is deliberately not a bounded cancellation mechanism. Keep the
external safety supervisor authoritative; do not wait past its safety deadline.
If emergency teardown interrupts migration, report an aborted/blocked operation,
not graceful completion. Do not call server_close from one of its own handlers.
The regression holds a handler open and verifies that managed server_close cannot
return until it completes, without sockets or GPUs. Targeted tests: 73 passed.

### Post-exit stdin cleanup

`StrataEngine.close()` now tolerates ValueError as well as OSError when closing
stdin during final cleanup after confirmed process exit. The QUIT path has
already closed stdin. A wrapper that rejects duplicate close must not prevent
stdout/log cleanup, clearing the process handle and lane retirement. Tests cover
both normal QUIT and an already-exited process with already-closed stdin, plus
retention of a process that remains alive. This is separate from HTTP handler
joining; it does not make concurrent calls to engine.close safe.

The runner reports a 125.378-second operation against a 120-second client timeout.
Its proposed 300-second consolidation-only socket timeout leaves JSON timeout_s
and safety deadlines unchanged. This is a client wait allowance, not a guarantee
that migration completes in 300 seconds. Keep durable server work joined before
ordinary cleanup. The duplicate-close reproducer uses a stream wrapper; standard
Python streams normally allow repeated close, so retain runtime traceback/wrapper
identity before attributing the GPU incident conclusively to this path.

## Isolation and precision gates

1. Inventory GPU UUIDs, processes, reserved devices, free VRAM, host RAM and disk
   read-only. Identify an explicitly available test device. An apparently idle
   device is not proof that it is unreserved. If none is available, report BLOCKED.
2. Use a separate checkout, build directory, Python environment, ports, logs,
   writable packs/caches and checkpoint root. Record every process the test owns.
   Never stop, restart, signal, unload, reconfigure or fault-inject existing lanes.
   Do not install or replace system drivers/toolkits. Keep builds and allocations
   within available CPU/RAM headroom; stop only owned tests if budgets are exceeded.
3. Q4 is the minimum for every learned tensor, including auxiliaries. Prefer Q8
   or higher where the full workload fits. Enforce `STRATA_PRECISION_POLICY` on
   every spawn; save its actual source/pack audit and artifact hashes. Use
   `minimum_bits: 4, preferred_bits: 8` (or minimum 8 for the Q8 case). Disable
   learned MTP/draft and image sidecars. Never bypass audit failures or silently
   downgrade. A filename containing Q4/Q8 is not sufficient evidence.
4. Start with one isolated single-GPU worker, one batch group, text-only, and two
   slots. Use a currently supported model and state layout. Keep model, tokenizer,
   configuration, context limit and KV format identical across migration workers.
   Model-family extensibility is not yet qualification for arbitrary architectures.
5. Budget model weights, expert caches, two live contexts, restore scratch,
   checkpoint buffers, host offload and retained disk snapshots. Q8 KV is preferred;
   record KV precision separately from weight precision. Bound test iterations and
   disk use externally: checkpoint garbage collection is not implemented.

Three-to-one requires three independently registered native worker processes.
They may share an isolated GPU only if measured aggregate budgets permit; this
proves process consolidation, not release of two physical GPUs. Do not assume
three large model instances fit one V100S. If only one worker fits, complete the
single-worker stages and mark real three-to-one BLOCKED. Fake-worker tests do not
substitute for it. A physical GPU release test needs separately reserved devices.

## Build and initial tests

Read the pinned [V100 build notes](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/docs/NVIDIA_V100.md)
and [CMake configuration](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/CMakeLists.txt).
For V100S use an already available compatible CUDA 12.x toolkit, SM70 and the
experimental flag. In the isolated checkout, with a job count set to fit spare RAM:

```sh
cmake -S . -B build-migration-validation \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build build-migration-validation \
  --target strata conversation_snapshot_test conversation_validation_test \
  -j "$TEST_BUILD_JOBS"
CUDA_VISIBLE_DEVICES="$TEST_GPU_UUID" \
  ./build-migration-validation/conversation_snapshot_test
./build-migration-validation/conversation_validation_test
```

Set `TEST_GPU_UUID` to the reserved device UUID and `TEST_BUILD_JOBS` explicitly.
Do not run an unrestricted GPU test suite against all visible devices. Record
fixture exit codes: 77 is SKIPPED, not PASS. The snapshot fixture is synthetic;
it tests GPU state transfers and byte preservation, not real-model generation.
Run the CPU test commands in ENGINE_FORK.md in the isolated Python environment,
including both HTTP suites. Retain compiler output and test logs.

## Assemble the real-model HTTP harness

Use [ENGINE_FORK.md](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/docs/ENGINE_FORK.md)
for the owner API contracts and examples. Load only test-owned services with
strict audits, calibrated admission ledgers, complete verified ModelIdentity,
`STRATA_SESSION_EXPORT=1`, and explicit disabled draft. Capture READY/INFO and
require `session_export`, `session_export_batch`, `session_restore_batch`,
`session_release_batch` and `session_route_batch` capabilities.

Wrap each ready service in NativeLaneWorker. Give each worker a distinct admission
registration. Create a durable CheckpointStore and MigrationCoordinator with
explicit unique agent-to-slot enrollment. Attach MigrationGateway before starting
any managed HTTP handler, then serve on an unused loopback port. Use separate
client and operator credentials; redact them from logs. Keep the gateway Python
process alive when its original frontend native process is retired. Do not launch
ordinary unmanaged traffic into these workers.

For the single-worker stage, enroll two agents in its two slots. For three-to-one,
use three workers, two slots each, and six agents: two on each source and destination.
This forces both concurrent destination residency and disk parking/rotation.
Create the coordinator/gateway before the first managed HTTP requests. Use real
Service/StrataEngine instances, not the fake engines from unit-test helpers.

## Continuation acceptance tests

- Establish repeatable greedy baseline token IDs for each agent on an uninterrupted
  run with the same topology, batch scheduling and prompt sequence. Give agents
  distinct facts and histories; test interleaved turns and two simultaneous clients.
- Compare an uninterrupted branch with a checkpoint/export/restore branch from
  the same consumed prefix. Log token IDs, consumed prefix count, native slot ID,
  process generation, reuse/prefill evidence and first divergent token. Repeat the
  baseline to measure inherent nondeterminism. Exact parity is the deterministic
  acceptance gate; report divergence rather than hiding it in a text similarity score.
- Exercise ordinary replies, max_new_tokens=1, immediate EOS where reproducible,
  short contexts, the runner's existing deep-context baseline and twice that context
  when the configured model limit and budgets allow. Report unsupported lengths
  as BLOCKED. Preserve full normal client message history on every follow-up.
- Prove restored KV/recurrent state is used: a correct answer after full prompt
  recomputation does not pass warm continuation. Measure restored prefix reuse and
  new prefill tokens. The last emitted but unconsumed token may legitimately need
  processing; document the boundary rather than requiring zero prefill.
- Checkpoints preserve consumed computational prefixes, not arbitrary in-flight
  responses or sampler RNG state. Do not promise bit-exact stochastic replay or
  transparent resumption of an interrupted SSE reply.

## HTTP routing and three-to-one

1. Exercise `/v1/chat/completions`, `/v1/messages` and `/v1/responses`, both JSON
   and SSE. Send normal frontend authentication and exactly one
   `X-Strata-Agent-ID` per request. Compare the same agent's continuation with the
   baseline, including concurrent streams. A 200 response alone is insufficient.
2. Verify wrong authentication, missing/duplicate/malformed/unknown IDs cause no
   native state changes. Agent IDs are routing labels, not per-agent authorization.
   Verify managed load/unload/config/VRAM/slot-file controls reject changes.
3. Start consolidation while two owned turns are active. Use a short wait to test
   a safe busy response and retry once drained. Observe exact slot drain receipts,
   not only disconnected sockets or finished Python futures.
4. Invoke the operator endpoint on the isolated gateway:

   ```http
   POST /v1/migration/consolidate
   Authorization: Bearer <client key>
   X-Strata-Migration-Key: <operator key>
   Content-Type: application/json

   {"destination":"lane-c","timeout_s":30}
   ```

5. Require all three lanes, including lane-c, to have valid durable snapshots and
   the migration journal before source exit. Check payload hashes and published
   manifests. Record ordering evidence from export/publication/process-exit events;
   do not infer it just from final files. Graceful shutdown must park state on disk.
   Resource charges must retire only after confirmed owned source process exit.
6. Continue all six agents through the same HTTP address and IDs. Force at least
   three complete rounds of parked-agent rotation, with two resident agents active
   concurrently between rotations. Verify latest prefixes, isolated histories,
   token parity and reuse after every restore. Record whole-batch waits for rotation.
7. Retry the successful same-destination operation: expect 200 without regating.
   Try a different destination: expect 409 without state changes. Check invalid
   operator credentials and destination, busy 503/retry behavior and a client
   disconnect while waiting and during streaming. Admission must remain charged
   until native work actually drains.
8. In separate disposable test scenarios, test corrupt checkpoint copies,
   incompatible identities, occupied restore targets, source export failure and
   process-generation changes. Require no stale restore, premature source stop or
   resource release. Cancel partial prefill separately: current owned admission
   may deliberately fail closed and require owner recovery. This is not a passing
   transparent-migration case. Never corrupt the only valid checkpoint.

Preserve bundles and journals after graceful shutdown for inspection. Fresh-owner
restart recovery from disk is separate work: automatic journal replay/crash recovery
is not implemented. An fsync/manifest test is not a power-loss durability proof.

## Required result package

Return PASS / FAIL / BLOCKED separately for build, synthetic GPU transfer,
real-model continuation, HTTP routing, process three-to-one, physical GPU release,
concurrent rotation and each negative scenario. Include:

- Exact commit, any runner harness patch and launch commands, GPU UUID/model/VRAM,
  driver/toolkit/compiler, host RAM, storage/filesystem and isolation evidence.
- Audited artifact hashes, precision policy/report, model identity, weight and KV
  precision, native capability output, context sizes, slots and memory budgets.
- Raw test logs, baseline/restored token IDs and first divergence, prefix reuse and
  prefill counts, checkpoint hashes, journal references, source-exit event ordering.
- Per-agent and aggregate tok/s, TTFT, migration pause and restore latency p50/p95
  with sample counts, total consolidation time, peak VRAM/RAM/disk. Measure latency
  without heavy correctness instrumentation in a separate matching run.
- Numerical or performance regressions, unsupported configurations, test cases not
  run and the smallest reproducible failing harness. No speedup claim without a
  same-day matching baseline. Do not deploy the result to running lanes.

## Pinned implementation and test references

All links below identify the engine under test, not an evolving branch.

- [Gateway](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/migration_http.py), [HTTP handler integration](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/server.py), [HTTP integration test scenarios](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/test_migration_http_integration.py).
- [Coordinator](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/lane_migration.py), [migration tests](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/test_lane_migration.py), [native worker](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/native_lane.py).
- [Checkpoint store](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/checkpoint_store.py), [batch restore tests](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/test_batch_restore.py), [owned-slot tests](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/test_bound_slot.py).
- [Precision enforcement](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/precision_policy.py), [admission](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/serve/request_admission.py).
- [GPU snapshot fixture](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/src/core/conversation_snapshot_test.cpp), [native generation and slot protocol](https://github.com/OneDroidyBoi/Strata/blob/b61e8fc0fbea1b2954393f17f92af8405f74b271/src/program/generate.cpp).
