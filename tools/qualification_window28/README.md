# Qualification client and public inventory handoff

This directory holds the corrected runner client, tests, and source-only patches.
It does not deploy or launch an engine. Preserve failed windows and use a separate
candidate for every qualification attempt.

## Current correction: window30 exact consumed-prefix checkpoint

Window30 passed six GPU baselines, three-worker READY and the endpoint/native
matrix, then failed in `checkpoint_prefixes`. Its legacy check compared P to
held C and required P to prefix C. The P-first fixture deliberately saves
C=P[:-k], so the correct checkpoint was rejected. This was reproduced with the
real lifecycle and wire parser locally. The earlier integration test called
coordinator consolidation directly and omitted this runner checkpoint helper.

The client now supports an explicit `expected_consumed_prefixes` mapping. In
this mode it requires exact agent coverage, nonempty C strictly prefixing P,
and saved held IDs equal to that exact C. Count/payload integrity checks remain.
Without the mapping the old full-prompt-priming contract remains unchanged;
the two contracts are not combined into a permissive either-direction check.

The corrected driver derives C from the isolated baseline fixtures and passes
it to BOTH checkpoint calls, including the busy-request helper. Install the
client and driver together in a new candidate. Apply `exact-c-client.patch`
and `exact-c-driver.patch` against preserved window30 sources, or copy the two
full Python files here. The client patch base is 8e9b95a6. Rebind source locks,
plans and affected review receipts; preserve failed-window evidence. No backend,
precision, native binary, or safety threshold changes are included.

Current SHA256:

- Client: `c12cff9c5d217f7c7aaee0cd9339e4f9f595d3c64c903d271430c3496f6cee18`
- Driver: `a3202ac00b41a3b1c6a308af7f1e62da63f377eaefca5bf3a596b3ec0822e6e6`
- Lifecycle integration test: `1ba986bc22ee9d9f75be360582888b3bfd91bb02708353a18bad667c316c8700`
- Backend remains `d96e241fcc7b2245cfdc5f77f57dcf3671bc4320af79dba4cb06ffb0ccc96945`.

153 local tests pass. The expanded integration executes the actual runner busy
helper with real coordinator/admission/drain receipts: checkpoint twice, two
concurrent disposable requests, busy response with unchanged ownership, drain,
restore saved prefixes, consolidate, bind saved payloads and rotate all six
agents twice with advanced histories. Only native execution and HTTP transport
are simulated. Negative cases reject shorter C, longer C, changed same-length
IDs, full P and advanced generated state. Invalid references fail before gating.
The previous GPU failure is reproduced before the correction. These are CPU
contract results; migrated GPU parity/reuse/rotation remains unqualified.

## Prior correction: window29 inventory ownership (aab3182)


The qualification client called `NativeLaneWorker._snapshot()` outside the
worker's ownership transaction. Real lifecycle guards rejected the call before
baseline generation. A permissive fake worker had hidden this integration error.

`NativeLaneWorker.quiesce_inventory(timeout_s=30)` now provides a public
operation. It gates admission, waits for exact request drains, holds FIFO and
native control ownership, and returns a frozen `NativeLaneInventory`:

- `state`: `inventoried` or `blocked`; `error` explains failure.
- `generation`: native inventory generation string, distinct from engine.gen.
- `sessions`: tuples of native session ID, kind and token count.
- `held`: tuples of slot number and exact held token-ID tuple.

Success leaves admission gated and establishes only an unchanged-state resume
proof. Require `worker.resume(timeout_s).state == 'resumed'` before generating.
Inspection does not grant the durable proof required to release slots or stop a
lane. Failure stays gated. A request against an already gated lane is rejected
without consuming its existing resume proof. Existing private guards remain.

The client uses this operation before and after seeding. Native inventory
versions may change during generation; the separate engine/Popen identity pin
must remain stable. At aab3182, the bundled client was byte-identical to the runner's
`window29-owner-inventory-api-candidate` finalized client.

## Inventory-correction inputs at aab3182 (historical hashes)

SHA256:

- `serve/native_lane.py`:
  `d96e241fcc7b2245cfdc5f77f57dcf3671bc4320af79dba4cb06ffb0ccc96945`
- `qualification_client.py`:
  `8e9b95a626147562702066482c7c04c0f5b3a3604f345ab506a49739c7a56573`
- `test_owner_prefix_fixture.py`:
  `0c500a58f82281a41b3a094f1149dcaf25c7d1bbe0dd0549532ead00b222574f`
- `serve/test_qualification_lifecycle.py`:
  `af1a70fb7225aeb25685c50c9e4a40146964704711ce31a4fc892778ee357f51`

`inventory-api.patch` applies to native_lane.py SHA256
`511a5995684cb788e6358efba392c0bd2f4fa82ed6f955dd938c9810b3288e34`.
It adds only the public inventory result and operation. No C++ changes.

## Runner integration

1. Preserve the failed candidate, source locks and results.
2. In a new candidate, install the exact backend file or apply
   `inventory-api.patch`; use the exact bundled client and fixture tests.
3. Run the real lifecycle integration from the Strata checkout:

   ```sh
   QUALIFICATION_CLIENT_PATH=/absolute/candidate/qualification_client.py \
     python -m pytest -q serve/test_qualification_lifecycle.py
   ```

4. Run the runner's wrapper/full offline tests. Update every affected backend
   overlay/client hash and runtime source lock, plan, CPU preflight binding and
   launch-review receipt. Do not label an old preflight receipt as testing the
   changed native_lane.py; use the established explicit binding/revalidation
   procedure.
5. Only the runner executes the next GPU qualification under existing safety
   gates, Q4 minimum, Q8+ preferred and learned-MTP-disabled policy.

The new integration tests use real NativeLaneWorker, MigrationCoordinator,
RequestAdmission, CheckpointStore, DrainReceipt and native wire parser. Only the
native model execution/transport is simulated. They reproduce the old ownership
failure, then cover seed, baseline, durable checkpoint, shutdown, six-agent
three-to-one consolidation, two rounds of rotation with advanced histories,
failed resumes, pending drain, protocol failure and process changes. Inventory
alone cannot release an undurable slot. The suite is included in CI.

Final local run: 151 tests passed across lifecycle, migration, native adapter,
batch checkpoint/restore, session protocol and qualification fixtures. This is
CPU integration evidence, not GPU cache reuse or numerical parity qualification.

## Prior correction: window28 engine generation

Strata b61e8fc sets `self.gen = self.__dict__.get("gen", 0) + 1` at READY.
The original baseline required a string. The corrected client requires a
positive integer and pins the exact engine/Popen object, PID and generation
through seed and continuation; missing generation remains invalid. It also
closes the harness on initial validation failure. PID reuse via a replacement
Popen is rejected. Cross-process adoption still requires independent start-time
binding. `generation-identity.patch` is historical: it applies only to the old
b3d4955a client and does not include the newer public-inventory correction.

## Reusable orchestration contracts: window30 follow-up

Runner-reported on 2026-10-20: window30 validated the aab3182 integration but
encountered two repeated orchestration mismatches before GPU qualification.
The controller expected a soak-r1 receipt while its helper emitted r2. Observer
bootstrap again omitted feed start ticks and complete ready publication. The
runner reports repairing the receipt check and adopting the existing observer
and feed using verified process identities, without restarting them. These are
orchestration findings, not new engine defects or GPU qualification evidence.
The corrected controller hash reported by the runner is
`ac504b872cec57d30d1ea2e43458c94fb4ba7bcc72a033857ebd4901d33cc534`.
Its reported ten tests include the exact helper receipt schema; this thread did
not independently rerun those window30 tests.

Future reusable runner tests should exercise the actual producer and consumer
together, rather than separately maintained idealized receipt dictionaries:

- Feed the exact soak helper's emitted receipt through the full launch
  predicate, including its schema version and feed port. Exercise a valid
  complete receipt and reject wrong versions, missing fields and stale data.
- Feed the actual bootstrap result through the prepare helper. Require observer
  and feed PIDs, start ticks, port and root identity before publishing ready
  state or invoking soak. Missing start ticks must publish no ready receipt and
  start no soak. This is separate from matching a PID alone.
- Exercise verified adoption after bootstrap succeeded but preparation failed.
  A matching PID with different start ticks must be rejected; valid adoption
  must not spawn duplicate observers or feeds. Keep process control mocked in
  CPU tests and preserve failed-window evidence.
- Execute the complete launch predicate with real temporary files and their
  hashes, canonical paths, and producer-generated receipts. Mock only external
  process/time/transport boundaries. Bind the test result to the exact candidate
  sources so copying an older window cannot silently restore these mismatches.

These are follow-up test requirements, not implemented orchestration tests in
this repository. They do not change or gate window30's already active soak.

### Window31 bootstrap/template seam

Runner-reported on October 20: the staged bootstrap failed before spawning its
feed or observer because its STOP-hook string replacement expected an old
direct-GET line. The updated observer template inserted a monotonic request
start timestamp, so the exact replacement anchor no longer matched. This is a
template composition failure, not an engine or GPU result. The runner owns the
narrow anchor correction and independent review; no candidate is edited here.

Before staging a future bundle, render every actual remote payload using the
exact staged bootstrap and template bytes, including all sequential injections,
then compile each complete generated Python payload without executing it. Test
that every required insertion occurs exactly once, that the STOP hook remains
in the final loop, and that a missing or ambiguous anchor prevents any remote
spawn. Compiling templates or testing replacement snippets separately does not
exercise this seam. Bind the rendered-payload test to the final source hashes.
