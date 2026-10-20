# Qualification client and public inventory handoff

This directory holds the corrected runner client, tests, and source-only patches.
It does not deploy or launch an engine. Preserve failed windows and use a separate
candidate for every qualification attempt.

## Current correction: window29 inventory ownership

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
must remain stable. The bundled client is byte-identical to the runner's
`window29-owner-inventory-api-candidate` finalized client.

## Exact inputs

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
