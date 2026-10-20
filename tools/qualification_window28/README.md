# Window28 baseline process identity correction

This is a separate runner candidate, not a server deployment. The client was
copied from the preserved window28 candidate (SHA256
`b3d4955a1aa64bbb71d10c4f9ecca919907de44881ae19dff8f39a3a11ee89fc`).
Only `fresh_isolated_native_baseline` changes in the client.

Pinned Strata `b61e8fc` sets `self.gen = self.__dict__.get("gen", 0) + 1`
in `serve/server.py` at READY setup. The harness required a string and its fake
used a string, rejecting the real positive integer counter. Missing generation
is still invalid; do not synthesize one after a failed check.

The correction requires a positive integer counter and a live Popen object.
It pins that exact object, PID, engine object and generation before and after
seed/continuation. A new Popen with the same PID is rejected. This is in-process
identity checking, not a persisted PID identity certificate; cross-process
adoption still requires the runner's independently verified start-time binding.
Harness cleanup now includes initial validation failures.

## Runner integration

1. Preserve the failed window and its source lock/results.
2. Create a separate candidate from the frozen inputs. Apply
   `generation-identity.patch`, or copy the two Python files in this directory.
   Do not modify the active or failed candidate.
3. Run `python -m pytest -q test_owner_prefix_fixture.py` there, followed by
   the runner's wrapper and full integration regressions. Any other fake of
   Strata's `engine.gen` must use a positive integer. Native session inventory
   generation strings are a different field and must not be changed.
4. Recompute candidate client/test hashes and the runtime source lock. Bind the
   new lock in the plan and regenerate all affected verification/review receipts;
   retain the established CPU preflight/runtime binding procedure.
5. Use the existing safety gates and sole-runner GPU authority for the new
   qualification attempt. Keep Q4 minimum, Q8+ preferred and the qualified
   learned-MTP-disabled policy.

Candidate SHA256:

- `qualification_client.py`:
  `28dd6c0a7eb20fb688198027005c652a8f8eee3c3a0c2e1a769a0a16aa6d3779`
- `test_owner_prefix_fixture.py`:
  `337856b15d564897f96cf61de95a8946ed8c10c4a270ab838580f4d34c9d0b3f`

Local evidence: changing only the fake generation to integer reproduced three
failures in the original seven tests. Corrected candidate: nine fixture tests
pass, including absent generation, invalid values, generation changes, process
exit and same-PID Popen replacement during seed and continuation. Together with
owner migration/native-lane tests, 70 tests pass. No GPU execution performed.
