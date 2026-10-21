# Frozen collector contention review

Source-only review of `host-feed-window31-template.py` copied into window32,
SHA256 `2e0625101acc8b93dbf48ef96fc57c25089025813db82b203b7696e9db30ffee`.
No candidate, process or remote input is changed here. The runner owns repairs.

Four diagnostic tests reproduce concrete failures using the actual selected
collector function definitions with simulated commands and time:

1. All optional FAN1–FAN6 probes fail, required CPU succeeds, and `_refresh_safety`
   returns `fans.age_s=None`. `safety_snapshot` then performs `None += age`,
   producing a TypeError and HTTP 503 despite usable required CPU evidence.
2. Six optional probes execute before required CPU. At two seconds each plus
   three one-second service checks, the 14-second budget fails before CPU is
   attempted. These times are a bounded reproduction, not incident timings.
3. The freshness maximum includes optional fan ages. A stale optional fan makes
   the whole snapshot fail even with fresh CPU evidence and a normal heartbeat.
4. After the eight-second refresh interval, lock contention propagates a 503
   even if the previous required CPU measurement is still only nine seconds old.

The resulting failures affect the required CPU/SEL response because the HTTP
handler includes `safety_snapshot()` in the same response construction. This is
not proof that any one path caused window31's seven lock-busy responses. SEL and
required CPU probes can also fail with that message; per-probe incident timings
and return codes are needed for attribution.

## Minimal runner-owned correction

- Collect required CPU evidence before optional fan enrichment. Bound optional
  work by the remaining required-measurement freshness budget. Missing optional
  data must not destroy the normal path; degraded mode still requires all its
  independent evidence. Consider a separately validated bulk RPM/status sample
  later if six individual reads remain expensive; do not assume its schema.
- Make nullable optional ages safe to transport. Preserve every original
  per-sensor timestamp. Do not stamp incomplete fan data as fresh, fill missing
  readings from another fan, or erase an explicit known fault.
- Use a distinct lock-busy result/exception generated only by flock exit 75.
  On that refresh failure, reuse prior valid cached required evidence only if
  its ORIGINAL measurement age, plus cache/transport time, remains at most
  15 seconds. Never reset the cache's original time on fallback. Malformed,
  status, temperature and identity failures must not become lock-busy fallback.
- Check normal-path required freshness independently from optional fan ages.
  Preserve real optional ages for the adapter; old/incomplete fan evidence must
  still fail the degraded path. Retain all existing fault and incident checks.
- Keep the nonblocking BMC lock. It avoids waiting behind the production fan
  controller but does not establish operating-system scheduling priority: six
  separate successful acquisitions can still compete with production. No claim
  of measured starvation is made here.

The current observer already uses bounded exponential sleep between retries:
`min(.75 * 2**attempt, 4, remaining)` within 23 seconds/eight attempts. Do not
add duplicate backoff based on an assumption that these retries are immediate.
SEL info has an existing age-bounded cache; individual SEL entry failures are a
separate path and must not be hidden by inventing records or advancing cursors.

## Reproduction and fixed-code tests

Run `python -m pytest -q tools/qualification_window28/test_collector_contention_reproducer.py`.
Set `FROZEN_HOST_FEED` to another local copy of the exact frozen source if needed.
The four passing tests confirm the old defects; they do NOT prove a fix. The
loader rejects other hashes and executes only selected functions, never the
feed's server or fork startup. This diagnostic is not added to regular CI.

For the runner's replacement, invert these scenarios into passing behavior:
normal CPU/SEL survives missing optional fans; required CPU is attempted first;
normal-path stale optional data does not invalidate fresh required evidence;
explicit lock-busy can serve the same cached CPU at age 9 but never after 15;
repeated busy fallback cannot extend that deadline. Add negative tests for
real CPU faults, malformed data, changed identities and retained fresh fan
faults, then execute actual producer→adapter→observer/controller composition
with both normal and degraded branches. Preserve all safety thresholds.

## Follow-up: snapshot identity alias

The runner subsequently corrected these collector paths, then encountered a
separate response-schema mismatch: the actual feed emitted `pid`, while the
observer required `host_feed_pid`. A manually assembled test response supplied
the missing field and hid the defect. The startup failed before model/controller
execution. Frozen failure evidence remains runner-owned.

The reviewed correction emits both names from the same captured `feed_pid`:
feed SHA256 `7ac134c35f7273ebe2d2fb571c772d6c5cb7118bbd37866a765527f5dd3ace8d`.
The new test evaluates the actual production snapshot dictionary expression and
passes that result directly to `bound_fan_sample`; it checks the PID alias,
start ticks, port and matching fan-source identity. This thread independently
ran the four pipeline integration tests successfully without editing the
candidate or executing remote actions. This verifies that scoped correction,
not the entire live observer loop or GPU qualification.

Future seam tests must carry producer-generated responses through consumers
without `setdefault`, alias insertion or normalization that repairs the output.
Keep the alias equality assertion because the observer consumes `host_feed_pid`
but still writes its heartbeat identity from `pid`. Prefer also exercising the
actual handler assembly and serialization with I/O mocked: the dictionary-only
test does not itself execute the preceding `feed_pid = os.getpid()` assignment
or surrounding HTTP control flow. This is follow-up coverage guidance, not a
new blocker for the already corrected candidate.

## Follow-up: sampler overwritten between observer iterations

Window33 reached a second loop iteration and failed because the loop assigned
`gpu_sample = gpu`, replacing the `gpu_sample()` function with its result dict.
The isolated window34 fix removes that assignment and consistently uses `gpu`
for data. Source review confirms the corrected loop retains the callable; this
thread independently ran the five pipeline tests successfully. The new static
test rejects assignments to the function name, but it is not two-iteration
runtime evidence. Existing branch-test helpers still inject a dict named
`gpu_sample` and therefore cannot exercise the real sampler call boundary.

The runner owns the requested additional test: execute the exact fully rendered
observer payload for at least two complete loop iterations with external I/O
mocked. Assert two GPU query invocations, two heartbeat publications, unchanged
callability and no abort; then test a real second-sample failure is rejected.
Do not repair the global namespace or replace the whole loop with extracted
branch snippets in that test. No model/controller was started in the reported
window33 failure; no GPU qualification follows from these source tests.

## Follow-up: handoff persistence dispatch and interpreter identity

The runner found window34's handoff validator was selected but its main receipt
writer's window tuple still ended at 33. The corrected tuple includes 34. The
new regression executes the actual bootstrap `main()` through the first feed
launch boundary with remote I/O mocked, verifies the immediate-predecessor
receipt is written exactly once to window34, and checks it precedes feed launch.
It stubs the validator result, so the separate validator tests remain necessary.
This thread independently passed 16 targeted tests across the handoff, bootstrap
and rendered-observer execution modules. No candidate edits were made.

Separately, launching with system Python failed because `regex` was absent;
the runner reports the evaluated virtual environment completed that step.
Future preflight should bind the actual interpreter executable/environment used
by bootstrap, helpers and launch—not just source hashes—and run required import
checks with that interpreter before side effects. This is follow-up guidance,
not a request to delay the runner's corrected attempt or change its inputs.
