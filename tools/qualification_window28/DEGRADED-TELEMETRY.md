# Stale fan heartbeat: reference contract

The user's October 20 policy permits continuing when the fan heartbeat alone is
stale but independent current evidence confirms fan operation and healthy
temperatures. Genuine fan failure, overtemperature and unavailable safety
evidence remain stop conditions. This reference is CPU-only; the runner owns
controller/feed implementation and deployment in a separately reviewed window.

## Evidence and actual hardware

The runner's sanitized `window30-live-fan-telemetry-evidence.md` identifies
`/usr/local/sbin/gpu-fan-control.sh`, SHA256
`8fd213e77e4802c68622b12b2ca1c9e908247d4f750281688a04d54572759025`.
Reported behavior: 5-second loop, 4-second SSH GPU query timeout, 15-second data
limit. The fail-safe branch commands all headers to 100% and deliberately leaves
the heartbeat unchanged. This explains why a stale heartbeat does not alone
prove a dead fan controller. It does not identify why GPU queries failed.

BMC reports populated FAN1–FAN6, all status `ok`, with sample RPMs
1700/2200/3500/3400/1700/1800. Each exposes Lower Non-Critical = 100 RPM;
Lower Critical and Non-Recoverable are unavailable. Use the available boundary
and health status as rotation evidence, not as a mechanically validated minimum
cooling speed. Do not invent missing critical limits or require unavailable PWM
readback. FAN2/FAN3 physical mapping is unverified, so require all six populated
fans rather than assuming which fan cools a particular GPU. FAN7 and secondary
unpopulated readings are excluded by the bound inventory, not silently ignored
if a previously populated sensor disappears.

Fresh direct VM samples provide four GPU core and HBM temperatures and 200 W
caps. BMC provides CPU temperature and additional board/card/NIC observations.
Retain the established 85°C GPU/HBM/CPU limits and all other existing gates;
do not infer new limits for ancillary sensors from one cool snapshot.

## Decision and provenance

`degraded_telemetry_contract.fallback_evidence` is an offline reference for only
the heartbeat fallback. It requires:

- Exact bound collector identities and original measurement ages, all finite
  and within 0–15 seconds at the decision. Include transport/queue time; reading
  an old sample now does not refresh it. Bound cross-host clock handling in the
  adapter. A new collector/process must be explicitly rebound.
- Fresh verified controller identity/state, normal or fail-safe, plus actual
  FAN1–FAN6 RPM strictly above their available 100 RPM noncritical boundaries,
  every sensor status `ok`, and no fan fault. A live service or command of 100%
  alone is insufficient. `controller_state_ok` is not a claim of PWM readback.
- Complete expected GPU core/HBM and CPU sensor inventories, finite values
  below 85°C. Missing, N/A, NaN, incomplete, future or stale measurements cannot
  establish healthy fallback. Current power-cap checks remain in the controller.

No fixed grace period based only on heartbeat age is introduced. Continue in a
logged degraded state while the independent proof stays fresh and healthy;
stop if that proof is lost. Preserve raw heartbeat age and record which evidence
authorized each decision. No synthetic heartbeat writes or timestamp refreshes.

Fan fault/overtemperature and existing incident, SEL, memory, power-cap and
restoration-deadline gates retain priority in both normal and degraded modes.
Do not return early on a fresh heartbeat while ignoring an explicit fan fault.
The runner must compose this reference with those existing gates; this small
predicate deliberately does not replace their controller implementation.

The historical GPU sample at 22:22:01 UTC and BMC sample at 22:22:21 UTC are
20 seconds apart. They document observed healthy values but cannot jointly
establish 15-second freshness at one 22:22:21 decision. New aggregate decisions
must re-age all original samples rather than treating collection success as
proof of simultaneous freshness.

## Local tests and integration

Seven truth-table tests cover independent healthy evidence, no timestamp
mutation, per-source stale/future/nonfinite ages and identity replacement,
command-only/service-only insufficiency, fan status/RPM failures, each GPU core
and HBM missing/nonfinite/over-limit reading, CPU overtemperature, missing sensor
inventory, total evidence loss, and the actual six-fan noncritical-boundary
shape. The older 1000/2000 RPM fixture values are explicitly synthetic, not
operational limits. The hardware-shaped test uses the reported 100 RPM boundary.

Run `python -m pytest -q tools/qualification_window28/test_degraded_telemetry_contract.py`.
These tests validate the decision contract, not actual sensors or degraded-mode
operation on the server. Runner integration still needs actual producer-to-
consumer schema tests and composition tests retaining all other gate priority.
No source lock, active input, fan service, controller or running lane is changed
by this reference.
