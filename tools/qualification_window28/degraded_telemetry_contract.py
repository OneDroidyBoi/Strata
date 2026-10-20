"""CPU reference contract for the proposed stale-heartbeat fallback.

Not a sampler/controller and not deployment approval. Adapters must bind source
identity and derive age from the original measurement, including transit time;
receipt time must never substitute for measurement time. Actual populated fan
inventory and available BMC lower-noncritical RPM boundary must come from verified hardware policy, not defaults.
This predicate replaces only a fan-heartbeat-age decision. Other safety gates
remain in force, including independent incidents and restoration deadlines.
"""
import math


def fallback_evidence(sample, policy):
    """Return (allowed, reason); stale heartbeat is intentionally not an input.

    Sample groups: fans {age_s,source,rpm,controller_state_ok,fault},
    gpus {age_s,source,temperatures: UUID->{core_c,hbm_c}},
    cpu {age_s,source,temperatures: sensor_id->C}.
    Sources are exact collector identities bound externally by the runner.
    controller_state_ok means the exact verified live controller and current
    normal/failsafe state. It does not claim PWM-duty readback; actual fan
    operation is established separately by RPM and sensor fault status.
    """
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    try:
        fans = policy['fan_lower_noncritical_rpm']
        gpu_ids = policy['gpu_ids']
        cpu_ids = policy['cpu_ids']
        sources = policy['sources']
        max_age = policy['max_age_s']
        max_temp = policy['max_temp_c']
        if (not isinstance(fans, dict) or not fans
                or any(not finite(v) or v < 0 for v in fans.values())
                or not isinstance(gpu_ids, tuple) or not gpu_ids or len(set(gpu_ids)) != len(gpu_ids)
                or not isinstance(cpu_ids, tuple) or not cpu_ids or len(set(cpu_ids)) != len(cpu_ids)
                or not finite(max_age) or not 0 < max_age <= 15
                or not finite(max_temp) or not 0 < max_temp <= 85):
            return False, 'invalid or unbound hardware policy'
        for group in ('fans', 'gpus', 'cpu'):
            row = sample[group]
            if (not isinstance(sources[group], str) or not sources[group]
                    or row['source'] != sources[group]):
                return False, group + ': unverified collector identity'
            if not finite(row['age_s']) or not 0 <= row['age_s'] <= max_age:
                return False, group + ': stale or invalid measurement age'
        fan = sample['fans']
        if fan['controller_state_ok'] is not True or fan['fault'] is not False:
            return False, 'fan control/fault evidence not healthy'
        rpm = fan['rpm']
        if (set(rpm) != set(fans) or set(fan['status']) != set(fans)
                or any(fan['status'][name] != 'ok' for name in fans)):
            return False, 'incomplete fan inventory'
        if any(not finite(rpm[name]) or rpm[name] <= minimum for name, minimum in fans.items()):
            return False, 'fan RPM at/below verified lower-noncritical boundary or invalid'
        gpu = sample['gpus']['temperatures']
        cpu = sample['cpu']['temperatures']
        if set(gpu) != set(gpu_ids) or set(cpu) != set(cpu_ids):
            return False, 'incomplete thermal inventory'
        temperatures = list(cpu.values())
        for row in gpu.values():
            temperatures.extend((row['core_c'], row['hbm_c']))
        if any(not finite(t) or not 0 <= t < max_temp for t in temperatures):
            return False, 'temperature unavailable, invalid, or at limit'
        return True, 'fresh verified fan state/RPM and complete independent thermal evidence'
    except (KeyError, TypeError, AttributeError, ValueError):
        return False, 'missing or malformed evidence/policy'
