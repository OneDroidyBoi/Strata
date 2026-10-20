#!/usr/bin/env python3
"""Bounded real-model three-worker to one-worker qualification.

This runner is for the sole authorized GPU executor. Importing it is inert;
the CLI requires three preflight-bound lane manifests/configs and separate
result/checkpoint roots. It never discovers or attaches to outside processes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
from typing import Any

LANES = ("lane-a", "lane-b", "lane-c")
AGENTS = {
    "agent-alpha": ("lane-a", 0), "agent-beta": ("lane-a", 1),
    "agent-gamma": ("lane-b", 0), "agent-delta": ("lane-b", 1),
    "agent-epsilon": ("lane-c", 0), "agent-zeta": ("lane-c", 1),
}
GIB = 1 << 30
HOST_HARD_FLOOR = 8 * GIB
HOST_PRACTICAL_RESERVE = 16 * GIB
GPU_RESERVE = 2 * GIB
PINNED_COMMIT = "b61e8fc0fbea1b2954393f17f92af8405f74b271"
PINNED_VALIDATOR_SHA256 = "88ef07f6612ed0a0e455d9ebf0ac20e730f8c710ecd67d4308424aa297c52998"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def install_candidate_validator(engine_root: Path, validator_path: Path) -> str:
    """Install the reviewed precision guard before importing runner/service code."""
    engine_root, validator_path = engine_root.resolve(), validator_path.resolve()
    if not validator_path.is_file() or sha256(validator_path) != PINNED_VALIDATOR_SHA256:
        raise RuntimeError("candidate precision validator is missing or differs from reviewed hash")
    sys.path.insert(0, str(engine_root))
    import serve
    spec = importlib.util.spec_from_file_location("serve.precision_policy", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed precision validator")
    overlay = importlib.util.module_from_spec(spec)
    sys.modules["serve.precision_policy"] = overlay
    setattr(serve, "precision_policy", overlay)
    spec.loader.exec_module(overlay)
    if (Path(overlay.__file__).resolve() != validator_path
            or not callable(getattr(overlay, "enforce_launch_policy", None))):
        raise RuntimeError("candidate precision validator overlay did not install exactly")
    return PINNED_VALIDATOR_SHA256


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def host_gate(*, available_bytes: int, forecast_next_bytes: int,
              hard_floor: int = HOST_HARD_FLOOR,
              practical_reserve: int = HOST_PRACTICAL_RESERVE) -> dict:
    """Require forecasted load plus the practical reserve; never cross hard floor."""
    values = (available_bytes, forecast_next_bytes, hard_floor, practical_reserve)
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("host gate inputs must be nonnegative integer bytes")
    projected = available_bytes - forecast_next_bytes
    passed = projected >= practical_reserve and projected >= hard_floor
    return {"pass": passed, "available_bytes": available_bytes,
            "forecast_next_bytes": forecast_next_bytes,
            "projected_available_bytes": projected,
            "hard_floor_bytes": hard_floor,
            "practical_reserve_bytes": practical_reserve}


def host_admission_partitions(*, total_bytes: int, available_bytes: int,
                              runner_pss_bytes: int, engine_pss_bytes: list[int],
                              checkpoint_scratch_host_bytes: int = 0,
                              reserve_bytes: int = HOST_HARD_FLOOR) -> list[dict]:
    """Divide only currently free host headroom among the three admission ledgers.

    MemAvailable already reflects the resident worker footprints. Adding PSS to
    MemAvailable and making each ledger repay an equal share of its measured PSS
    double-counted large model mappings and rejected valid uneven PSS samples.
    Keep PSS as diagnostic evidence, but bound new request growth by free memory
    after the hard reserve and checkpoint scratch charge.
    """
    if len(engine_pss_bytes) != 3:
        raise ValueError("three per-engine PSS readings are required")
    vals = [total_bytes, available_bytes, runner_pss_bytes, checkpoint_scratch_host_bytes,
            reserve_bytes, *engine_pss_bytes]
    if any(type(x) is not int or x < 0 for x in vals):
        raise ValueError("host partition inputs must be nonnegative integer bytes")
    if total_bytes <= reserve_bytes or available_bytes <= reserve_bytes + checkpoint_scratch_host_bytes:
        raise RuntimeError("host hard reserve leaves no admission capacity")
    free_headroom = min(total_bytes - reserve_bytes,
                        available_bytes - reserve_bytes - checkpoint_scratch_host_bytes)
    cap, remainder = divmod(free_headroom, 3)
    if cap <= 0:
        raise RuntimeError("host hard reserve leaves no per-lane admission capacity")
    rows = []
    for i in range(3):
        rows.append({"host_capacity_bytes": cap + (1 if i < remainder else 0),
                     "fixed_host_bytes": 0,
                     "measured_host_baseline_pss_bytes": engine_pss_bytes[i]
                        + runner_pss_bytes // 3})
    if sum(x["host_capacity_bytes"] for x in rows) > free_headroom:
        raise AssertionError("host partitions exceed measured free headroom")
    return rows


def validate_staged_lane(base: Path, lane: str, spec: dict) -> tuple[Path, Path, Path, dict, dict]:
    """Bind staged config/manifest and CPU-only receipt without printing contents."""
    required = {"config", "manifest", "preflight", "config_sha256", "manifest_sha256", "preflight_sha256"}
    if not required.issubset(spec):
        raise ValueError(f"{lane}: staged input declaration is incomplete")
    paths = [((base / spec[k]).resolve()) for k in ("config", "manifest", "preflight")]
    if any(not p.is_relative_to(base.resolve()) for p in paths):
        raise ValueError(f"{lane}: staged inputs must remain under the isolated base directory")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError(f"{lane}: staged input file is missing")
    cfg_path, manifest_path, preflight_path = paths
    if sha256(cfg_path) != spec["config_sha256"] or sha256(manifest_path) != spec["manifest_sha256"]:
        raise RuntimeError(f"{lane}: staged config or manifest hash changed")
    if sha256(preflight_path) != spec["preflight_sha256"]:
        raise RuntimeError(f"{lane}: staged CPU preflight receipt hash changed")
    cfg, manifest, preflight = read_json(cfg_path), read_json(manifest_path), read_json(preflight_path)
    if (preflight.get("preflight_only") is not True or preflight.get("no_model_load") is not True
            or preflight.get("engine_commit") != PINNED_COMMIT
            or preflight.get("candidate_validator_sha256") != PINNED_VALIDATOR_SHA256
            or preflight.get("run_three_worker_sha256") != sha256(Path(__file__).resolve())
            or preflight.get("artifact_policy_passed") is not True
            or preflight.get("pack_payload_verified") is not True
            or preflight.get("config_sha256") != sha256(cfg_path)
            or preflight.get("manifest_sha256") != sha256(manifest_path)):
        raise RuntimeError(f"{lane}: CPU-only preflight receipt is not bound to staged inputs")
    if (manifest.get("architecture") != "qwen4exp" or manifest.get("batch") != 2
            or manifest.get("slot_groups") != 1 or manifest.get("draft") != "disabled"
            or manifest.get("context_tokens") != 8192 or str(manifest.get("kv_precision", "")).lower() != "int8"
            or not manifest.get("gpu_uuid") or type(manifest.get("gpu_ordinal")) is not int):
        raise RuntimeError(f"{lane}: manifest differs from the reviewed two-slot Q4 migration profile")
    validate_preflight_bindings(base, lane, cfg_path, manifest_path, cfg, manifest, preflight)
    return cfg_path, manifest_path, preflight_path, cfg, manifest


def validate_preflight_bindings(base: Path, lane: str, config_path: Path, manifest_path: Path,
                                cfg: dict, manifest: dict, preflight: dict) -> None:
    """Bind preflight evidence to every immutable runtime/artifact identity input."""
    candidate = base / "runner" / "precision_policy_candidate-owner-afb60e7.py"
    native_lane = base / "engine" / "serve" / "native_lane.py"
    required = ("config_path", "manifest_path", "config_sha256", "manifest_sha256",
                "model_artifact_sha256", "pack_artifact_sha256", "ple_artifact_sha256",
                "target_files", "pack_files", "native_lane_sha256", "policy_path",
                "policy_sha256", "binary_sha256", "expert_profile_path",
                "expert_profile_sha256", "tokenizer_path", "tokenizer_files_sha256",
                "template_path", "template_sha256", "architecture", "context_tokens",
                "kv_precision", "weight_precision", "gpu_ordinal", "gpu_uuid", "draft",
                "batch", "slot_groups")
    if any(key not in preflight for key in required):
        raise RuntimeError(f"{lane}: CPU preflight receipt lacks required source bindings")
    for key, path in (("config_path", config_path), ("manifest_path", manifest_path)):
        if preflight[key] != str(path.resolve()):
            raise RuntimeError(f"{lane}: preflight {key} points at different staged input")
    if preflight["config_sha256"] != sha256(config_path) or preflight["manifest_sha256"] != sha256(manifest_path):
        raise RuntimeError(f"{lane}: preflight config/manifest digest mismatch")
    for key in ("model_artifact_sha256", "pack_artifact_sha256", "ple_artifact_sha256",
                "architecture", "context_tokens", "kv_precision", "weight_precision",
                "gpu_ordinal", "gpu_uuid", "draft", "batch", "slot_groups"):
        if preflight[key] != manifest.get(key):
            raise RuntimeError(f"{lane}: preflight {key} differs from manifest")
    if preflight["target_files"] != manifest.get("model_files") or preflight["pack_files"] != manifest.get("pack_files"):
        raise RuntimeError(f"{lane}: preflight shard/pack inventory differs from manifest")
    if (not candidate.is_file() or sha256(candidate) != PINNED_VALIDATOR_SHA256
            or preflight.get("candidate_validator_sha256") != PINNED_VALIDATOR_SHA256):
        raise RuntimeError(f"{lane}: candidate precision validator hash binding failed")
    if (not native_lane.is_file() or preflight["native_lane_sha256"] != manifest.get("native_lane_sha256")
            or sha256(native_lane) != manifest.get("native_lane_sha256")):
        raise RuntimeError(f"{lane}: native migration source hash differs from preflight/manifest")
    policy_path = Path(preflight["policy_path"]).resolve()
    configured_policy = Path(cfg.get("env", {}).get("STRATA_PRECISION_POLICY", ""))
    if not configured_policy.is_absolute():
        configured_policy = Path(cfg.get("cwd") or base) / configured_policy
    configured_policy = configured_policy.resolve()
    if (not policy_path.is_file() or sha256(policy_path) != preflight["policy_sha256"]
            or configured_policy != policy_path):
        raise RuntimeError(f"{lane}: precision policy path/hash differs from config or receipt")
    exe = Path(cfg["exe"])
    if not exe.is_absolute():
        exe = Path(cfg.get("cwd") or base) / exe
    if not exe.is_file() or sha256(exe) != preflight["binary_sha256"]:
        raise RuntimeError(f"{lane}: runtime binary hash differs from CPU preflight")
    profile = Path(preflight["expert_profile_path"])
    if (profile != Path(manifest["expert_profile_path"]).resolve()
            or not profile.is_file() or sha256(profile) != manifest.get("expert_profile_sha256")
            or preflight["expert_profile_sha256"] != manifest.get("expert_profile_sha256")):
        raise RuntimeError(f"{lane}: expert-profile runtime binding differs from manifest")
    tokenizer = Path(preflight["tokenizer_path"])
    token_files = {name: sha256(tokenizer / name)
                   for name in ("vocab.json", "merges.txt", "token_type.json")}
    encoded_token_files = hashlib.sha256(json.dumps(token_files, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if encoded_token_files != preflight["tokenizer_files_sha256"]:
        raise RuntimeError(f"{lane}: tokenizer files differ from CPU preflight")
    template = Path(preflight["template_path"])
    if not template.is_file() or sha256(template) != preflight["template_sha256"]:
        raise RuntimeError(f"{lane}: chat template differs from CPU preflight")
    runtime_profile = {"architecture": cfg.get("arch"), "context_tokens": cfg.get("max_context"),
                       "gpu_ordinal": cfg.get("gpu"), "gpu_uuid": manifest.get("gpu_uuid"),
                       "kv_precision": manifest.get("kv_precision"),
                       "weight_precision": manifest.get("weight_precision"),
                       "draft": manifest.get("draft"), "batch": manifest.get("batch"),
                       "slot_groups": manifest.get("slot_groups")}
    if preflight.get("runtime_profile") != runtime_profile:
        raise RuntimeError(f"{lane}: CPU preflight runtime profile differs from staged config/manifest")


def resource_sample(gpu_ordinal: int, engine_pid: int) -> dict:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,temperature.gpu,power.limit",
         "--format=csv,noheader,nounits"], text=True)
    rows = [[x.strip() for x in line.split(",")] for line in raw.splitlines()]
    if len(rows) != 4 or not 0 <= gpu_ordinal < len(rows):
        raise RuntimeError("expected four GPUs and an explicit valid ordinal")
    row = rows[gpu_ordinal]
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
    def rss(pid: int) -> int:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        return int(next(x for x in lines if x.startswith("VmRSS:")).split()[1]) * 1024
    def pss(pid: int) -> int:
        lines = Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
        return int(next(x for x in lines if x.startswith("Pss:")).split()[1]) * 1024
    return {"gpu_index": gpu_ordinal, "gpu_uuid": row[1],
            "gpu_total_bytes": int(float(row[2]) * 1024 * 1024),
            "gpu_used_bytes": int(float(row[3]) * 1024 * 1024),
            "gpu_temperature_c": float(row[4]), "gpu_cap_w": float(row[5]),
            "host_mem_total_bytes": meminfo["MemTotal"],
            "host_mem_available_bytes": meminfo["MemAvailable"],
            "runner_rss_bytes": rss(os.getpid()), "engine_rss_bytes": rss(engine_pid),
            "runner_pss_bytes": pss(os.getpid()), "engine_pss_bytes": pss(engine_pid)}


def install_measured_admissions(services: dict, samples: dict, *,
                                total_host_bytes: int, available_host_bytes: int,
                                checkpoint_scratch_host_bytes: int) -> dict:
    """Replace load-time bootstrap controllers with measured post-READY ledgers."""
    from serve.request_admission import RequestAdmission
    rows = host_admission_partitions(
        total_bytes=total_host_bytes, available_bytes=available_host_bytes,
        runner_pss_bytes=max(samples[lane]["runner_pss_bytes"] for lane in LANES),
        engine_pss_bytes=[samples[lane]["engine_pss_bytes"] for lane in LANES],
        checkpoint_scratch_host_bytes=checkpoint_scratch_host_bytes)
    configs = {}
    for index, lane in enumerate(LANES):
        service = services[lane]
        template = service.request_admission.config
        sample = samples[lane]
        config = dict(template)
        partition = dict(rows[index])
        measured_baseline = partition.pop("measured_host_baseline_pss_bytes")
        config.update(partition)
        config.update({"gpu_id": str(sample["gpu_index"]),
                       "gpu_capacity_bytes": sample["gpu_total_bytes"] - GPU_RESERVE,
                       "fixed_gpu_bytes": sample["gpu_used_bytes"],
                       "max_requests": 2, "max_context_tokens": 8192})
        requests = config["max_requests"]
        tokens = config["max_context_tokens"]
        host_peak = requests * (config["host_request_bytes"]
                                + (tokens + config["lookahead_tokens"])
                                * config["host_bytes_per_token"])
        gpu_peak = requests * (config["gpu_request_bytes"]
                               + (tokens + config["lookahead_tokens"])
                               * config["gpu_bytes_per_token"])
        if (config["host_capacity_bytes"] < config["fixed_host_bytes"] + host_peak
                or config["gpu_capacity_bytes"] < config["fixed_gpu_bytes"] + gpu_peak):
            raise RuntimeError(f"{lane}: measured host/GPU admission budget cannot cover two full contexts")
        service.request_admission = RequestAdmission(config)
        configs[lane] = dict(config, measured_host_baseline_pss_bytes=measured_baseline)
    if len({id(s.request_admission) for s in services.values()}) != 3:
        raise RuntimeError("measured worker admissions are not distinct")
    return configs


def mem_available_bytes() -> int:
    line = next(x for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:"))
    return int(line.split()[1]) * 1024


def mem_total_bytes() -> int:
    line = next(x for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemTotal:"))
    return int(line.split()[1]) * 1024


def checked_endpoint_matrix(client, *, model: str, messages_by_agent: dict,
                            results, max_new: int = 4) -> list:
    """Six-agent/three-endpoint matrix that extends each transcript each call."""
    import qualification_client as qc
    if max_new not in (1, 4, 8, 16, 32) or set(messages_by_agent) != set(AGENTS):
        raise ValueError("matrix requires all six enrolled agents and a bounded output")
    histories = messages_by_agent
    rows = []
    for agent, messages in histories.items():
        for endpoint in qc.ENDPOINTS:
            for stream in (False, True):
                p = client.request(endpoint, model=model, messages=messages, agent_id=agent,
                                   stream=stream, max_new=max_new, temperature=0.0)
                p.name = f"endpoint:{agent}:{endpoint}:{'sse' if stream else 'json'}"
                results.write(p); rows.append(p)
                if p.status != "PASS":
                    continue
                messages.extend([{"role": "assistant", "content": p.detail["output_text"]},
                                 {"role": "user", "content": f"Continue {agent}'s own test fact; matrix step {len(rows)}."}])
    return rows


class ReceiptEvidence:
    """Test-owned exact drain receipt registry installed before managed HTTP."""
    def __init__(self, harness, results):
        self.harness, self.results = harness, results
        self._lock = threading.RLock()
        self._rows: dict[str, dict] = {}
        self._by_agent: dict[str, list[str]] = {}

    def install(self) -> None:
        for lane, worker in self.harness.workers.items():
            admission = worker.admission
            original_run = admission.run
            def make_run(lane_name, worker_obj, original):
                def run(factory, prompt_tokens, max_new, quiescent=None, *, drain_receipt=None):
                    gateway = worker_obj.service.migration_gateway
                    binding = getattr(gateway._local, "binding", None) if gateway else None
                    agent = getattr(binding, "agent", None)
                    request_id = secrets.token_hex(12)
                    route = worker_obj.service.migration_gateway.coordinator.routes.get(agent) if agent else None
                    row = {"request_id": request_id, "lane": lane_name,
                           "agent": agent, "slot": getattr(route, "slot", None),
                           "process_generation": getattr(binding, "generation", None),
                           "receipt_confirmed": False, "receipt_failed": False,
                           "admission_drained": False, "tickets": []}
                    if drain_receipt is not None:
                        begin = drain_receipt.begin
                        def tracked_begin(generation, process, slot=None):
                            ticket = begin(generation, process, slot)
                            row["tickets"].append({"ticket_id": ticket.ticket_id,
                                                    "generation": ticket.generation,
                                                    "slot": slot})
                            with self._lock:
                                self._rows[request_id] = row
                                if agent:
                                    self._by_agent.setdefault(agent, []).append(request_id)
                            return ticket
                        drain_receipt.begin = tracked_begin
                        def completed(receipt):
                            try:
                                row["receipt_confirmed"] = receipt.confirmed() is True
                                row["receipt_failed"] = receipt.failed() is True
                            except Exception:
                                row["receipt_failed"] = True
                        drain_receipt.add_done_callback(completed)
                    with self._lock:
                        self._rows[request_id] = row
                        if agent:
                            self._by_agent.setdefault(agent, []).append(request_id)
                    try:
                        yield from original(factory, prompt_tokens, max_new,
                                            quiescent, drain_receipt=drain_receipt)
                    finally:
                        row["admission_drained"] = worker_obj.admission.wait_drained(0)
                        row["finished_wall"] = time.time()
                        self.results.event("request-drain-evidence", **row)
                return run
            admission.run = make_run(lane, worker, original_run)

    def both_agents_active_with_receipts(self, agents: tuple[str, str]) -> bool:
        with self.harness.coordinator._cv:
            active = set(self.harness.coordinator._active)
        if not set(agents).issubset(active):
            return False
        with self._lock:
            for agent in agents:
                ids = self._by_agent.get(agent, [])
                if not ids or not any(self._rows[x].get("tickets") and not self._rows[x].get("receipt_confirmed")
                                      for x in ids):
                    return False
        return True

    def last_agent_drained(self, agent: str) -> dict:
        with self._lock:
            ids = self._by_agent.get(agent, [])
            if not ids:
                return {"receipt_confirmed": False, "admission_drained": False}
            row = self._rows[ids[-1]]
            drained_now = self.harness.workers[row["lane"]].admission.wait_drained(0)
            return {"receipt_confirmed": row.get("receipt_confirmed") is True,
                    "receipt_failed": row.get("receipt_failed") is True,
                    "admission_drained": row.get("admission_drained") is True or drained_now,
                    "request_id": row.get("request_id"), "ticket_count": len(row.get("tickets", []))}

    def last_request_id(self, agent: str) -> str | None:
        with self._lock:
            ids = self._by_agent.get(agent, [])
            return ids[-1] if ids else None


def _state_snapshot(harness) -> dict:
    lanes = {}
    for name, worker in harness.workers.items():
        if name in harness.coordinator._stopped:
            proc = getattr(worker.engine, "proc", None)
            lanes[name] = {"stopped": True, "process_exit": proc.poll() if proc is not None else None,
                           "admission": worker.admission.ledger.snapshot(),
                           "process_generation": getattr(worker.engine, "gen", None)}
            continue
        with worker._transaction():
            if not worker.wait_drained(10.0):
                raise RuntimeError("trusted test snapshot could not drain a worker")
            gen, _, detail = worker._snapshot()
            lanes[name] = {"generation": gen, "sessions": detail[1],
                           "admission": worker.admission.ledger.snapshot(),
                           "process_generation": getattr(worker.engine, "gen", None)}
    with harness.coordinator._cv:
        coord = {"routes": {k: vars(v) for k, v in harness.coordinator._routes.items()},
                 "active": sorted(harness.coordinator._active),
                 "blocked": harness.coordinator._blocked,
                 "destination": harness.coordinator._destination,
                 "stopped": sorted(harness.coordinator._stopped)}
    return {"lanes": lanes, "coordinator": coord}


def install_native_metrics(harness, results) -> dict:
    """Capture native DONE counters and phase-scoped restore evidence."""
    state = {"phase": {"name": "setup"}, "events": [], "pending_rotation_restore": set(),
             "rotation_restores": [], "lock": threading.RLock()}
    for lane, worker in harness.workers.items():
        engine, original = worker.engine, worker.engine.generate
        def make_generate(lane_name, engine_obj, method):
            def generate(*args, **kwargs):
                started = time.monotonic()
                try:
                    yield from method(*args, **kwargs)
                finally:
                    gateway = getattr(harness.frontend, "migration_gateway", None)
                    binding = getattr(gateway._local, "binding", None) if gateway else None
                    slot_local = getattr(engine_obj, "_bound_slot_local", None)
                    slot = getattr(slot_local, "slot", None) if slot_local else None
                    agent = getattr(binding, "agent", None)
                    if agent is None and slot is not None:
                        with harness.coordinator._cv:
                            agent = next((a for a, r in harness.coordinator._routes.items()
                                          if r.lane == lane_name and r.slot == slot), None)
                    last = dict(getattr(engine_obj, "last", {}) or {})
                    with state["lock"]:
                        phase = state["phase"]["name"]
                        event = {"phase": phase, "lane": lane_name, "agent": agent, "slot": slot,
                                 "process_generation": getattr(engine_obj, "gen", None),
                                 "generated": last.get("generated"), "prompt_tokens": last.get("prompt_tokens"),
                                 "prompt_read": last.get("prompt_read"), "reused": last.get("reused"),
                                 "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                                 "finish": last.get("finish"), "engine_generate_wall_s": time.monotonic() - started}
                        state["events"].append(event)
                        if phase == "rotation" and agent in state["pending_rotation_restore"]:
                            state["rotation_restores"].append(dict(event))
                            if type(event["reused"]) is int and event["reused"] > 0:
                                state["pending_rotation_restore"].remove(agent)
                        results.event("native-request-metrics", **event)
            return generate
        engine.generate = make_generate(lane, engine, original)
    return state


def validate_native_baseline(native: dict, agents=AGENTS) -> None:
    """Require deterministic repeats and prove prefix reuse on the warm pass.

    The first continuation is intentionally cold and should report reused=0.
    After restoring its checkpoint, the repeat is the operation that proves
    native prefix reuse for each enrolled agent.
    """
    if not all(native["repeat_comparison"][a]["exact_parity"] for a in agents):
        raise RuntimeError("uninterrupted same-prefix repeat diverged")
    if not all(type(native["repeat"][a].get("reused")) is int and native["repeat"][a]["reused"] > 0
               for a in agents):
        raise RuntimeError("uninterrupted baseline repeat did not prove native prefix reuse for all six agents")


def validate_migrated_native(migrated: dict, agents=AGENTS) -> None:
    """Require exact token parity and positive native prefix reuse per agent."""
    if not all(migrated["comparison"][a]["exact_parity"] for a in agents):
        raise RuntimeError("migrated native token IDs differ from uninterrupted baseline")
    if not all(type(migrated["trials"][a].get("reused")) is int and migrated["trials"][a]["reused"] > 0
               for a in agents):
        raise RuntimeError("migrated branch did not prove restored-prefix reuse for all six agents")


def validate_rotation_restore_reuse(metrics: dict) -> None:
    """Require native reuse evidence after each actual rotation restore."""
    pending = sorted(metrics["pending_rotation_restore"])
    restores = metrics["rotation_restores"]
    if pending:
        raise RuntimeError("rotation restore lacked a subsequent native reuse sample for: " + ", ".join(pending))
    if not restores or any(type(row.get("reused")) is not int or row["reused"] <= 0 for row in restores):
        raise RuntimeError("rotation restore did not prove positive native prefix reuse")


def summarize_rotation_metrics(metrics: dict, agents=AGENTS) -> dict:
    """Summarize only native samples tagged rotation; TTFT is not captured here."""
    rows = [row for row in metrics["events"] if row.get("phase") == "rotation"]
    by_agent = {}
    for agent in agents:
        samples = [row for row in rows if row.get("agent") == agent]
        completed = [row for row in samples if type(row.get("generated")) is int
                     and type(row.get("decode_ms")) in (int, float) and row["decode_ms"] > 0
                     and type(row.get("engine_generate_wall_s")) in (int, float)
                     and row["engine_generate_wall_s"] > 0]
        generated = sum(row["generated"] for row in completed)
        decode_ms = sum(row["decode_ms"] for row in completed)
        wall_s = sum(row["engine_generate_wall_s"] for row in completed)
        by_agent[agent] = {"native_sample_count": len(samples), "rate_sample_count": len(completed),
                           "generated_tokens": generated,
                           "decode_tok_s_native": generated * 1000.0 / decode_ms if decode_ms else None,
                           "engine_generate_wall_tok_s": generated / wall_s if wall_s else None,
                           "engine_generate_wall_s_total": wall_s}
    complete = [row for row in rows if type(row.get("generated")) is int
                and type(row.get("decode_ms")) in (int, float) and row["decode_ms"] > 0
                and type(row.get("engine_generate_wall_s")) in (int, float)
                and row["engine_generate_wall_s"] > 0]
    generated = sum(row["generated"] for row in complete)
    decode_ms = sum(row["decode_ms"] for row in complete)
    wall_s = sum(row["engine_generate_wall_s"] for row in complete)
    return {"source": "thread-local native DONE fields captured at engine.generate completion",
            "per_agent": by_agent,
            "aggregate": {"native_sample_count": len(rows), "rate_sample_count": len(complete),
                          "generated_tokens": generated,
                          "decode_tok_s_native": generated * 1000.0 / decode_ms if decode_ms else None,
                          "engine_generate_wall_tok_s": generated / wall_s if wall_s else None,
                          "engine_generate_wall_s_total": wall_s},
            "limitations": ["engine_generate_wall_tok_s includes engine request overhead but excludes HTTP/client overhead",
                            "TTFT is not captured by the native DONE hook",
                            "reported rates are observational correctness-run metrics, not a matching performance benchmark"]}


def require_busy_observation(status: int, unchanged: bool) -> None:
    """Fail closed when a fast active turn lets consolidation commit."""
    if status != 503 or unchanged is not True:
        raise RuntimeError("busy response/state invariance failed; owner recovery required before continuing")


def install_rotation_metrics(harness, results, native_metrics: dict) -> None:
    coordinator = harness.coordinator
    original = coordinator._activate
    def activate(agent, timeout_s):
        start = time.monotonic()
        route_before = coordinator.routes.get(agent)
        was_parked = getattr(route_before, "lane", None) is None
        try:
            value = original(agent, timeout_s)
            if native_metrics["phase"]["name"] == "rotation" and was_parked:
                with native_metrics["lock"]:
                    native_metrics["pending_rotation_restore"].add(agent)
                results.event("rotation-activation", phase="rotation", agent=agent,
                              from_lane=getattr(route_before, "lane", None),
                              to_lane=getattr(coordinator.routes.get(agent), "lane", None),
                              elapsed_s=time.monotonic() - start, result="ready")
            return value
        except BaseException as exc:
            if native_metrics["phase"]["name"] == "rotation":
                results.event("rotation-activation", phase="rotation", agent=agent,
                              from_lane=getattr(route_before, "lane", None),
                              elapsed_s=time.monotonic() - start,
                              result="error", exception=type(exc).__name__)
            raise
    coordinator._activate = activate


def busy_503_restore_then_consolidate(client, harness, *, model: str,
                                      messages_by_agent: dict, agents: tuple[str, str],
                                      operator_token: str, receipt_evidence: ReceiptEvidence,
                                      expected_prefixes: dict, expected_prompts: dict,
                                      expected_consumed_prefixes: dict,
                                      results) -> tuple[list, dict]:
    """Exercise a genuine busy response, drain exact receipts, restore branch, retry once."""
    import qualification_client as qc
    prefixes = qc.checkpoint_prefixes(harness, expected_prompts=expected_prompts,
                                      expected_consumed_prefixes=expected_consumed_prefixes)
    qc.assert_saved_slot_state_equal(expected_prefixes, prefixes)
    before_routes = {a: (r.lane, r.slot) for a, r in harness.coordinator.routes.items()}
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.request, "/v1/chat/completions", model=model,
                               messages=messages_by_agent[a], agent_id=a,
                               stream=True, max_new=32, temperature=0.0) for a in agents]
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not receipt_evidence.both_agents_active_with_receipts(agents):
            time.sleep(0.05)
        active = receipt_evidence.both_agents_active_with_receipts(agents)
        if not active:
            raise RuntimeError("busy test lacked two simultaneously active owned requests with pending receipts")
        before_ledgers = {lane: worker.admission.ledger.snapshot()
                          for lane, worker in harness.workers.items()}
        started = time.time()
        status, headers, body = client._call(
            "/v1/migration/consolidate", {"destination": "lane-c", "timeout_s": 0.01},
            extra_headers=[("Authorization", "Bearer " + client.client_api_key),
                           ("X-Strata-Migration-Key", operator_token)])
        unchanged = ({a: (r.lane, r.slot) for a, r in harness.coordinator.routes.items()} == before_routes
                     and {lane: worker.admission.ledger.snapshot()
                          for lane, worker in harness.workers.items()} == before_ledgers
                     and harness.coordinator._blocked is False
                     and harness.coordinator._destination is None)
        busy = qc.Probe("migration:busy-active-turns", "PASS" if status == 503 and unchanged else "FAIL",
                        started, time.time(), status,
                        {"expected_status": 503, "retry_after": headers.get("retry-after"),
                         "routes_unchanged": unchanged, "body": body.decode("utf-8", "replace")[:300]})
        rows.append(busy); results.write(busy)
        request_rows = [f.result() for f in futures]
        for agent, probe in zip(agents, request_rows):
            probe.name = "migration:busy-drain-request:" + agent
            rows.append(probe); results.write(probe)
        if any(p.status != "PASS" for p in request_rows):
            raise RuntimeError("active busy-test request failed")
    try:
        require_busy_observation(status, unchanged)
    except RuntimeError:
        # A fast completion can race the short wait and allow consolidation to
        # succeed. The active requests have now drained, but the topology may
        # already be irreversibly consolidated: retain the response and stop.
        atomic_json(Path(results.path).parent / "busy-race-failure.json",
                    {"http_status": status, "unchanged": unchanged,
                     "routes": {a: (r.lane, r.slot) for a, r in harness.coordinator.routes.items()},
                     "blocked": harness.coordinator._blocked,
                     "destination": harness.coordinator._destination})
        raise
    for agent in agents:
        drain = receipt_evidence.last_agent_drained(agent)
        if not (drain.get("receipt_confirmed") and not drain.get("receipt_failed")
                and drain.get("ticket_count", 0) >= 1):
            raise RuntimeError(f"{agent}: busy request exact native drain receipt did not confirm")
    # The busy requests form a disposable branch; restore their exact full-batch
    # prefixes before the one post-drain consolidation retry.
    qc.restore_prefixes(harness, prefixes)
    qc.assert_current_slots_match_saved(harness, prefixes)
    retry = client.consolidate(destination="lane-c", operator_token=operator_token, timeout_s=30.0)
    rows.append(retry)
    if busy.status != "PASS" or retry.status != "PASS":
        raise RuntimeError("busy response or single drained retry failed")
    return rows, prefixes


def _seed_and_prompts(client, harness, model: str, services: dict, results) -> tuple[dict, dict]:
    facts = {
        "agent-alpha": "the orchard marker is Cedar",
        "agent-beta": "the vessel marker is Bluebird",
        "agent-gamma": "the station marker is Quartz",
        "agent-delta": "the archive marker is Juniper",
        "agent-epsilon": "the bridge marker is Copper",
        "agent-zeta": "the telescope marker is Marigold",
    }
    histories = {}
    for agent, fact in facts.items():
        messages = [{"role": "user", "content": "Remember this unique qualification fact: " + fact + "."}]
        p = client.request("/v1/chat/completions", model=model, messages=messages,
                           agent_id=agent, max_new=8, temperature=0.0)
        if p.status != "PASS":
            raise RuntimeError(f"seed request failed for {agent}")
        histories[agent] = messages + [{"role": "assistant", "content": p.detail["output_text"]}]
    # All services share an identical tokenizer/template; use the frontend's exact path.
    atomic_json(results.parent / "seed-histories.json", histories)
    return histories, None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def run_three_worker(*, base: Path, plan_path: Path, result_dir: Path,
                     checkpoint_dir: Path, prefix_baseline_path: Path | None = None) -> dict:
    base = base.resolve(); plan_path = plan_path.resolve()
    result_dir = result_dir.resolve(); checkpoint_dir = checkpoint_dir.resolve()
    if result_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("result and checkpoint roots must be fresh, isolated paths")
    plan = read_json(plan_path)
    policy_path = Path(plan["precision_policy_path"])
    if not policy_path.is_absolute():
        policy_path = base / policy_path
    policy_path = policy_path.resolve()
    if plan.get("engine_commit") != PINNED_COMMIT or set(plan.get("lanes", {})) != set(LANES):
        raise RuntimeError("plan must pin the reviewed engine commit and exactly three lanes")
    checkpoint_scratch = plan.get("checkpoint_scratch_bytes")
    if type(checkpoint_scratch) is not int or checkpoint_scratch <= 0:
        raise ValueError("plan must include a positive reviewed checkpoint_scratch_bytes estimate")
    if not policy_path.is_file():
        raise FileNotFoundError("planned precision policy is missing")
    lanes = {name: validate_staged_lane(base, name, plan["lanes"][name]) for name in LANES}
    for lane, item in lanes.items():
        receipt = read_json(item[2])
        if Path(receipt["policy_path"]).resolve() != policy_path.resolve():
            raise RuntimeError(f"{lane}: CPU preflight does not bind planned precision policy")
    manifests = {name: item[4] for name, item in lanes.items()}
    if len({m["gpu_uuid"] for m in manifests.values()}) != 3:
        raise RuntimeError("three unique GPU UUIDs are required")
    if len({m["gpu_ordinal"] for m in manifests.values()}) != 3:
        raise RuntimeError("three unique GPU ordinals are required")
    # Configs contain admission values but they are never copied to output/logs.
    for name, item in lanes.items():
        cfg, manifest = item[3], item[4]
        admission = cfg.get("resource_admission")
        if (not isinstance(admission, dict) or str(admission.get("gpu_id")) != str(manifest.get("gpu_ordinal"))
                or admission.get("gpu_capacity_bytes", 0) <= 0
                or admission.get("host_capacity_bytes", 0) <= 0
                or admission.get("max_requests") != 2
                or admission.get("max_context_tokens") != 8192):
            raise RuntimeError(f"{name}: config lacks bounded bootstrap admission inputs")
    expected_cap = plan.get("gpu_power_cap_w", 200)
    if type(expected_cap) not in (int, float) or expected_cap <= 0:
        raise ValueError("plan must bind a positive expected GPU power cap")
    result_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if shutil_disk_free(checkpoint_dir) < checkpoint_scratch:
        raise RuntimeError("checkpoint disk scratch gate failed before model load")
    validator_path = base / "runner" / "precision_policy_candidate-owner-afb60e7.py"
    validator_sha = install_candidate_validator(base / "engine", validator_path)
    sys.path.insert(0, str(base)); sys.path.insert(0, str(base / "runner"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import runner_factory
    import real_model_harness
    import qualification_client as qc
    services, ready_rows = {}, {}
    harness = None; httpd = None; receipt_evidence = None
    evidence = {"stage": "three_to_one", "engine_commit": PINNED_COMMIT,
                "candidate_validator_sha256": validator_sha,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "lanes": {}, "stages": {}}
    atomic_json(result_dir / "run.json", evidence)
    try:
        # Pre-load forecast uses a conservative model-load delta; each later start
        # must leave >=16GiB projected MemAvailable and >=8GiB actual hard floor.
        first_forecast = plan.get("forecast_engine_rss_bytes")
        if type(first_forecast) is not int or first_forecast <= 0:
            raise ValueError("plan must bind a positive forecast_engine_rss_bytes")
        forecast = first_forecast
        observed_load_deltas = {}
        for index, name in enumerate(LANES):
            available_before = mem_available_bytes()
            gate = host_gate(available_bytes=available_before, forecast_next_bytes=forecast)
            atomic_json(result_dir / f"host-gate-before-{name}.json", gate)
            if not gate["pass"]:
                raise RuntimeError(f"host-memory forecast gate failed before {name}")
            cfg_path, manifest_path, _, cfg, manifest = lanes[name]
            # This staged ledger is bootstrap-only. It is never exposed to HTTP;
            # measured PSS-based admissions replace all three after READY.
            client_key = secrets.token_urlsafe(32)
            service, ready = runner_factory.construct_service(
                engine_root=base / "engine", config_path=cfg_path,
                precision_policy_path=policy_path,
                resolved_manifest=manifest,
                client_api_key=client_key)
            if name == "lane-a":
                frontend_client_key = client_key
            services[name] = service
            engine = service.engine
            ordinal = int(manifest["gpu_ordinal"])
            sample = resource_sample(ordinal, engine.proc.pid)
            delta = max(0, available_before - sample["host_mem_available_bytes"])
            observed_load_deltas[name] = delta
            forecast = max(forecast, delta)
            if sample["gpu_uuid"] != manifest["gpu_uuid"]:
                raise RuntimeError(f"{name}: runtime GPU UUID differs from staged manifest")
            if sample["gpu_cap_w"] != float(expected_cap):
                raise RuntimeError(f"{name}: runtime GPU power cap differs from reviewed plan")
            if sample["gpu_total_bytes"] - sample["gpu_used_bytes"] < GPU_RESERVE:
                raise RuntimeError(f"{name}: per-GPU 2GiB reserve failed after READY")
            if sample["gpu_temperature_c"] >= float(plan.get("gpu_temp_max_c", 85)):
                raise RuntimeError(f"{name}: GPU temperature gate failed")
            if sample["host_mem_available_bytes"] < HOST_HARD_FLOOR:
                raise RuntimeError(f"{name}: host hard floor failed after READY")
            if shutil_disk_free(checkpoint_dir) < checkpoint_scratch:
                raise RuntimeError(f"{name}: checkpoint disk scratch gate failed")
            ready["resource_sample_after_ready"] = sample
            ready_rows[name] = ready
            evidence["lanes"][name] = {"ready_info": ready["ready_info"],
                                       "configuration_fingerprint": ready["configuration_fingerprint"],
                                       "launch_evidence_fingerprint": ready["launch_evidence_fingerprint"],
                                       "model_identity": ready["model_identity"],
                                       "resource_sample": sample,
                                       "observed_memavailable_delta_bytes": delta,
                                       "engine_pid": ready["engine_pid"],
                                       "engine_generation": ready["engine_generation"]}
            atomic_json(result_dir / f"ready-{name}.json", evidence["lanes"][name])
            atomic_json(result_dir / "run.json", evidence)
        # Re-read each process's smaps_rollup after every engine is READY. PSS
        # is the shared-host attribution used by admission; the earlier samples
        # remain individually preserved as per-load evidence.
        samples = {}
        for lane in LANES:
            manifest = manifests[lane]
            sample = resource_sample(int(manifest["gpu_ordinal"]),
                                     services[lane].engine.proc.pid)
            if (sample["gpu_uuid"] != manifest["gpu_uuid"]
                    or sample["gpu_cap_w"] != float(expected_cap)
                    or sample["gpu_total_bytes"] - sample["gpu_used_bytes"] < GPU_RESERVE
                    or sample["gpu_temperature_c"] >= float(plan.get("gpu_temp_max_c", 85))
                    or sample["host_mem_available_bytes"] < HOST_HARD_FLOOR):
                raise RuntimeError(f"{lane}: post-all-ready resource gate failed")
            samples[lane] = sample
            evidence["lanes"][lane]["post_all_ready_resource_sample"] = sample
        atomic_json(result_dir / "run.json", evidence)
        scratch_host = plan.get("checkpoint_scratch_host_bytes")
        if type(scratch_host) is not int or scratch_host < 0:
            raise ValueError("plan must bind nonnegative checkpoint_scratch_host_bytes")
        measured_admissions = install_measured_admissions(
            services, samples, total_host_bytes=mem_total_bytes(),
            available_host_bytes=samples["lane-c"]["host_mem_available_bytes"],
            checkpoint_scratch_host_bytes=scratch_host)
        atomic_json(result_dir / "measured-admissions.json", measured_admissions)
        fingerprints = {lane: row["configuration_fingerprint"] for lane, row in ready_rows.items()}
        if len(set(fingerprints.values())) != 1:
            raise RuntimeError("three lanes do not share a semantic compatibility fingerprint")
        ids = [row["model_identity"] for row in ready_rows.values()]
        if len({json.dumps(x, sort_keys=True) for x in ids}) != 1:
            raise RuntimeError("three READY lanes do not share the same audited ModelIdentity")
        identity = services["lane-a"].engine.model_identity
        operator = secrets.token_urlsafe(32)
        harness = real_model_harness.assemble_ready_services(
            stage="three_to_one", services=services, model_identity=identity,
            checkpoint_root=checkpoint_dir, agent_slots=AGENTS,
            configuration_fingerprints=fingerprints, frontend_lane="lane-a",
            operator_token=operator)
        results = qc.JsonlResults(result_dir / "probes.jsonl")
        receipt_evidence = ReceiptEvidence(harness, results); receipt_evidence.install()
        # Initial per-lane lifecycle proof before HTTP traffic.
        for lane, worker in harness.workers.items():
            start = time.time(); snap = worker.quiesce_snapshot(harness.store, 60.0)
            if snap.state != "parked":
                raise RuntimeError(f"{lane}: initial full-batch snapshot failed")
            resumed = worker.resume(60.0)
            if resumed.state != "resumed":
                raise RuntimeError(f"{lane}: initial full-batch resume failed")
            results.write(qc.Probe(f"native:initial-snapshot-resume:{lane}", "PASS", start,
                                   time.time(), None, {"checkpoint_id": snap.checkpoint_id}))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]
        httpd = harness.start_http(port=port, host="127.0.0.1")
        client = qc.HttpProbeClient(f"http://127.0.0.1:{port}", frontend_client_key,
                                    results=results, timeout_s=120.0)
        model = services["lane-a"].model
        histories, _ = _seed_and_prompts(client, harness, model, services,
                                         result_dir / "probes.jsonl")
        initial = {a: messages + [{"role": "user", "content": "Add one short detail to the remembered test fact."}]
                   for a, messages in histories.items()}
        matrix_histories = {a: [dict(m) for m in messages] for a, messages in initial.items()}
        matrix = checked_endpoint_matrix(client, model=model, messages_by_agent=matrix_histories,
                                         results=results, max_new=4)
        if len(matrix) != 36 or any(p.status != "PASS" for p in matrix):
            raise RuntimeError("pre-migration endpoint matrix failed")
        state_snapshot = lambda: _state_snapshot(harness)
        negatives = (client.negative_routing(model=model, messages=matrix_histories["agent-alpha"],
                                             known_agent="agent-alpha", unknown_agent="unknown-agent",
                                             state_snapshot=state_snapshot)
                     + client.managed_control_negatives(state_snapshot=state_snapshot)
                     + client.migration_negative_controls(operator_token=operator,
                         valid_destination="lane-c", other_destination="missing-lane",
                         state_snapshot=state_snapshot))
        if any(x.status != "PASS" for x in negatives):
            raise RuntimeError("pre-migration negative controls failed")
        # The P-first baseline is prepared in a separately loaded and closed
        # single-worker process before any of these three migration workers are
        # constructed. Bind it to the same reviewed lane-a semantic profile.
        if prefix_baseline_path is None:
            raise ValueError("the fresh isolated P-first baseline is mandatory before migration loads")
        prefix_baseline = read_json(prefix_baseline_path)
        if (prefix_baseline.get("schema") != "strata-owner-isolated-baseline-v1"
                or prefix_baseline.get("closed_verified") is not True
                or prefix_baseline.get("configuration_fingerprint") != fingerprints["lane-a"]
                or set(prefix_baseline.get("fixtures", {})) != set(AGENTS)):
            raise RuntimeError("isolated P-first baseline is missing, unclosed, or config-incompatible")
        prompts, fixture_histories, native_first = {}, {}, {}
        for agent in AGENTS:
            fixture = prefix_baseline["fixtures"][agent]
            messages = fixture.get("messages")
            prompt = fixture.get("prompt_token_ids_P")
            baseline_row = fixture.get("baseline")
            if not isinstance(messages, list) or not isinstance(prompt, list) or not isinstance(baseline_row, dict):
                raise RuntimeError(f"{agent}: isolated baseline fixture schema is incomplete")
            rendered = qc.tokenize_full_history(services["lane-a"], messages,
                template_kwargs={"enable_thinking": False})
            if rendered != prompt:
                raise RuntimeError(f"{agent}: lane-a renders P differently from isolated baseline")
            prompts[agent] = prompt
            fixture_histories[agent] = [dict(message) for message in messages]
            native_first[agent] = baseline_row
        prefix_seed_rows = {}
        def persist_prefix_seed(row):
            key = row["agent"]
            prefix_seed_rows[key] = row
            atomic_json(result_dir / "native-prefix-seed-evidence.json", prefix_seed_rows)
        for agent in AGENTS:
            fixture = prefix_baseline["fixtures"][agent]
            if fixture.get("prompt_token_ids_P") != prompts[agent]:
                raise RuntimeError(f"{agent}: baseline P was altered before native seed")
            seed = qc.seed_owner_prefix_fixture(harness, agent=agent,
                prompt_ids=prompts[agent], tail_tokens=fixture["tail_tokens"],
                persist=persist_prefix_seed, require_empty_slot=False)
            if seed["consumed_prefix_token_ids_C"] != fixture.get("consumed_prefix_token_ids_C"):
                raise RuntimeError(f"{agent}: native C differs from the isolated baseline fixture")
        atomic_json(result_dir / "native-prefix-prime.json", {
            agent: {"P_sha256": prefix_seed_rows[agent]["P_sha256"],
                    "C_sha256": prefix_seed_rows[agent]["C_sha256"],
                    "P_count": prefix_seed_rows[agent]["P_count"],
                    "C_count": prefix_seed_rows[agent]["C_count"],
                    "status": prefix_seed_rows[agent]["status"]}
            for agent in AGENTS})
        expected_consumed_prefixes = {
            agent: list(prefix_baseline["fixtures"][agent]["consumed_prefix_token_ids_C"])
            for agent in AGENTS}
        aligned_prefixes = qc.checkpoint_prefixes(harness, expected_prompts=prompts,
                                                  expected_consumed_prefixes=expected_consumed_prefixes)
        atomic_json(result_dir / "native-prefix-checkpoints.json", aligned_prefixes)
        qc.assert_current_slots_match_saved(harness, aligned_prefixes)
        native = {"first": native_first, "prefixes": aligned_prefixes,
                  "source": "fresh-isolated-single-worker"}
        atomic_json(result_dir / "native-baseline-isolated.json", native)
        atomic_json(result_dir / "native-baseline-isolated-validation.json", {"status": "PASS"})
        native_metrics = install_native_metrics(harness, results)
        qc.install_durable_event_recorder(harness, results)
        native_metrics["phase"]["name"] = "busy-consolidation"
        install_rotation_metrics(harness, results, native_metrics)
        # Exercise one active request on each source lane that must exit.
        source_agents = ("agent-alpha", "agent-gamma")
        busy, busy_prefixes = busy_503_restore_then_consolidate(
            client, harness, model=model, messages_by_agent=matrix_histories,
            agents=source_agents, operator_token=operator,
            receipt_evidence=receipt_evidence, expected_prefixes=aligned_prefixes,
            expected_prompts=prompts, expected_consumed_prefixes=expected_consumed_prefixes,
            results=results)
        atomic_json(result_dir / "busy-consolidation.json",
                    [{"name": p.name, "status": p.status, "http_status": p.http_status,
                      "started_at": p.started_at, "completed_at": p.completed_at,
                      "detail": p.detail} for p in busy])
        if not busy or busy[0].status != "PASS" or not busy[-1].status == "PASS":
            raise RuntimeError("busy 503/drained retry consolidation failed")
        durable = qc.validate_durable_exit_order(
            str(results.path), source_lanes={"lane-a", "lane-b"}, expected_lanes=set(LANES))
        if not durable.get("pass"):
            raise RuntimeError("durable-before-source-exit audit failed")
        atomic_json(result_dir / "durable-exit-audit.json", durable)
        qc.assert_routes_reference_saved_payloads(harness, aligned_prefixes)
        routes = harness.coordinator.routes
        if sum(r.lane == "lane-c" for r in routes.values()) != 2 or sum(r.lane is None for r in routes.values()) != 4:
            raise RuntimeError("post-consolidation routes are not two resident/four parked")
        native_metrics["phase"]["name"] = "native-migrated"
        migrated = qc.migrated_native_branch(harness, prompts, native["first"], max_new=4,
                                             sampling={"temperature": 0.0})
        atomic_json(result_dir / "native-migrated-branch.json", migrated)
        validate_migrated_native(migrated)
        atomic_json(result_dir / "native-migrated-branch-validation.json", {"status": "PASS"})
        # Three complete, transcript-preserving rounds. Pair selection comes
        # from the live coordinator; every probe is tagged with its agent.
        rotation_histories = {a: [dict(m) for m in matrix_histories[a]] for a in AGENTS}
        for agent, trial in migrated["trials"].items():
            lane, _slot = AGENTS[agent]
            decoded = services[lane].tok.decode(trial["output_token_ids"])
            rotation_histories[agent].extend([
                {"role": "assistant", "content": decoded},
                {"role": "user", "content": f"Continue {agent}'s isolated history after native restored branch."}])
        rotation_rows = []
        native_metrics["phase"]["name"] = "rotation"
        for round_no in range(1, 4):
            for agent in AGENTS:
                p = client.request("/v1/chat/completions", model=model,
                    messages=rotation_histories[agent], agent_id=agent,
                    stream=(round_no % 2 == 0), max_new=4)
                p.name = f"rotation:r{round_no}:{agent}"; results.write(p); rotation_rows.append(p)
                if p.status != "PASS": raise RuntimeError(f"rotation failed for {agent}")
                rotation_histories[agent].extend([
                    {"role": "assistant", "content": p.detail["output_text"]},
                    {"role": "user", "content": f"Round {round_no}: add one distinct detail to {agent}'s fact."}])
                with harness.coordinator._cv:
                    resident = [a for a, r in harness.coordinator._routes.items()
                                if r.lane == "lane-c"]
                if len(resident) != 2:
                    raise RuntimeError("destination must have exactly two resident agents during rotation")
                pair_messages = {a: rotation_histories[a] for a in resident}
                pair = client.concurrent_pair(model=model, messages_by_agent=pair_messages,
                                              agents=(resident[0], resident[1]), max_new=4)
                for a, q in zip(resident, pair):
                    q.name = f"rotation:r{round_no}:concurrent:{a}"; results.write(q); rotation_rows.append(q)
                    if q.status != "PASS": raise RuntimeError(f"concurrent rotation failed for {a}")
                    rotation_histories[a].extend([
                        {"role": "assistant", "content": q.detail["output_text"]},
                        {"role": "user", "content": f"Concurrent round {round_no}: continue only {a}'s distinct history."}])
        validate_rotation_restore_reuse(native_metrics)
        rotation_metrics = summarize_rotation_metrics(native_metrics)
        atomic_json(result_dir / "rotation-native-metrics.json", rotation_metrics)
        atomic_json(result_dir / "rotation-histories.json", rotation_histories)
        post_matrix_histories = {a: [*rotation_histories[a],
                                     {"role": "user", "content": "Confirm the last detail."}]
                                 for a in AGENTS}
        native_metrics["phase"]["name"] = "http-post-matrix"
        post_matrix = checked_endpoint_matrix(client, model=model,
                                              messages_by_agent=post_matrix_histories,
                                              results=results, max_new=4)
        if len(post_matrix) != 36 or any(p.status != "PASS" for p in post_matrix):
            raise RuntimeError("post-migration endpoint matrix failed")
        retry = client.migration_retry_controls(destination="lane-c", operator_token=operator,
                                                other_destination="lane-a", state_snapshot=state_snapshot)
        if any(p.status != "PASS" for p in retry):
            raise RuntimeError("post-consolidation retry controls failed")
        # Cancellation is intentionally last: a failed/ambiguous partial-prefill
        # drain can gate admission and require owner recovery. It cannot erase a
        # completed core migration result.
        native_metrics["phase"]["name"] = "cancel"
        cancel_agent = "agent-alpha"
        previous_request = receipt_evidence.last_request_id(cancel_agent)
        cancel = client.cancel_stream_after_event(
            model=model,
            messages=rotation_histories[cancel_agent] + [
                {"role": "user", "content": "Produce a short streamed continuation for the drain test."}],
            agent_id=cancel_agent, max_new=32,
            drain_check=lambda: ({**receipt_evidence.last_agent_drained(cancel_agent),
                                  "receipt_confirmed": (
                                      receipt_evidence.last_agent_drained(cancel_agent).get("receipt_confirmed") is True
                                      and receipt_evidence.last_request_id(cancel_agent) != previous_request)}),
            drain_timeout_s=30.0)
        results.write(cancel)
        atomic_json(result_dir / "cancellation.json", {"status": cancel.status,
                                                       "http_status": cancel.http_status,
                                                       "detail": cancel.detail})
        completion = {"status": "PASS", "stages": {"ready_three_workers": "PASS",
            "initial_snapshots": "PASS", "http_pre_matrix": "PASS", "negative_controls": "PASS",
            "aligned_native_baseline": "PASS", "busy_then_consolidate": "PASS",
            "durable_source_exit_order": "PASS", "migrated_native_parity": "PASS",
            "three_round_rotation": "PASS", "rotation_restore_reuse": "PASS",
            "http_post_matrix": "PASS",
            "migration_retry_controls": "PASS",
            "cancel_stream_drain": cancel.status},
            "counts": {"pre_endpoint": len(matrix), "post_endpoint": len(post_matrix),
                       "negative": len(negatives), "rotation": len(rotation_rows)},
            "baseline_aligned_token_ids": {a: native["first"][a]["output_token_ids"] for a in AGENTS},
            "migrated_comparison": migrated["comparison"],
            "rotation_native_metrics": rotation_metrics,
            "limitations": ["physical GPU release is not claimed; three-worker sharing must be read from measured manifests",
                            "cancellation is a terminal separate case; BLOCKED means owner recovery is required",
                            "TTFT is not captured by the native completion hook",
                            "rotation rates are observational measurements from the correctness run, not a matching performance benchmark"]}
        atomic_json(result_dir / "complete.json", completion)
        evidence["stages"] = completion["stages"]
        atomic_json(result_dir / "run.json", evidence)
        return completion
    except BaseException as exc:
        # Keep exception text/tracebacks out of result artifacts: lower-level
        # diagnostics can echo request credentials or private model paths.
        evidence["failure_type"] = type(exc).__name__
        evidence["failed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(result_dir / "failure.json", evidence)
        raise
    finally:
        primary_exception = sys.exc_info()[0] is not None
        cleanup_failures = []
        if httpd is not None:
            try:
                httpd.shutdown(); httpd.server_close()
            except Exception as exc:
                cleanup_failures.append({"component": "loopback-http", "exception": type(exc).__name__})
        if harness is not None:
            try:
                harness.gateway._admin_token = None
            except Exception as exc:
                cleanup_failures.append({"component": "gateway-token", "exception": type(exc).__name__})
        cleanup_failures.extend(close_services_reverse(services))
        if cleanup_failures:
            atomic_json(result_dir / "cleanup-failures.json", cleanup_failures)
            completion = locals().get("completion")
            if completion is not None:
                completion["status"] = "FAIL"
                completion["cleanup"] = "FAIL"
                atomic_json(result_dir / "complete.json", completion)
            elif not primary_exception:
                raise RuntimeError("owned service cleanup did not complete")


def shutil_disk_free(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def close_services_reverse(services: dict) -> list[dict]:
    """Attempt every owned engine close in reverse load order, even on errors."""
    failures = []
    for name in reversed(LANES):
        service = services.get(name)
        if service is None:
            continue
        try:
            service.migration_gateway = None
        except Exception as exc:
            failures.append({"component": "service-detach", "lane": name,
                             "exception": type(exc).__name__})
        try:
            service.engine.close()
        except Exception as exc:
            failures.append({"component": "engine-close", "lane": name,
                             "exception": type(exc).__name__})
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    a = ap.parse_args()
    run_three_worker(base=a.base, plan_path=a.plan, result_dir=a.result_dir,
                     checkpoint_dir=a.checkpoint_dir)


if __name__ == "__main__":
    main()
