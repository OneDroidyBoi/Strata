from __future__ import annotations

import importlib.util
import pathlib
import sys
import threading
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("owner_prefix_client", ROOT / "qualification_client.py")
qc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qc
spec.loader.exec_module(qc)


class FakeEngine:
    def __init__(self):
        self.slot_cv = threading.RLock()
        self.slot_held = [[]]
        self.info = {"slot_cache": 1, "session_export_batch": 1, "session_restore_batch": 1}
        self.gen = 1  # StrataEngine READY uses a positive integer lifecycle counter.
        self.proc = SimpleNamespace(pid=4242, poll=lambda: None)
        self.last = {"reused": 0, "prompt_read": 0, "finish": "stop", "generated": 1}
        self.inventory_count_override = None
        self.override_inventory_after_generation = None
        self.mutate_held_after_generation = False


class FakeWorker:
    def __init__(self, engine):
        self.engine = engine

    def _snapshot(self):
        count = self.engine.inventory_count_override
        if count is None:
            count = len(self.engine.slot_held[0])
        rows = (("slot-0", "slot", count),) if count else ()
        return self.engine.gen, tuple(row[0] for row in rows), (self.engine.gen, rows)


class FakeCoordinator:
    def __init__(self, engine):
        self.routes = {"agent-a": SimpleNamespace(lane="lane-a", slot=0)}
        self.engine = engine

    def generate(self, agent, ids, max_new, sampling, cancel, *, timeout_s=30):
        ids = list(ids)
        if max_new == 1:
            output = [999]  # deliberately discarded; never appended to P/C
            self.engine.slot_held[0] = ids + output[:-1]
            self.engine.last = {"reused": 0, "prompt_read": len(ids), "finish": "stop", "generated": 1}
            self.engine.inventory_count_override = self.engine.override_inventory_after_generation
            if self.engine.mutate_held_after_generation:
                self.engine.slot_held[0][-1] += 1
        else:
            held_before = list(self.engine.slot_held[0])
            reused = min(len(held_before), len(ids)) if held_before == ids[:len(held_before)] else 0
            output = [901, 902, 903, 904][:max_new]
            self.engine.slot_held[0] = ids + output[:-1]
            self.engine.last = {"reused": reused, "prompt_read": len(ids) - reused,
                                "finish": "length", "generated": len(output)}
        yield from output


def harness():
    engine = FakeEngine()
    worker = FakeWorker(engine)
    coordinator = FakeCoordinator(engine)
    frontend = SimpleNamespace(encode_prompt=lambda messages, tools, kwargs, store=False: [1, 2, 3, 4, 5])
    return SimpleNamespace(stage="single_worker", workers={"lane-a": worker},
                           coordinator=coordinator, frontend=frontend), engine


class OwnerPrefixFixtureTests(unittest.TestCase):
    def test_missing_or_invalid_ready_generation_rejected_and_closed(self):
        for value in (None, True, 0, -1, 1.0, "fresh-process-1"):
            with self.subTest(value=value):
                h, engine = harness()
                if value is None:
                    del engine.gen  # proc exists, but READY generation is absent
                else:
                    engine.gen = value
                closed, saved = [], []
                with self.assertRaisesRegex(RuntimeError, "live, identified"):
                    qc.fresh_isolated_native_baseline(lambda: h,
                        fixtures={"alpha": ("agent-a", [], 2)},
                        close_harness=closed.append,
                        persist=lambda *args: saved.append(args))
                self.assertEqual(closed, [h])
                self.assertEqual(saved, [])
                self.assertEqual(engine.slot_held, [[]])

    def test_process_change_during_seed_or_continuation_rejects_baseline(self):
        for phase in ("seed", "continuation"):
            for change in ("generation", "same_pid_new_process", "exit"):
                with self.subTest(phase=phase, change=change):
                    h, engine = harness()
                    original = h.coordinator.generate
                    def generate(agent, ids, max_new, sampling, cancel, **kwargs):
                        yield from original(agent, ids, max_new, sampling, cancel, **kwargs)
                        if (max_new == 1) == (phase == "seed"):
                            if change == "generation":
                                engine.gen += 1
                            elif change == "same_pid_new_process":
                                engine.proc = SimpleNamespace(pid=4242, poll=lambda: None)
                            else:
                                engine.proc.poll = lambda: 0
                    h.coordinator.generate = generate
                    closed, saved = [], []
                    with self.assertRaises(RuntimeError):
                        qc.fresh_isolated_native_baseline(lambda: h,
                            fixtures={"alpha": ("agent-a", [], 2)},
                            close_harness=closed.append,
                            persist=lambda _, row: saved.append(row))
                    self.assertEqual(closed, [h])
                    self.assertFalse(any("baseline" in row for row in saved))

    def test_p_first_strict_prefix_seed_drains_and_matches_slot_and_inventory(self):
        h, engine = harness()
        persisted = []
        p = [10, 11, 12, 13, 14, 15]
        row = qc.seed_owner_prefix_fixture(h, agent="agent-a", prompt_ids=p, tail_tokens=2,
                                           persist=persisted.append)
        self.assertEqual(row["prompt_token_ids_P"], p)
        self.assertEqual(row["consumed_prefix_token_ids_C"], p[:-2])
        self.assertEqual(row["held_after_token_ids"], p[:-2])
        self.assertEqual(row["inventory_slot_tokens"], len(p) - 2)
        self.assertEqual(row["native_output_token_ids_discarded"], [999])
        self.assertTrue(row["drain_confirmed"])
        self.assertEqual(persisted[0]["stage"], "before-native-seed")
        self.assertEqual(persisted[0]["prompt_token_ids_P"], p)
        self.assertEqual(persisted[-1]["status"], "PASS")

    def test_inventory_or_slot_mismatch_is_a_hard_failure_after_evidence_persist(self):
        h, engine = harness()
        engine.override_inventory_after_generation = 99
        persisted = []
        with self.assertRaisesRegex(RuntimeError, "inventory slot count"):
            qc.seed_owner_prefix_fixture(h, agent="agent-a", prompt_ids=[1, 2, 3, 4],
                                         tail_tokens=1, persist=persisted.append)
        self.assertEqual(persisted[-1]["stage"], "native-seed-observed")
        self.assertEqual(persisted[-1]["prompt_token_ids_P"], [1, 2, 3, 4])
        self.assertEqual(persisted[-1]["consumed_prefix_token_ids_C"], [1, 2, 3])
        self.assertEqual(persisted[-1]["inventory_slot_tokens"], 99)

    def test_held_id_mismatch_is_a_hard_failure_after_evidence_persist(self):
        h, engine = harness()
        engine.mutate_held_after_generation = True
        persisted = []
        with self.assertRaisesRegex(RuntimeError, "slot-held IDs"):
            qc.seed_owner_prefix_fixture(h, agent="agent-a", prompt_ids=[1, 2, 3, 4],
                                         tail_tokens=1, persist=persisted.append)
        self.assertEqual(persisted[-1]["stage"], "native-seed-observed")
        self.assertNotEqual(persisted[-1]["held_after_token_ids"], [1, 2, 3])

    def test_fresh_comparison_rejects_preexisting_slot_and_persists_arrays(self):
        h, engine = harness()
        engine.slot_held[0] = [80]
        persisted = []
        with self.assertRaisesRegex(RuntimeError, "not empty"):
            qc.seed_owner_prefix_fixture(h, agent="agent-a", prompt_ids=[1, 2, 3, 4],
                                         tail_tokens=1, persist=persisted.append)
        self.assertEqual(persisted[0]["prompt_token_ids_P"], [1, 2, 3, 4])
        self.assertEqual(persisted[0]["consumed_prefix_token_ids_C"], [1, 2, 3])

    def test_isolated_baseline_requires_positive_reuse_bounded_by_c_and_tail(self):
        h, engine = harness()
        closed = []
        saved = []
        result = qc.fresh_isolated_native_baseline(lambda: h,
            fixtures={"agent-a": ("agent-a", [{"role": "user", "content": "exact history"}], 2)},
            close_harness=lambda obj: closed.append(obj),
            persist=lambda agent, row: saved.append((agent, row)))
        self.assertEqual(result["agent-a"]["baseline"]["reused"], 3)
        self.assertEqual(result["agent-a"]["baseline"]["prompt_read"], 2)
        self.assertEqual(len(closed), 1)
        self.assertEqual(saved[-1][0], "agent-a")
        self.assertEqual(saved[-1][1]["seed"]["consumed_prefix_token_ids_C"], [1, 2, 3])

    def test_isolated_baseline_closes_worker_even_when_reuse_gate_fails(self):
        h, engine = harness()
        engine.slot_held[0] = [77]  # baseline is not a fresh slot
        closed = []
        with self.assertRaisesRegex(RuntimeError, "not empty"):
            qc.fresh_isolated_native_baseline(lambda: h,
                fixtures={"agent-a": ("agent-a", [{"role": "user", "content": "exact history"}], 1)},
                close_harness=lambda obj: closed.append(obj))
        self.assertEqual(closed, [h])

    def test_fresh_process_can_measure_multiple_histories_sequentially_through_its_owned_slot(self):
        h, engine = harness()
        closed = []
        result = qc.fresh_isolated_native_baseline(lambda: h,
            fixtures={
                "alpha": ("agent-a", [{"role": "user", "content": "alpha"}], 2),
                "beta": ("agent-a", [{"role": "user", "content": "beta"}], 2),
            },
            close_harness=lambda obj: closed.append(obj))
        self.assertEqual(set(result), {"alpha", "beta"})
        self.assertEqual(result["alpha"]["seed"]["consumed_prefix_token_ids_C"], [1, 2, 3])
        self.assertEqual(result["beta"]["seed"]["consumed_prefix_token_ids_C"], [1, 2, 3])
        self.assertEqual(closed, [h])


if __name__ == "__main__":
    unittest.main()
