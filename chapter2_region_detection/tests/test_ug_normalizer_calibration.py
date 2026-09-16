import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from formal_experiments.data_collection.collect_ug_calibration_states import (
    CALIBRATION_SEEDS as COLLECTOR_SEEDS,
    STATE_RECORD_FIELDS,
    build_summary,
    state_only_record,
)
from formal_experiments.evaluation.calibrate_ug_normalizer import (
    CALIBRATION_SEED_ORDER,
    CEM_ALPHA,
    CEM_ELITE_RATIO,
    CEM_NUM_ITERATIONS,
    CEM_POPULATION_SIZE,
    CEM_PROB_FLOOR,
    DEFAULT_REWARD_MODEL,
    DEFAULT_STATES,
    DEFAULT_WORLD_MODEL,
    PLANNER_CALLS_PER_AGENT,
    calibration_cem_config,
    load_calibration_records,
    select_balanced_warmup_states,
    tensor_summary,
)
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


def state(index=0):
    value = np.zeros(FORMAL_STATE_DIM, dtype=np.float32)
    value[0] = float(index) * 0.01
    return value


def raw_record(seed, agent, index):
    return {
        "split": "calibration",
        "episode_seed": int(seed),
        "agent_name": str(agent),
        "decision_index": int(index),
        "global_tick_start": int(index),
        "state": state(index).tolist(),
    }


def full_records(per_agent=104):
    rows = []
    for agent in BLUE_AGENTS:
        for index in range(per_agent):
            seed = CALIBRATION_SEED_ORDER[index % len(CALIBRATION_SEED_ORDER)]
            rows.append(raw_record(seed, agent, index))
    return rows


class TestUGNormalizerCalibration(unittest.TestCase):

    def test_state_only_projection_has_exact_schema_and_no_hidden_fields(self):
        transition = SimpleNamespace(
            episode_seed=3000,
            agent_name="blue_agent_0",
            decision_index=3,
            global_tick_start=7,
            state=state(3),
            incident_host_ids=("host",),
            response_reward=-9.0,
        )
        item = state_only_record(transition)
        self.assertEqual(tuple(item.keys()), STATE_RECORD_FIELDS)
        self.assertNotIn("incident_host_ids", item)
        self.assertNotIn("response_reward", item)
        self.assertEqual(item["split"], "calibration")

    def test_state_only_projection_rejects_non_calibration_seed(self):
        transition = SimpleNamespace(
            episode_seed=4000,
            agent_name="blue_agent_0",
            decision_index=0,
            global_tick_start=0,
            state=state(),
        )
        with self.assertRaises(ValueError):
            state_only_record(transition)

    def test_collection_summary_requires_all_agents_and_reports_no_labels(self):
        rows = full_records(per_agent=104)
        summary = build_summary(rows)
        self.assertEqual(summary["seeds"], list(COLLECTOR_SEEDS))
        self.assertFalse(summary["contains_hidden_truth"])
        self.assertFalse(summary["contains_reward_labels"])
        for agent in BLUE_AGENTS:
            self.assertGreaterEqual(summary["per_agent_counts"][agent], 100)

        with self.assertRaises(RuntimeError):
            build_summary([item for item in rows if item["agent_name"] != "blue_agent_4"])

    def test_loader_rejects_extra_hidden_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.jsonl"
            item = raw_record(3000, "blue_agent_0", 0)
            item["incident_host_ids"] = ["hidden"]
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration_records(path)

    def test_loader_accepts_exact_calibration_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.jsonl"
            items = [
                raw_record(3000, "blue_agent_0", 0),
                raw_record(3001, "blue_agent_0", 1),
            ]
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in items),
                encoding="utf-8",
            )
            loaded = load_calibration_records(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["warmup"].episode_seed, 3000)
            self.assertEqual(loaded[1]["warmup"].episode_seed, 3001)

    def test_balanced_selection_is_deterministic_and_covers_all_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.jsonl"
            rows = full_records(per_agent=104)
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in rows),
                encoding="utf-8",
            )
            loaded = load_calibration_records(path)

        first = select_balanced_warmup_states(
            loaded,
            agent_name="blue_agent_0",
            planner_calls=PLANNER_CALLS_PER_AGENT,
        )
        second = select_balanced_warmup_states(
            loaded,
            agent_name="blue_agent_0",
            planner_calls=PLANNER_CALLS_PER_AGENT,
        )

        first_seeds = [item.episode_seed for item in first]
        second_seeds = [item.episode_seed for item in second]
        self.assertEqual(first_seeds, second_seeds)
        self.assertEqual(tuple(first_seeds[:8]), CALIBRATION_SEED_ORDER)

        counts = Counter(first_seeds)
        self.assertEqual(set(counts), set(CALIBRATION_SEED_ORDER))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_balanced_selection_rejects_missing_seed_or_insufficient_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.jsonl"
            rows = [
                raw_record(seed, "blue_agent_0", idx)
                for idx, seed in enumerate(CALIBRATION_SEED_ORDER[:-1])
            ]
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in rows),
                encoding="utf-8",
            )
            loaded = load_calibration_records(path)
        with self.assertRaises(ValueError):
            select_balanced_warmup_states(
                loaded,
                agent_name="blue_agent_0",
                planner_calls=8,
            )

    def test_calibration_cem_profile_is_formal_and_reproducible(self):
        cfg0 = calibration_cem_config(agent_index=0, device="cpu")
        cfg4 = calibration_cem_config(agent_index=4, device="cpu")
        self.assertEqual(cfg0.horizon, 4)
        self.assertEqual(cfg0.n_actions, 4)
        self.assertEqual(cfg0.population_size, CEM_POPULATION_SIZE)
        self.assertEqual(cfg0.num_iterations, CEM_NUM_ITERATIONS)
        self.assertEqual(cfg0.elite_ratio, CEM_ELITE_RATIO)
        self.assertEqual(cfg0.alpha, CEM_ALPHA)
        self.assertEqual(cfg0.prob_floor, CEM_PROB_FLOOR)
        self.assertEqual(cfg4.seed - cfg0.seed, 4)

    def test_defaults_bind_v2_artifacts_and_tensor_summary_is_finite(self):
        self.assertIn("ug_cem_v2/step6", DEFAULT_STATES)
        self.assertIn("world_model_v2", DEFAULT_WORLD_MODEL)
        self.assertIn("world_model_v2", DEFAULT_REWARD_MODEL)
        self.assertNotIn("outputs/world_model/", DEFAULT_WORLD_MODEL)
        self.assertNotIn("outputs/world_model/", DEFAULT_REWARD_MODEL)

        summary = tensor_summary(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(summary["shape"], [3])
        self.assertEqual(summary["values"], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            tensor_summary(torch.tensor([float("nan")]))


if __name__ == "__main__":
    unittest.main()
