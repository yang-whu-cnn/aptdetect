import unittest
from types import SimpleNamespace

import numpy as np

from formal_experiments.evaluation.run_ug_cem_step7_smoke import (
    DEFAULT_NORMALIZERS,
    DEFAULT_OUT,
    FAMILY_DURATION,
    LOCAL_LONG_DECISIONS_PER_EPISODE,
    LOCAL_LONG_SEEDS,
    LOCAL_SHORT_DECISIONS_PER_EPISODE,
    LOCAL_SHORT_SEEDS,
    OFFICIAL_TICKS,
    OFFICIAL_TRAIN_SEEDS,
    _assert_resolution_contract,
    _ordered_local_records,
    _validate_episode_step_accounting,
    run_local_state_smoke_profile,
)
from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM


def formal_state(available=False):
    value=np.zeros(FORMAL_STATE_DIM,dtype=np.float32)
    value[17]=1.0 if available else 0.0
    return value


class FakePlanner:
    def __init__(self, action_id=0):
        self.action_id=int(action_id)
        self.reset_calls=0
        self.plan_calls=0

    def reset_episode(self):
        self.reset_calls += 1

    def plan(self, state):
        self.plan_calls += 1
        return SimpleNamespace(
            action_id=self.action_id,
            best_score=1.0,
            expected_return=2.0,
            uncertainty=0.5,
            planning_latency_sec=0.01,
        )


def records_for(seeds, per_seed):
    rows=[]
    for seed in seeds:
        for i in range(per_seed):
            agent=BLUE_AGENTS[i % len(BLUE_AGENTS)]
            rows.append({
                "warmup": SimpleNamespace(
                    episode_seed=int(seed),
                    agent_name=agent,
                    state=formal_state(),
                ),
                "decision_index": i // len(BLUE_AGENTS),
                "global_tick_start": i // len(BLUE_AGENTS),
            })
    return rows


class TestUGCEMStep7Smoke(unittest.TestCase):

    def test_frozen_smoke_profiles_match_taskbook(self):
        self.assertEqual(LOCAL_SHORT_SEEDS,(3000,))
        self.assertEqual(LOCAL_SHORT_DECISIONS_PER_EPISODE,20)
        self.assertEqual(LOCAL_LONG_SEEDS,(3000,3001,3002,3003,3004))
        self.assertEqual(LOCAL_LONG_DECISIONS_PER_EPISODE,100)
        self.assertEqual(OFFICIAL_TRAIN_SEEDS,(1000,1001))
        self.assertEqual(OFFICIAL_TICKS,50)

    def test_default_paths_are_step6_v2_and_step7(self):
        self.assertIn("ug_cem_v2/step6",DEFAULT_NORMALIZERS)
        self.assertIn("ug_cem_v2/step7",DEFAULT_OUT)

    def test_family_duration_contract(self):
        self.assertEqual(FAMILY_DURATION,{
            "Sleep":1,
            "Analyse":2,
            "Remove":3,
            "Restore":5,
        })

    def test_ordered_local_records_uses_tick_agent_decision_order(self):
        rows=[
            {"warmup":SimpleNamespace(episode_seed=3000,agent_name="blue_agent_1",state=formal_state()),"decision_index":0,"global_tick_start":0},
            {"warmup":SimpleNamespace(episode_seed=3000,agent_name="blue_agent_0",state=formal_state()),"decision_index":1,"global_tick_start":1},
            {"warmup":SimpleNamespace(episode_seed=3000,agent_name="blue_agent_0",state=formal_state()),"decision_index":0,"global_tick_start":0},
        ]
        out=_ordered_local_records(rows,seed=3000)
        self.assertEqual([x["warmup"].agent_name for x in out],["blue_agent_0","blue_agent_1","blue_agent_0"])

    def test_local_smoke_exact_call_count_and_development_only_label(self):
        planners={agent:FakePlanner(i % 4) for i,agent in enumerate(BLUE_AGENTS)}
        rows=records_for((3000,),25)
        report=run_local_state_smoke_profile(
            planners=planners,records=rows,seeds=(3000,),decisions_per_episode=20,profile_name="short"
        )
        self.assertEqual(report["total_planner_calls"],20)
        self.assertTrue(report["development_only"])
        self.assertFalse(report["closed_loop_cc4"])
        self.assertTrue(report["pass"])

    def test_local_smoke_resets_all_planners_at_episode_boundary(self):
        planners={agent:FakePlanner() for agent in BLUE_AGENTS}
        rows=records_for((3000,3001),12)
        run_local_state_smoke_profile(
            planners=planners,records=rows,seeds=(3000,3001),decisions_per_episode=10,profile_name="two"
        )
        for planner in planners.values():
            self.assertEqual(planner.reset_calls,2)

    def test_local_smoke_rejects_insufficient_records(self):
        planners={agent:FakePlanner() for agent in BLUE_AGENTS}
        rows=records_for((3000,),3)
        with self.assertRaises(RuntimeError):
            run_local_state_smoke_profile(
                planners=planners,records=rows,seeds=(3000,),decisions_per_episode=4,profile_name="bad"
            )

    def test_sleep_resolution_contract(self):
        ok=SimpleNamespace(fallback=False,executed_action_family="Sleep")
        _assert_resolution_contract(state=formal_state(False),action_id=0,resolution=ok)
        with self.assertRaises(RuntimeError):
            _assert_resolution_contract(
                state=formal_state(False),action_id=0,
                resolution=SimpleNamespace(fallback=True,executed_action_family="Sleep")
            )

    def test_targeted_available_resolution_contract(self):
        for action_id,family in ((1,"Analyse"),(2,"Remove"),(3,"Restore")):
            _assert_resolution_contract(
                state=formal_state(True),action_id=action_id,
                resolution=SimpleNamespace(fallback=False,executed_action_family=family)
            )

    def test_targeted_unavailable_must_fallback_sleep(self):
        for action_id in (1,2,3):
            _assert_resolution_contract(
                state=formal_state(False),action_id=action_id,
                resolution=SimpleNamespace(fallback=True,executed_action_family="Sleep")
            )

    def test_targeted_contract_rejects_wrong_runtime_behavior(self):
        with self.assertRaises(RuntimeError):
            _assert_resolution_contract(
                state=formal_state(True),action_id=1,
                resolution=SimpleNamespace(fallback=True,executed_action_family="Sleep")
            )
        with self.assertRaises(RuntimeError):
            _assert_resolution_contract(
                state=formal_state(False),action_id=3,
                resolution=SimpleNamespace(fallback=False,executed_action_family="Restore")
            )

    def test_episode_step_accounting_accepts_minus_one_to_49_for_50_steps(self):
        _validate_episode_step_accounting(
            controller_tick_start=-1,
            controller_tick_end=49,
            environment_steps_executed=50,
            requested_steps=50,
            all_agents_done=True,
        )

    def test_episode_step_accounting_rejects_short_or_unfinished_episode(self):
        with self.assertRaises(RuntimeError):
            _validate_episode_step_accounting(
                controller_tick_start=-1,
                controller_tick_end=48,
                environment_steps_executed=49,
                requested_steps=50,
                all_agents_done=True,
            )

        with self.assertRaises(RuntimeError):
            _validate_episode_step_accounting(
                controller_tick_start=-1,
                controller_tick_end=49,
                environment_steps_executed=50,
                requested_steps=50,
                all_agents_done=False,
            )

    def test_local_smoke_action_counts_sum_to_total(self):
        planners={agent:FakePlanner(i % 4) for i,agent in enumerate(BLUE_AGENTS)}
        rows=records_for((3000,),20)
        report=run_local_state_smoke_profile(
            planners=planners,records=rows,seeds=(3000,),decisions_per_episode=20,profile_name="count"
        )
        self.assertEqual(sum(report["action_counts"].values()),20)


if __name__ == "__main__":
    unittest.main()
