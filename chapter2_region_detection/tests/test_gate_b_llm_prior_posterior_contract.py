import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from pathlib import Path

from formal_experiments.ours.llm_prior_v2 import (
    FORMAL_ACTION_NAMES,
    FORMAL_HORIZON,
    FORMAL_K_CANDIDATES,
    FORMAL_LLM_API_MODEL,
    FORMAL_LLM_MODEL_LABEL,
    FORMAL_LLM_PROVIDER,
    FORMAL_LLM_TEMPERATURE,
    FormalLLMPriorConfig,
    build_formal_prompt,
    deterministic_fallback_plans,
    parse_prior_response,
)
from formal_experiments.data_collection.incident_response import IncidentResponseBookkeeper
from formal_experiments.ours.posterior_features import (
    PLAN_ONE_HOT_DIM,
    POSTERIOR_CANDIDATE_DIM,
    build_posterior_candidate_features,
    evidence_from_rollout,
    encode_plan_one_hot,
)
from shared.formal_state import FORMAL_STATE_DIM


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "lwm_rl_gate_b_v2_1.yaml"
REGISTRY_PATH = ROOT / "configs" / "llm_model_registry_v1.yaml"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TestGateBLLMPriorPosteriorContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_formal_action_k_h_and_model_contract(self):
        self.assertEqual(FORMAL_ACTION_NAMES, ("no_op","analyse","remove","restore"))
        self.assertEqual(FORMAL_K_CANDIDATES, 6)
        self.assertEqual(FORMAL_HORIZON, 4)
        self.assertEqual(FORMAL_LLM_PROVIDER, "ofox_openai_compatible")
        self.assertEqual(FORMAL_LLM_MODEL_LABEL, "GPT-5.6 Sol")
        self.assertEqual(FORMAL_LLM_API_MODEL, "openai/gpt-5.6-sol")
        self.assertEqual(FORMAL_LLM_TEMPERATURE, 0.2)
        llm=self.cfg["llm_prior"]
        registry=self.registry["registry"]
        tier_h=self.registry["models"]["tier_h"]
        self.assertEqual(llm["mode"],"registry_driven_multi_model")
        self.assertEqual(registry["gateway"],"ofox")
        self.assertEqual(llm["gateway_base_url"],"https://api.ofox.ai/v1")
        self.assertEqual(tier_h["exact_model_id"],"openai/gpt-5.6-sol")
        self.assertEqual(llm["api_key_env"],"OFOX_API_KEY")
        self.assertEqual(llm["k_candidates"],6)
        self.assertEqual(llm["plan_horizon"],4)
        self.assertEqual(llm["generation_temperature"],0.2)

    def test_prompt_uses_only_four_action_vocab_and_d27_state(self):
        state=np.arange(FORMAL_STATE_DIM,dtype=np.float32)/100.0
        messages=build_formal_prompt(state,agent_name="blue_agent_0")
        joined="\n".join(x["content"] for x in messages)
        for name in FORMAL_ACTION_NAMES:
            self.assertIn(name,joined)
        for forbidden in ("monitor","light_evidence","heavy_evidence","local_mitigate","strong_mitigate","control_traffic"):
            self.assertNotIn(forbidden,joined)
        self.assertIn("hidden compromise truth",joined)

    def test_parser_valid_six_candidates_and_preferences_sum_one(self):
        payload={"candidates":[
            {"actions":["no_op","analyse","remove","restore"],"prior_score":0.9},
            {"actions":["analyse","analyse","remove","restore"],"prior_score":0.8},
            {"actions":["remove","analyse","remove","restore"],"prior_score":0.7},
            {"actions":["restore","analyse","remove","restore"],"prior_score":0.6},
            {"actions":["no_op","no_op","no_op","no_op"],"prior_score":0.5},
            {"actions":["restore","restore","restore","restore"],"prior_score":0.4},
        ]}
        out=parse_prior_response(json.dumps(payload))
        self.assertEqual(out.plans.shape,(6,4))
        self.assertEqual(out.raw_prior_scores.shape,(6,))
        self.assertAlmostEqual(float(out.prior_preferences.sum()),1.0,places=6)
        self.assertEqual(out.sources,("llm",)*6)

    def test_parser_deduplicates_and_keeps_highest_score(self):
        payload={"candidates":[
            {"actions":["no_op","analyse","remove","restore"],"prior_score":0.2},
            {"actions":["no_op","analyse","remove","restore"],"prior_score":0.9},
        ]}
        out=parse_prior_response(json.dumps(payload))
        self.assertAlmostEqual(float(out.raw_prior_scores[0]),0.9,places=6)
        self.assertEqual(out.sources[0],"llm")
        self.assertEqual(out.sources.count("fallback_uniform"),5)

    def test_invalid_or_empty_response_uses_neutral_deterministic_fallback(self):
        first=parse_prior_response("not json")
        second=parse_prior_response("not json")
        np.testing.assert_array_equal(first.plans,second.plans)
        np.testing.assert_allclose(first.prior_preferences,np.full(6,1/6,dtype=np.float32))
        self.assertEqual(first.sources,("fallback_uniform",)*6)

    def test_parser_rejects_unknown_actions_wrong_length_and_bad_scores(self):
        payload={"candidates":[
            {"actions":["no_op","analyse","remove","control_traffic"],"prior_score":0.9},
            {"actions":["no_op","analyse"],"prior_score":0.8},
            {"actions":["no_op","analyse","remove","restore"],"prior_score":2.0},
        ]}
        out=parse_prior_response(json.dumps(payload))
        self.assertEqual(out.sources,("fallback_uniform",)*6)

    def test_fallback_samples_unique_formal_plan_space(self):
        plans=deterministic_fallback_plans(needed=6)
        self.assertEqual(len(plans),6)
        self.assertEqual(len(set(plans)),6)
        for plan in plans:
            self.assertEqual(len(plan),4)
            self.assertTrue(all(0 <= a < 4 for a in plan))

    def test_plan_one_hot_shape_and_categorical_semantics(self):
        plans=np.asarray([[0,1,2,3],[3,2,1,0],[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]])
        encoded=encode_plan_one_hot(plans)
        self.assertEqual(tuple(encoded.shape),(6,16))
        self.assertEqual(PLAN_ONE_HOT_DIM,16)
        self.assertTrue(torch.all(encoded.reshape(6,4,4).sum(dim=2)==1))

    def test_rollout_value_and_uncertainty_are_member_return_mean_and_population_std(self):
        members=torch.tensor([
            [1.,2.,3.,4.,5.],
            [2.,2.,2.,2.,2.],
            [0.,1.,0.,1.,0.],
            [3.,4.,5.,6.,7.],
            [5.,4.,3.,2.,1.],
            [1.,1.,2.,2.,3.],
        ])
        expected=members.mean(dim=1)
        ev=evidence_from_rollout(SimpleNamespace(expected_return=expected,member_returns=members))
        torch.testing.assert_close(ev.expected_return,expected)
        torch.testing.assert_close(ev.predictive_uncertainty,members.std(dim=1,unbiased=False))

    def test_posterior_feature_shape_is_46_and_contains_only_frozen_components(self):
        state=np.zeros(27,dtype=np.float32)
        plans=np.asarray([[0,1,2,3],[3,2,1,0],[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]])
        prior=np.asarray([.3,.2,.2,.1,.1,.1],dtype=np.float32)
        members=torch.arange(30,dtype=torch.float32).reshape(6,5)
        rollout=SimpleNamespace(expected_return=members.mean(dim=1),member_returns=members)
        feat=build_posterior_candidate_features(state,plans,prior,rollout)
        self.assertEqual(tuple(feat.shape),(6,46))
        self.assertEqual(POSTERIOR_CANDIDATE_DIM,46)
        self.assertEqual(46,27+16+1+1+1)

    def test_posterior_rejects_bad_prior_or_inconsistent_expected_return(self):
        state=np.zeros(27,dtype=np.float32)
        plans=np.zeros((6,4),dtype=np.int64)
        members=torch.ones(6,5)
        rollout=SimpleNamespace(expected_return=torch.ones(6),member_returns=members)
        with self.assertRaises(ValueError):
            build_posterior_candidate_features(state,plans,np.ones(6,dtype=np.float32),rollout)
        bad=SimpleNamespace(expected_return=torch.zeros(6),member_returns=members)
        with self.assertRaises(ValueError):
            evidence_from_rollout(bad)

    def test_b1_active_tick_penalty_equals_closed_event_eradication_duration(self):
        bookkeeper=IncidentResponseBookkeeper(
            episode_seed=1000,
            agent_name="blue_agent_0",
        )
        bookkeeper.reset(
            initial_red_presence_by_host={"host_a":False},
            global_tick=0,
        )
        first=bookkeeper.record_tick(
            global_tick_end=1,
            red_presence_after={"host_a":True},
        )
        self.assertEqual(first.incident_active_ticks,0)
        second=bookkeeper.record_tick(
            global_tick_end=2,
            red_presence_after={"host_a":True},
        )
        third=bookkeeper.record_tick(
            global_tick_end=3,
            red_presence_after={"host_a":True},
        )
        fourth=bookkeeper.record_tick(
            global_tick_end=4,
            red_presence_after={"host_a":False},
        )
        self.assertEqual(
            second.incident_active_ticks
            + third.incident_active_ticks
            + fourth.incident_active_ticks,
            3,
        )
        completed=bookkeeper.completed_events
        self.assertEqual(len(completed),1)
        self.assertEqual(completed[0].t_compromise,1)
        self.assertEqual(completed[0].t_normal,4)
        self.assertEqual(completed[0].attack_eradication_time,3)
        self.assertEqual(
            bookkeeper.total_incident_active_ticks,
            completed[0].attack_eradication_time,
        )

    def test_config_has_no_old_cost_delay_or_extra_evidence(self):
        posterior=self.cfg["posterior_observation"]
        self.assertEqual(posterior["components"],[
            "current_state","candidate_plan","llm_prior_preference","predicted_value","predictive_uncertainty"
        ])
        self.assertFalse(posterior["action_cost_evidence"])
        self.assertFalse(posterior["checkpoint_coverage_evidence"])
        self.assertFalse(posterior["delay_evidence"])
        self.assertFalse(posterior["early_warning_evidence"])
        reward=self.cfg["shared_contract"]["response_objective"]
        self.assertFalse(reward["early_warning_lead_reward"])
        self.assertFalse(reward["fixed_action_cost"])
        self.assertEqual(reward["reward_protocol"],"final_paper_20260917_v1")
        self.assertEqual(reward["delay_term"],"sum_active_incident_compromise_age")
        self.assertEqual(reward["failure_scope"],"all_tracked_host_local_work_failures")

    def test_legacy_modules_explicitly_excluded(self):
        legacy=self.cfg["legacy_isolation"]
        self.assertTrue(all(value is False for value in legacy.values()))

    def test_ppo_pretrain_seed_boundary_and_shared_gamma(self):
        ppo=self.cfg["ppo_pretrain_contract"]
        self.assertEqual(ppo["train_seeds"],list(range(1000,1032)))
        self.assertEqual(ppo["validation_seeds"],list(range(2000,2008)))
        self.assertTrue(ppo["calibration_test_forbidden"])
        self.assertEqual(ppo["gamma_tick"],0.99)
        self.assertTrue(ppo["duration_aware_discount"])
        self.assertEqual(ppo["rollout_length"],128)
        self.assertEqual(ppo["update_epochs"],5)
        self.assertEqual(ppo["clip_epsilon"],0.2)
        self.assertEqual(ppo["gae_lambda"],0.95)
        self.assertEqual(ppo["entropy_coefficient"],0.01)


if __name__ == "__main__":
    unittest.main()
