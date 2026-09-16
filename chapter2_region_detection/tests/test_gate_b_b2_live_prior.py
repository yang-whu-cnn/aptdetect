import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from formal_experiments.evaluation.run_gate_b_b2_live_prior_smoke import (
    build_report,
    select_smoke_state,
)
from formal_experiments.ours.gemini_prior_client import (
    GeminiPriorLiveClient,
    api_key_source,
    gemini_network_mode,
    build_genai_http_options,
    masked_detected_proxies,
    prior_response_schema,
    require_api_key_env,
)
from formal_experiments.ours.llm_prior_v2 import FORMAL_K_CANDIDATES
from formal_experiments.training.response_reward_predictor import ResponseRewardPredictor
from shared.formal_state import FORMAL_STATE_DIM


class FakeInteractions:
    def __init__(self, text):
        self.text=text
        self.calls=[]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.text)


class FakeClient:
    def __init__(self, text):
        self.interactions=FakeInteractions(text)


def valid_json():
    return json.dumps({
        "candidates":[
            {"actions":["no_op","analyse","remove","restore"],"prior_score":0.9,"reason":"a"},
            {"actions":["analyse","analyse","remove","restore"],"prior_score":0.8,"reason":"b"},
            {"actions":["remove","analyse","remove","restore"],"prior_score":0.7,"reason":"c"},
            {"actions":["restore","analyse","remove","restore"],"prior_score":0.6,"reason":"d"},
            {"actions":["no_op","no_op","no_op","no_op"],"prior_score":0.5,"reason":"e"},
            {"actions":["restore","restore","restore","restore"],"prior_score":0.4,"reason":"f"},
        ]
    })


class TestGateBB2LivePrior(unittest.TestCase):

    def test_key_source_precedence_and_missing_guard(self):
        self.assertEqual(api_key_source({"GEMINI_API_KEY":"g"}),"GEMINI_API_KEY")
        self.assertEqual(api_key_source({"GOOGLE_API_KEY":"x","GEMINI_API_KEY":"g"}),"GOOGLE_API_KEY")
        self.assertIsNone(api_key_source({}))
        with self.assertRaises(RuntimeError):
            require_api_key_env({})

    def test_network_mode_defaults_and_mutual_exclusion(self):
        self.assertEqual(gemini_network_mode({}),"sdk_environment_proxy")
        self.assertEqual(gemini_network_mode({"GEMINI_DISABLE_ENV_PROXY":"1"}),"direct_no_env_proxy")
        self.assertEqual(gemini_network_mode({"GEMINI_PROXY_URL":"http://127.0.0.1:7890"}),"explicit_proxy")
        with self.assertRaises(ValueError):
            gemini_network_mode({
                "GEMINI_PROXY_URL":"http://127.0.0.1:7890",
                "GEMINI_DISABLE_ENV_PROXY":"1",
            })

    def test_http_options_support_direct_and_explicit_proxy_modes(self):
        direct=build_genai_http_options({"GEMINI_DISABLE_ENV_PROXY":"1"})
        self.assertFalse(direct.client_args["trust_env"])
        explicit=build_genai_http_options({"GEMINI_PROXY_URL":"http://127.0.0.1:7890"})
        self.assertEqual(explicit.client_args["proxy"],"http://127.0.0.1:7890")
        self.assertFalse(explicit.client_args["trust_env"])

    def test_masked_proxy_diagnostics_remove_userinfo(self):
        out=masked_detected_proxies({
            "HTTPS_PROXY":"http://user:secret@127.0.0.1:7890",
        })
        self.assertEqual(out["https"],"http://127.0.0.1:7890")
        self.assertNotIn("secret",repr(out))
    def test_schema_locks_exact_k_h_actions_and_score_range(self):
        schema=prior_response_schema()
        candidates=schema["properties"]["candidates"]
        self.assertEqual(candidates["minItems"],6)
        self.assertEqual(candidates["maxItems"],6)
        item=candidates["items"]
        actions=item["properties"]["actions"]
        self.assertEqual(actions["minItems"],4)
        self.assertEqual(actions["maxItems"],4)
        self.assertEqual(actions["items"]["enum"],["no_op","analyse","remove","restore"])
        score=item["properties"]["prior_score"]
        self.assertEqual(score["minimum"],0.0)
        self.assertEqual(score["maximum"],1.0)

    def test_fake_live_request_uses_interactions_structured_output_and_store_false(self):
        fake=FakeClient(valid_json())
        with patch.dict(os.environ,{"GEMINI_API_KEY":"secret-test-key"},clear=True):
            client=GeminiPriorLiveClient(client=fake,require_env_key=True)
            result=client.generate(np.zeros(FORMAL_STATE_DIM,dtype=np.float32),agent_name="blue_agent_0")
        self.assertTrue(result.api_call_succeeded)
        self.assertEqual(result.key_source,"GEMINI_API_KEY")
        self.assertEqual(result.network_mode,"sdk_environment_proxy")
        self.assertEqual(result.prior.plans.shape,(6,4))
        self.assertEqual(len(fake.interactions.calls),1)
        call=fake.interactions.calls[0]
        self.assertEqual(call["model"],"gemini-3.1-pro-preview")
        self.assertEqual(call["generation_config"]["temperature"],0.2)
        self.assertFalse(call["store"])
        self.assertEqual(call["response_format"]["mime_type"],"application/json")
        self.assertNotIn("secret-test-key",repr(call))

    def test_client_never_accepts_api_key_argument(self):
        import inspect
        sig=inspect.signature(GeminiPriorLiveClient.__init__)
        self.assertNotIn("api_key",sig.parameters)

    def test_empty_output_is_hard_failure_not_silent_fallback(self):
        fake=FakeClient("")
        client=GeminiPriorLiveClient(client=fake,require_env_key=False)
        with self.assertRaises(RuntimeError):
            client.generate(np.zeros(27,dtype=np.float32))

    def test_live_result_hashes_are_present_but_key_value_is_not_exposed(self):
        fake=FakeClient(valid_json())
        with patch.dict(os.environ,{"GEMINI_API_KEY":"secret-test-key"},clear=True):
            client=GeminiPriorLiveClient(client=fake,require_env_key=True)
            result=client.generate(np.zeros(27,dtype=np.float32))
        self.assertEqual(len(result.prompt_sha256),64)
        self.assertEqual(len(result.response_sha256),64)
        self.assertNotIn("secret-test-key",repr(result))

    def test_smoke_report_omits_raw_prompt_raw_response_and_key_value(self):
        fake=FakeClient(valid_json())
        with patch.dict(os.environ,{"GEMINI_API_KEY":"secret-test-key"},clear=True):
            result=GeminiPriorLiveClient(client=fake).generate(np.zeros(27,dtype=np.float32),agent_name="blue_agent_0")
        report=build_report(result,agent_name="blue_agent_0",episode_seed=3000)
        self.assertTrue(report["pass"])
        self.assertFalse(report["api_key_value_recorded"])
        self.assertFalse(report["raw_prompt_recorded"])
        self.assertFalse(report["raw_response_recorded"])
        self.assertIn(report["network_mode"],{"sdk_environment_proxy","direct_no_env_proxy","explicit_proxy"})
        dumped=json.dumps(report)
        self.assertNotIn("secret-test-key",dumped)
        self.assertNotIn(result.raw_response_text,dumped)

    def test_smoke_report_allows_parser_fallback_but_requires_live_api_success(self):
        fake=FakeClient(json.dumps({"candidates":[]}))
        client=GeminiPriorLiveClient(client=fake,require_env_key=False)
        result=client.generate(np.zeros(27,dtype=np.float32))
        report=build_report(result,agent_name="blue_agent_0",episode_seed=3000)
        self.assertTrue(report["api_call_succeeded"])
        self.assertTrue(report["pass"])
        self.assertEqual(report["fallback_count"],FORMAL_K_CANDIDATES)

    def test_select_smoke_state_is_deterministic_and_planner_visible(self):
        rows=[
            {"warmup":SimpleNamespace(agent_name="blue_agent_0",episode_seed=3000,state=np.full(27,2,dtype=np.float32)),"global_tick_start":2,"decision_index":1},
            {"warmup":SimpleNamespace(agent_name="blue_agent_0",episode_seed=3000,state=np.full(27,1,dtype=np.float32)),"global_tick_start":1,"decision_index":2},
            {"warmup":SimpleNamespace(agent_name="blue_agent_1",episode_seed=3000,state=np.full(27,3,dtype=np.float32)),"global_tick_start":0,"decision_index":0},
        ]
        state=select_smoke_state(rows,agent_name="blue_agent_0",episode_seed=3000)
        np.testing.assert_array_equal(state,np.full(27,1,dtype=np.float32))

    def test_select_smoke_state_rejects_missing_agent_seed(self):
        with self.assertRaises(ValueError):
            select_smoke_state([],agent_name="blue_agent_0",episode_seed=3000)


if __name__ == "__main__":
    unittest.main()
