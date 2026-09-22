import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from formal_experiments.evaluation.run_gate_b_b2_live_prior_smoke import (
    build_report,
    select_smoke_state,
)
from formal_experiments.ours.ofox_prior_client import (
    OFOX_BASE_URL,
    OFOXPriorLiveClient,
    api_key_source,
    build_openai_http_client,
    classify_live_exception,
    masked_detected_proxies,
    ofox_network_mode,
    prior_response_schema,
    require_api_key_env,
    response_format_json_schema,
)
from formal_experiments.ours.llm_prior_v2 import FORMAL_K_CANDIDATES
from shared.formal_state import FORMAL_STATE_DIM


class FakeCompletions:
    def __init__(self, text):
        self.text=text
        self.calls=[]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message=SimpleNamespace(content=self.text)
        choice=SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeChat:
    def __init__(self, text):
        self.completions=FakeCompletions(text)


class FakeClient:
    def __init__(self, text):
        self.chat=FakeChat(text)


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

    def test_key_source_and_missing_guard(self):
        self.assertEqual(api_key_source({"OFOX_API_KEY":"x"}),"OFOX_API_KEY")
        self.assertIsNone(api_key_source({}))
        with self.assertRaises(RuntimeError):
            require_api_key_env({})

    def test_network_mode_defaults_and_mutual_exclusion(self):
        self.assertEqual(ofox_network_mode({}),"sdk_environment_proxy")
        self.assertEqual(ofox_network_mode({"OFOX_DISABLE_ENV_PROXY":"1"}),"direct_no_env_proxy")
        self.assertEqual(ofox_network_mode({"OFOX_PROXY_URL":"http://127.0.0.1:7897"}),"explicit_proxy")
        with self.assertRaises(ValueError):
            ofox_network_mode({
                "OFOX_PROXY_URL":"http://127.0.0.1:7897",
                "OFOX_DISABLE_ENV_PROXY":"1",
            })

    def test_explicit_network_modes_construct_httpx_clients(self):
        direct=build_openai_http_client({"OFOX_DISABLE_ENV_PROXY":"1"})
        explicit=build_openai_http_client({"OFOX_PROXY_URL":"http://127.0.0.1:7897"})
        try:
            self.assertIsNotNone(direct)
            self.assertIsNotNone(explicit)
        finally:
            direct.close()
            explicit.close()

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

    def test_ofox_json_schema_wrapper_matches_openai_compatible_contract(self):
        fmt=response_format_json_schema()
        self.assertEqual(fmt["type"],"json_schema")
        self.assertEqual(fmt["json_schema"]["name"],"lwm_rl_candidate_plans")
        self.assertEqual(fmt["json_schema"]["schema"],prior_response_schema())

    def test_fake_live_request_uses_chat_completions_gpt56_and_json_schema(self):
        fake=FakeClient(valid_json())
        with patch.dict(os.environ,{"OFOX_API_KEY":"secret-test-key"},clear=True):
            client=OFOXPriorLiveClient(client=fake,require_env_key=True)
            result=client.generate(np.zeros(FORMAL_STATE_DIM,dtype=np.float32),agent_name="blue_agent_0")
        self.assertTrue(result.api_call_succeeded)
        self.assertEqual(result.key_source,"OFOX_API_KEY")
        self.assertEqual(result.network_mode,"sdk_environment_proxy")
        self.assertEqual(result.base_url,OFOX_BASE_URL)
        self.assertEqual(result.model,"openai/gpt-5.6-sol")
        self.assertEqual(result.prior.plans.shape,(6,4))
        self.assertEqual(len(fake.chat.completions.calls),1)
        call=fake.chat.completions.calls[0]
        self.assertEqual(call["model"],"openai/gpt-5.6-sol")
        self.assertEqual(call["temperature"],0.2)
        self.assertEqual(call["response_format"]["type"],"json_schema")
        self.assertNotIn("secret-test-key",repr(call))

    def test_client_never_accepts_api_key_argument(self):
        import inspect
        sig=inspect.signature(OFOXPriorLiveClient.__init__)
        self.assertNotIn("api_key",sig.parameters)

    def test_empty_output_is_hard_failure_not_silent_fallback(self):
        fake=FakeClient("")
        client=OFOXPriorLiveClient(client=fake,require_env_key=False)
        with self.assertRaises(RuntimeError):
            client.generate(np.zeros(27,dtype=np.float32))

    def test_live_result_hashes_are_present_but_key_value_is_not_exposed(self):
        fake=FakeClient(valid_json())
        with patch.dict(os.environ,{"OFOX_API_KEY":"secret-test-key"},clear=True):
            result=OFOXPriorLiveClient(client=fake).generate(np.zeros(27,dtype=np.float32))
        self.assertEqual(len(result.prompt_sha256),64)
        self.assertEqual(len(result.response_sha256),64)
        self.assertNotIn("secret-test-key",repr(result))

    def test_smoke_report_omits_raw_prompt_raw_response_and_key_value(self):
        fake=FakeClient(valid_json())
        with patch.dict(os.environ,{"OFOX_API_KEY":"secret-test-key"},clear=True):
            result=OFOXPriorLiveClient(client=fake).generate(np.zeros(27,dtype=np.float32),agent_name="blue_agent_0")
        report=build_report(result,agent_name="blue_agent_0",episode_seed=3000)
        self.assertTrue(report["pass"])
        self.assertEqual(report["provider"],"ofox_openai_compatible")
        self.assertEqual(report["base_url"],"https://api.ofox.ai/v1")
        self.assertEqual(report["model"],"openai/gpt-5.6-sol")
        self.assertFalse(report["api_key_value_recorded"])
        self.assertFalse(report["raw_prompt_recorded"])
        self.assertFalse(report["raw_response_recorded"])
        dumped=json.dumps(report)
        self.assertNotIn("secret-test-key",dumped)
        self.assertNotIn(result.raw_response_text,dumped)

    def test_smoke_report_allows_parser_fallback_but_requires_live_api_success(self):
        fake=FakeClient(json.dumps({"candidates":[]}))
        client=OFOXPriorLiveClient(client=fake,require_env_key=False)
        result=client.generate(np.zeros(27,dtype=np.float32))
        report=build_report(result,agent_name="blue_agent_0",episode_seed=3000)
        self.assertTrue(report["api_call_succeeded"])
        self.assertTrue(report["pass"])
        self.assertEqual(report["fallback_count"],FORMAL_K_CANDIDATES)

    def test_live_exception_classification_separates_failure_classes(self):
        self.assertEqual(classify_live_exception(RuntimeError("429 insufficient_quota")),"quota_blocked")
        self.assertEqual(classify_live_exception(RuntimeError("401 authentication error API key")),"auth_blocked")
        self.assertEqual(classify_live_exception(RuntimeError("SSL ConnectError via proxy")),"network_blocked")
        self.assertEqual(classify_live_exception(RuntimeError("404 model not found")),"model_blocked")

    def test_masked_proxy_diagnostics_remove_userinfo_and_ignore_ofox_flags(self):
        out=masked_detected_proxies({
            "HTTPS_PROXY":"http://user:secret@127.0.0.1:7897",
            "OFOX_DISABLE_ENV_PROXY":"1",
        })
        self.assertEqual(out["https"],"http://127.0.0.1:7897")
        self.assertNotIn("secret",repr(out))
        self.assertNotIn("ofox_disable_env",out)

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
