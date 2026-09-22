import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "formal_experiments" / "ours" / "lwm_runtime.py"
COLLECTOR = ROOT / "formal_experiments" / "evaluation" / "run_b4_provisional_ppo.py"
AUDIT_ALL = ROOT / "formal_experiments" / "evaluation" / "audit_b0_2_all_models.py"
DOC = ROOT / "docs" / "gate-b-B4-provisional-ppo.md"
B02_DOC = ROOT / "docs" / "gate-b-B0.2-policy-ood.md"


class TestGateB4PreProvisionalContract(unittest.TestCase):
    def test_runtime_moves_cpu_snapshot_to_actual_policy_device(self):
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("def _policy_device", source)
        self.assertIn("prepared.candidate_features.to(policy_device)", source)
        self.assertIn("features.to(policy_device)", source)
        self.assertIn("features.detach().cpu().clone()", source)

    def test_collector_is_train_cache_only_and_single_model_cli(self):
        source = COLLECTOR.read_text(encoding="utf-8")
        self.assertIn('split="train"', source)
        self.assertIn('parser.add_argument("--model-alias", required=True)', source)
        self.assertNotIn("--all-models", source)
        self.assertNotIn("range(4000", source)
        self.assertNotIn("range(3000", source)
        self.assertNotIn("range(2000", source)

    def test_collector_enforces_shared_canonical_action(self):
        source = COLLECTOR.read_text(encoding="utf-8")
        self.assertIn("canonicalize_requested_action", source)
        self.assertIn("EXECUTED_FAMILY_TO_ACTION_ID", source)
        self.assertIn("model-space canonical action differs from executed family", source)

    def test_all_model_gate_cannot_be_overridden_by_pooled_union(self):
        source = AUDIT_ALL.read_text(encoding="utf-8")
        self.assertIn('if any(status == "FAIL" for status in statuses)', source)
        self.assertIn('elif any(status == "COVERAGE_INCOMPLETE" for status in statuses)', source)
        self.assertIn('elif all(status == "PASS" for status in statuses)', source)
        self.assertIn('pooled["affects_gate"] = False', source)
        self.assertIn('"shared_world_model_requires_every_variant_pass": True', source)

    def test_tail_diagnostic_is_frozen_at_25_percent_and_non_gating(self):
        source = AUDIT_ALL.read_text(encoding="utf-8")
        self.assertIn("TAIL_FRACTION = 0.25", source)
        self.assertIn('"diagnostic_only": True', source)
        self.assertIn('"affects_gate": False', source)

    def test_provisional_stages_are_frozen_before_live_results(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("2,000 -> 5,000 -> 10,000 -> 20,000", text)
        self.assertIn("max 20,000 real decision transitions / model variant", text)
        self.assertIn("fresh policy initialization", text)
        self.assertIn("formal_result_eligible=false", text)

    def test_b02_requires_each_llm_variant_to_pass(self):
        text = B02_DOC.read_text(encoding="utf-8")
        self.assertIn("PASS iff H PASS AND M PASS AND L PASS", text)
        self.assertIn("Pooled union 结果只做 diagnostic", text)
        self.assertIn("last 25% probe transitions / model", text)


if __name__ == "__main__":
    unittest.main()
