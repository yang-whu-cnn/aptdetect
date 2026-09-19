import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from formal_experiments.evaluation.generate_paper_figures import (
    ELIGIBILITY_SCHEMA, ELIGIBILITY_VALIDATOR, FULL_REWARD_SEMANTICS,
    TABLE_ROWS, discover_formal_runs, generate,
)


def write_json(path, value): path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
def write_jsonl(path, rows): path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows), encoding="utf-8")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class TestFormalPaperFigures(unittest.TestCase):
    def make_repeat(self, root, table_id, row_id, repeat, *,
                    reward_semantics=FULL_REWARD_SEMANTICS, with_curve=True,
                    with_latency=True, directory_suffix=""):
        path = root/table_id/row_id/f"repeat_{repeat}{directory_suffix}"; path.mkdir(parents=True)
        if table_id == "table2": reward_mode, component = "full_reward", row_id
        elif table_id == "table3": reward_mode, component = row_id, "lwm_rl"
        else: reward_mode, component = (("full_reward", "lwm_rl") if row_id == "lwm_rl" else ("method_specific", row_id))
        manifest = {"method": row_id, "repeat_index": repeat, "run_mode": "formal",
            "formal_result_eligible": True, "eligibility": "PASS", "table_id": table_id,
            "row_id": row_id, "reward_mode": reward_mode, "component_variant": component,
            "reward_semantics": reward_semantics}
        if with_curve:
            write_jsonl(path/"training_curve.jsonl", [{"environment_steps": 0, "training_objective_reward": repeat-1.0}, {"environment_steps": 100, "training_objective_reward": repeat+1.0}])
            manifest["figure_artifacts"] = {"training_curve": {"path": "training_curve.jsonl", "sha256": sha(path/"training_curve.jsonl")}}
        write_json(path/"manifest.json", manifest)
        episode = {"episode_end_tick": 500, "incidents": [{"t_compromise": 10, "t_recovered": 20+repeat}, {"t_compromise": 450, "t_recovered": None}]}
        if with_latency: episode["decision_latency_ms"] = [float(repeat), float(repeat+1)]
        write_jsonl(path/"episodes.jsonl", [episode]); write_json(path/"metrics.json", {"synthetic_fixture": True})
        write_json(path/"eligibility_report.json", {"schema": ELIGIBILITY_SCHEMA,
            "validator": ELIGIBILITY_VALIDATOR, "status": "PASS", "passed": True,
            "input_sha256": {name: sha(path/name) for name in ("manifest.json", "episodes.jsonl", "metrics.json")}})
        return path

    @staticmethod
    def validation(path, **_kwargs):
        name = Path(path).name; repeat = int(name.removeprefix("repeat_").split("-")[0])
        manifest = json.loads((Path(path)/"manifest.json").read_text())
        offset = list(TABLE_ROWS[manifest["table_id"]]).index(manifest["row_id"])
        return {"passed": True, "errors": [], "metrics": {"cc4_official_reward": 100.+offset+repeat,
            "operation_failure_penalty": 10.+repeat, "recovery_precision": .5+repeat/100,
            "recovery_time_censored_mean": 20.+repeat}}

    def make_table(self, root, table_id, **kwargs):
        for row_id in TABLE_ROWS[table_id]:
            for repeat in range(1, 6): self.make_repeat(root, table_id, row_id, repeat, **kwargs)

    def test_eligible_fixture_generates_all_five_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; out=Path(tmp)/"one"; out2=Path(tmp)/"two"
            for table_id in TABLE_ROWS: self.make_table(root, table_id)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory", side_effect=self.validation):
                first=generate(root,out); second=generate(root,out2)
            self.assertEqual(first["status"], "PASS"); self.assertEqual(first["accepted_repeat_count"],70)
            for name, figure in first["figures"].items():
                self.assertEqual(figure["status"],"PASS",name)
                self.assertEqual(figure["artifacts"]["svg_sha256"],second["figures"][name]["artifacts"]["svg_sha256"])
                self.assertEqual(figure["artifacts"]["png_sha256"],second["figures"][name]["artifacts"]["png_sha256"])

    def test_manifest_self_report_and_wrong_eligibility_filename_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; path=self.make_repeat(root,"table1","lwm_rl",1)
            payload=json.loads((path/"eligibility_report.json").read_text()); (path/"eligibility_report.json").unlink(); write_json(path/"eligibility.json",payload)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=AssertionError("must reject early")):
                accepted,rejected=discover_formal_runs(root)
            self.assertEqual(accepted,[]); self.assertIn("eligibility_report.json",rejected[0]["reason"])

    def test_eligibility_report_input_binding_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; path=self.make_repeat(root,"table1","lwm_rl",1)
            with (path/"episodes.jsonl").open("a",encoding="utf-8") as h: h.write("{}\n")
            accepted,rejected=discover_formal_runs(root)
            self.assertEqual(accepted,[]); self.assertIn("binding mismatch",rejected[0]["reason"])

    def test_same_lwm_name_is_isolated_by_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"
            for repeat in (1,2,3): self.make_repeat(root,"table1","lwm_rl",repeat)
            for repeat in (4,5): self.make_repeat(root,"table2","lwm_rl",repeat)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation): result=generate(root,Path(tmp)/"out")
            self.assertEqual(result["status"],"BLOCKED")
            issues=" ".join(result["row_completeness_issues"]); self.assertIn("table1/lwm_rl",issues); self.assertIn("table2/lwm_rl",issues)

    def test_duplicate_repeat_blocks_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; self.make_table(root,"table1"); self.make_repeat(root,"table1","lwm_rl",5,directory_suffix="-duplicate")
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation): result=generate(root,Path(tmp)/"out")
            self.assertEqual(result["figures"]["table1_method_comparison"]["status"],"BLOCKED")
            self.assertTrue(any("table1/lwm_rl" in x for x in result["row_completeness_issues"]))

    def test_table1_only_is_partial_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; self.make_table(root,"table1")
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation): result=generate(root,Path(tmp)/"out")
            self.assertEqual(result["status"],"PARTIAL")
            self.assertEqual([result["figures"][x]["status"] for x in ("table1_method_comparison","recovery_survival","efficiency_tradeoff")],["PASS"]*3)
            self.assertEqual(result["figures"]["training_learning_curve"]["status"],"BLOCKED")
            self.assertEqual(result["figures"]["component_reward_ablation"]["status"],"BLOCKED")

    def test_different_reward_semantics_cannot_enter_learning_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; self.make_table(root,"table2"); path=root/"table2"/"wm_rl"/"repeat_3"
            manifest=json.loads((path/"manifest.json").read_text()); manifest["reward_semantics"]="different_reward"; write_json(path/"manifest.json",manifest)
            report=json.loads((path/"eligibility_report.json").read_text()); report["input_sha256"]["manifest.json"]=sha(path/"manifest.json"); write_json(path/"eligibility_report.json",report)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation): result=generate(root,Path(tmp)/"out")
            self.assertEqual(result["figures"]["training_learning_curve"]["status"],"BLOCKED")
            self.assertTrue(any("Full-Reward semantics" in x["reason"] for x in result["rejected_inputs"]))


if __name__ == "__main__": unittest.main()
