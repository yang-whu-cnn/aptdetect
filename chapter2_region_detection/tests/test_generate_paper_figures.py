import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from formal_experiments.evaluation.generate_paper_figures import (
    ELIGIBILITY_SCHEMA, ELIGIBILITY_VALIDATOR, FULL_REWARD_PHYSICAL_REJECTION,
    FAIL_ONLY_CHECKPOINT_SHA256, FAIL_ONLY_MANIFEST_SHA256, FULL_REWARD_SEMANTICS,
    OFOX_ENTRY_SET_SHA256, OFOX_LOGICAL_MANIFEST_SHA256,
    ROW_LABELS, TABLE_ROWS, discover_formal_runs, generate,
)
from formal_experiments.evaluation.aggregate_tables import generate_table_artifacts
from formal_experiments.evaluation.metrics_v3 import aggregate_repeat
from formal_experiments.evaluation.validate_formal_run import validate_run_directory
from tests.test_formal_manifest import episode as production_episode
from tests.test_formal_manifest import manifest as production_manifest


def write_json(path, value): path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
def write_jsonl(path, rows): path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows), encoding="utf-8")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class TestFormalPaperFigures(unittest.TestCase):
    def make_production_curve_repeat(self, root):
        path=root/"table2"/"lwm_rl"/"repeat_1"; path.mkdir(parents=True)
        episodes=[production_episode(seed) for seed in range(4000,4100)]
        write_jsonl(path/"episodes.jsonl",episodes)
        write_json(path/"metrics.json",aggregate_repeat(episodes).to_dict())
        for filename in ("config.resolved.yaml","decisions.jsonl.zst","stdout.log","policy_spec.json"):
            (path/filename).write_bytes(b"production-test")
        write_jsonl(path/"training_curve.jsonl",[
            {"environment_steps":0,"training_objective_reward":0.0},
            {"environment_steps":100,"training_objective_reward":1.0},
        ])
        value=production_manifest()
        value.update({"method":"LWM-RL","method_slug":"lwm_rl_cc4","table_id":"table2",
            "row_id":"lwm_rl","reward_mode":"full_reward","component_variant":"lwm_rl",
            "reward_semantics":FULL_REWARD_SEMANTICS,"canonical_run_id":"lwm_full"})
        bound=("config.resolved.yaml","policy_spec.json","episodes.jsonl",
               "decisions.jsonl.zst","metrics.json","training_curve.jsonl")
        value["artifact_sha256"]={name:sha(path/name) for name in bound}
        value["config_sha256"]=value["artifact_sha256"]["config.resolved.yaml"]
        value["model_sha256"]=value["artifact_sha256"]["policy_spec.json"]
        curve_sha=value["artifact_sha256"]["training_curve.jsonl"]
        value["figure_artifacts"]={"training_curve":{
            "schema":"cc4_v3_training_curve_v1","path":"training_curve.jsonl",
            "sha256":curve_sha,"split":"train","x_field":"environment_steps",
            "y_field":"training_objective_reward","record_count":2}}
        write_json(path/"manifest.json",value)
        result=validate_run_directory(path,formal=True)
        self.assertTrue(result["passed"],result["errors"])
        return path

    def make_repeat(self, root, table_id, row_id, repeat, *,
                    reward_semantics=None, with_curve=True,
                    with_latency=True, directory_suffix=""):
        path = root/table_id/row_id/f"repeat_{repeat}{directory_suffix}"; path.mkdir(parents=True)
        table1_identity = {
            "uamcts_cc4": ("full_reward", "cc4_v3_full_reward"),
            "rsmbrl_cc4": ("full_reward", "cc4_v3_full_reward"),
            "carl_cc4": ("caics_reward", "standard_caics_cc4_mapping"),
            "dca_cc4": ("not_applicable", "fixed_response_mapping_no_learning_reward"),
            "priorrl_ppo_cc4": ("full_reward", "cc4_v3_full_reward"),
            "terla_a4": ("terla_cyber_reward", "negative_red_sessions_plus_service_unreliability_ot"),
        }
        if table_id == "table2": reward_mode, component = "full_reward", row_id
        elif table_id == "table3": reward_mode, component = row_id, "lwm_rl"
        else: reward_mode, component = table1_identity.get(row_id, ("full_reward", "lwm_rl"))[0], row_id
        if reward_semantics is None:
            if table_id == "table1": reward_semantics = table1_identity.get(row_id, (None, FULL_REWARD_SEMANTICS))[1]
            else: reward_semantics = f"cc4_v3_{row_id}" if table_id == "table3" else FULL_REWARD_SEMANTICS
        manifest = {"method": row_id, "method_slug": row_id, "repeat_index": repeat,
            "training_seed": 51000+repeat, "run_mode": "formal",
            "formal_result_eligible": True, "eligibility": "PASS", "table_id": table_id,
            "row_id": row_id, "reward_mode": reward_mode, "component_variant": component,
            "reward_semantics": reward_semantics, "artifact_sha256": {}}
        if (table_id,row_id) == ("table2","lwm_rl"):
            manifest["canonical_run_id"] = "lwm_full"
        if (table_id,row_id) == ("table3","fail_only"):
            manifest["fail_only_training_contract"] = {
                "loss_name": "smooth_l1", "beta": 1.0, "legacy_mse": "diagnostic_only",
                "gate_pass": True, "train_seed_count": 32, "validation_seed_count": 8,
                "train_validation_overlap": 0, "test_leak_count": 0,
                "checkpoint_sha256": FAIL_ONLY_CHECKPOINT_SHA256,
                "manifest_sha256": FAIL_ONLY_MANIFEST_SHA256,
            }
            manifest["ofox_cache_audit"] = {
                "provider": "ofox", "provider_calls": 6, "generated": 6,
                "verified_entries": 6, "failed": 0, "d27_exact_count": 6,
                "split": "train_only", "coverage_pass": True,
                "online_calls_allowed": False, "entry_set_sha256": OFOX_ENTRY_SET_SHA256,
                "logical_manifest_sha256": OFOX_LOGICAL_MANIFEST_SHA256,
            }
        if with_curve:
            write_jsonl(path/"training_curve.jsonl", [{"environment_steps": 0, "training_objective_reward": repeat-1.0}, {"environment_steps": 100, "training_objective_reward": repeat+1.0}])
            curve_sha = sha(path/"training_curve.jsonl")
            manifest["figure_artifacts"] = {"training_curve": {
                "schema": "cc4_v3_training_curve_v1", "path": "training_curve.jsonl",
                "sha256": curve_sha, "split": "train", "x_field": "environment_steps",
                "y_field": "training_objective_reward", "record_count": 2}}
            manifest["artifact_sha256"]["training_curve.jsonl"] = curve_sha
        write_json(path/"manifest.json", manifest)
        episode = {"episode_end_tick": 500, "incidents": [{"t_compromise": 10, "t_recovered": 20+repeat}, {"t_compromise": 450, "t_recovered": None}]}
        if with_latency: episode["decision_latency_ms"] = [float(repeat), float(repeat+1)]
        write_jsonl(path/"episodes.jsonl", [episode]); write_json(path/"metrics.json", {"synthetic_fixture": True})
        eligibility_names = ["manifest.json", "episodes.jsonl", "metrics.json"]
        if with_curve: eligibility_names.append("training_curve.jsonl")
        write_json(path/"eligibility_report.json", {"schema": ELIGIBILITY_SCHEMA,
            "validator": ELIGIBILITY_VALIDATOR, "status": "PASS", "passed": True,
            "input_sha256": {name: sha(path/name) for name in eligibility_names}})
        return path

    @staticmethod
    def validation(path, **_kwargs):
        name = Path(path).name; repeat = int(name.removeprefix("repeat_").split("-")[0])
        manifest = json.loads((Path(path)/"manifest.json").read_text())
        offset = list(TABLE_ROWS[manifest["table_id"]]).index(manifest["row_id"])
        return {"passed": True, "errors": [], "metrics": {"cc4_official_reward": 100.+offset+repeat,
            "operation_failure_penalty": 10.+repeat, "recovery_precision": .5+repeat/100,
            "recovery_time_censored_mean": 20.+repeat,
            "recovery_time_completed_only_mean": 10.+repeat,
            "recovery_unrecovered_rate": .2+repeat/100}}

    def make_table(self, root, table_id, **kwargs):
        for row_id in TABLE_ROWS[table_id]:
            if (table_id, row_id) in (("table1", "lwm_rl"), ("table3", "full_reward")):
                continue
            for repeat in range(1, 6): self.make_repeat(root, table_id, row_id, repeat, **kwargs)

    def test_eligible_fixture_generates_all_five_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; out=Path(tmp)/"one"; out2=Path(tmp)/"two"
            for table_id in TABLE_ROWS: self.make_table(root, table_id)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory", side_effect=self.validation):
                first=generate(root,out); second=generate(root,out2)
            self.assertEqual(first["status"], "PASS"); self.assertEqual(first["accepted_repeat_count"],60)
            self.assertEqual(first["logical_repeat_count"],70)
            self.assertIn("table1/lwm_rl",first["complete_rows"])
            self.assertIn("table3/full_reward",first["complete_rows"])
            for name, figure in first["figures"].items():
                self.assertEqual(figure["status"],"PASS",name)
                self.assertEqual(figure["artifacts"]["svg_sha256"],second["figures"][name]["artifacts"]["svg_sha256"])
                self.assertEqual(figure["artifacts"]["png_sha256"],second["figures"][name]["artifacts"]["png_sha256"])

    def test_full_reward_physical_duplicate_blocks_alias_instead_of_selecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; out=Path(tmp)/"out"
            for table_id in TABLE_ROWS: self.make_table(root,table_id)
            for repeat in range(1,6): self.make_repeat(root,"table3","full_reward",repeat)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation):
                result=generate(root,out)
            self.assertEqual(result["status"],"PARTIAL")
            self.assertEqual(result["figures"]["component_reward_ablation"]["status"],"BLOCKED")
            self.assertTrue(any(FULL_REWARD_PHYSICAL_REJECTION in row["reason"] for row in result["rejected_inputs"]))
            self.assertIn(FULL_REWARD_PHYSICAL_REJECTION,result["row_completeness_issues"])

    def test_table1_lwm_physical_duplicate_blocks_shared_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; out=Path(tmp)/"out"
            for table_id in TABLE_ROWS: self.make_table(root,table_id)
            for repeat in range(1,6): self.make_repeat(root,"table1","lwm_rl",repeat)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation):
                result=generate(root,out)
            self.assertEqual(result["status"],"PARTIAL")
            self.assertTrue(any(FULL_REWARD_PHYSICAL_REJECTION in row["reason"] for row in result["rejected_inputs"]))
            self.assertIn(FULL_REWARD_PHYSICAL_REJECTION,result["row_completeness_issues"])

    def test_curve_grid_mismatch_rolls_back_the_atomic_five_figure_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; out=Path(tmp)/"out"
            for table_id in TABLE_ROWS: self.make_table(root,table_id)
            path=root/"table2"/"rl_only"/"repeat_1"
            write_jsonl(path/"training_curve.jsonl",[
                {"environment_steps":0,"training_objective_reward":0.0},
                {"environment_steps":50,"training_objective_reward":1.0},
            ])
            manifest=json.loads((path/"manifest.json").read_text())
            curve_sha=sha(path/"training_curve.jsonl")
            manifest["figure_artifacts"]["training_curve"]["sha256"]=curve_sha
            manifest["artifact_sha256"]["training_curve.jsonl"]=curve_sha
            write_json(path/"manifest.json",manifest)
            report=json.loads((path/"eligibility_report.json").read_text())
            report["input_sha256"]["manifest.json"]=sha(path/"manifest.json")
            report["input_sha256"]["training_curve.jsonl"]=curve_sha
            write_json(path/"eligibility_report.json",report)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation):
                result=generate(root,out)
            self.assertEqual(result["status"],"PARTIAL")
            self.assertTrue(all(row["status"]=="BLOCKED" for row in result["figures"].values()))
            for stem in result["figures"]:
                self.assertFalse((out/f"{stem}.svg").exists())
                self.assertFalse((out/f"{stem}.png").exists())

    def test_training_curve_requires_matching_general_artifact_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; path=self.make_repeat(root,"table2","lwm_rl",1)
            manifest=json.loads((path/"manifest.json").read_text())
            manifest["artifact_sha256"].pop("training_curve.jsonl")
            write_json(path/"manifest.json",manifest)
            report=json.loads((path/"eligibility_report.json").read_text())
            report["input_sha256"]["manifest.json"]=sha(path/"manifest.json")
            write_json(path/"eligibility_report.json",report)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation):
                accepted,rejected=discover_formal_runs(root)
            self.assertEqual(accepted,[])
            self.assertIn("same sha256 binding in artifact_sha256",rejected[0]["reason"])

    def test_production_manifest_triple_binds_training_curve_and_rejects_tampering(self):
        for mutation in ("file","figure_declaration","artifact_declaration","eligibility_declaration"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"formal"; path=self.make_production_curve_repeat(root)
                accepted,rejected=discover_formal_runs(root)
                self.assertEqual(len(accepted),1); self.assertEqual(rejected,[])
                if mutation == "file":
                    with (path/"training_curve.jsonl").open("a",encoding="utf-8") as handle:
                        handle.write("{}\n")
                elif mutation in ("figure_declaration","artifact_declaration"):
                    manifest=json.loads((path/"manifest.json").read_text())
                    if mutation == "figure_declaration":
                        manifest["figure_artifacts"]["training_curve"]["sha256"]="f"*64
                    else:
                        manifest["artifact_sha256"]["training_curve.jsonl"]="f"*64
                    write_json(path/"manifest.json",manifest)
                    report=json.loads((path/"eligibility_report.json").read_text())
                    report["input_sha256"]["manifest.json"]=sha(path/"manifest.json")
                    write_json(path/"eligibility_report.json",report)
                else:
                    report=json.loads((path/"eligibility_report.json").read_text())
                    report["input_sha256"]["training_curve.jsonl"]="f"*64
                    write_json(path/"eligibility_report.json",report)
                accepted,rejected=discover_formal_runs(root)
                self.assertEqual(accepted,[],mutation)
                self.assertEqual(len(rejected),1,mutation)

    def test_fail_only_requires_approved_loss_and_ofox_cache_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; path=self.make_repeat(root,"table3","fail_only",1)
            manifest=json.loads((path/"manifest.json").read_text())
            manifest["ofox_cache_audit"]["provider_calls"]=7
            write_json(path/"manifest.json",manifest)
            report=json.loads((path/"eligibility_report.json").read_text())
            report["input_sha256"]["manifest.json"]=sha(path/"manifest.json")
            write_json(path/"eligibility_report.json",report)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation):
                accepted,rejected=discover_formal_runs(root)
            self.assertEqual(accepted,[])
            self.assertIn("frozen OFOX cache contract mismatch",rejected[0]["reason"])

    def test_table_artifacts_include_alias_and_table_level_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; out=Path(tmp)/"tables"; out2=Path(tmp)/"tables2"
            for table_id in TABLE_ROWS: self.make_table(root,table_id)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation):
                report=generate_table_artifacts(root,out)
                report2=generate_table_artifacts(root,out2)
            self.assertEqual(report["status"],"PASS")
            self.assertEqual(report["aliases"]["table1/lwm_rl"]["source"],"table2/lwm_rl")
            self.assertEqual(report["aliases"]["table3/full_reward"]["source"],"table2/lwm_rl")
            self.assertEqual(report["physical_accepted_repeat_count"],60)
            self.assertEqual(report["logical_aggregated_repeat_count"],70)
            for name in ("table1.csv","table2.csv","table3.csv","tables.md","eligibility_report.json"):
                self.assertTrue((out/name).is_file(),name)
            self.assertEqual(report["output_sha256"],report2["output_sha256"])
            self.assertEqual(len((out/"table3.csv").read_text().splitlines()),4)

    def test_partial_table_audit_removes_stale_publication_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; out=Path(tmp)/"tables"; out.mkdir()
            self.make_table(root,"table1")
            for name in ("table1.csv","table2.csv","table3.csv","tables.md"):
                (out/name).write_text("stale",encoding="utf-8")
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation):
                report=generate_table_artifacts(root,out)
            self.assertEqual(report["status"],"PARTIAL")
            for name in ("table1.csv","table2.csv","table3.csv","tables.md"):
                self.assertFalse((out/name).exists(),name)
            self.assertTrue((out/"eligibility_report.json").is_file())

    def test_manifest_self_report_and_wrong_eligibility_filename_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; path=self.make_repeat(root,"table1","dca_cc4",1)
            payload=json.loads((path/"eligibility_report.json").read_text()); (path/"eligibility_report.json").unlink(); write_json(path/"eligibility.json",payload)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=AssertionError("must reject early")):
                accepted,rejected=discover_formal_runs(root)
            self.assertEqual(accepted,[]); self.assertIn("eligibility_report.json",rejected[0]["reason"])

    def test_eligibility_report_input_binding_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; path=self.make_repeat(root,"table1","dca_cc4",1)
            with (path/"episodes.jsonl").open("a",encoding="utf-8") as h: h.write("{}\n")
            accepted,rejected=discover_formal_runs(root)
            self.assertEqual(accepted,[]); self.assertIn("binding mismatch",rejected[0]["reason"])

    def test_table1_method_slug_must_match_row_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; path=self.make_repeat(root,"table1","dca_cc4",1)
            manifest=json.loads((path/"manifest.json").read_text()); manifest["method_slug"]="rsmbrl_cc4"; write_json(path/"manifest.json",manifest)
            report=json.loads((path/"eligibility_report.json").read_text()); report["input_sha256"]["manifest.json"]=sha(path/"manifest.json"); write_json(path/"eligibility_report.json",report)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation):
                accepted,rejected=discover_formal_runs(root)
            self.assertEqual(accepted,[]); self.assertIn("frozen method identity",rejected[0]["reason"])

    def test_adapted_table1_labels_match_taskbook(self):
        self.assertEqual(ROW_LABELS["uamcts_cc4"],"UAMCTS-CC4 (adapted)")
        self.assertEqual(ROW_LABELS["carl_cc4"],"CARL-CC4 (adapted)")

    def test_duplicate_repeat_blocks_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; self.make_table(root,"table1"); self.make_repeat(root,"table1","dca_cc4",5,directory_suffix="-duplicate")
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation): result=generate(root,Path(tmp)/"out")
            self.assertEqual(result["figures"]["table1_method_comparison"]["status"],"BLOCKED")
            self.assertTrue(any("table1/dca_cc4" in x for x in result["row_completeness_issues"]))

    def test_table1_only_is_partial_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; self.make_table(root,"table1")
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation): result=generate(root,Path(tmp)/"out")
            self.assertEqual(result["status"],"PARTIAL")
            self.assertTrue(all(row["status"]=="BLOCKED" for row in result["figures"].values()))
            for stem in result["figures"]:
                self.assertFalse((Path(tmp)/"out"/f"{stem}.svg").exists())
                self.assertFalse((Path(tmp)/"out"/f"{stem}.png").exists())

    def test_different_reward_semantics_cannot_enter_learning_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"formal"; self.make_table(root,"table2"); path=root/"table2"/"wm_rl"/"repeat_3"
            manifest=json.loads((path/"manifest.json").read_text()); manifest["reward_semantics"]="different_reward"; write_json(path/"manifest.json",manifest)
            report=json.loads((path/"eligibility_report.json").read_text()); report["input_sha256"]["manifest.json"]=sha(path/"manifest.json"); write_json(path/"eligibility_report.json",report)
            with patch("formal_experiments.evaluation.generate_paper_figures.validate_run_directory",side_effect=self.validation): result=generate(root,Path(tmp)/"out")
            self.assertEqual(result["figures"]["training_learning_curve"]["status"],"BLOCKED")
            self.assertTrue(any("Full-Reward semantics" in x["reason"] for x in result["rejected_inputs"]))


if __name__ == "__main__": unittest.main()
