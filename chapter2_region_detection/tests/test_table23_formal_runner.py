import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from formal_experiments.ours.run_table23_formal import (
    FORMAL_EPISODE_TICKS, POLICY_SEEDS, TEST_SEEDS, TRAIN_SEEDS,
    VALIDATION_SEEDS, FormalStageRunner, ResumeJournal, RunProfile,
    RealCC4ProductionCallbacks, TRAINING_CURVE_SCHEMA, _derive_fail_only_training_contract,
    _load_verified_ofox_cache_audit, _training_curve_figure_artifact,
    assert_immutable, formal_batch_plan,
    job_dependencies, physical_jobs, preflight, resolved_repeat_metadata, row_specs,
    select_validation_checkpoint, sha256_file, validate_formal_batch_plan,
    verify_training_curve_bindings,
)


class TestTable23FormalRunner(unittest.TestCase):
    def test_frozen_profile_and_shared_lwm_full_identity(self):
        profile = RunProfile()
        self.assertEqual((profile.train_seeds, profile.validation_seeds, profile.test_seeds),
                         (TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS))
        self.assertEqual(profile.episode_ticks, FORMAL_EPISODE_TICKS)
        self.assertEqual(len(row_specs()), 7); self.assertEqual(len(physical_jobs()), 6)
        shared = [x for x in row_specs() if x.canonical_id == "lwm_full"]
        self.assertEqual({(x.table_id, x.row_id) for x in shared},
                         {("table2", "lwm_rl"), ("table3", "full_reward")})
        with self.assertRaisesRegex(ValueError, "formal profile"):
            RunProfile("formal", (1000,), VALIDATION_SEEDS, TEST_SEEDS, 500)

    def test_dev_is_never_eligible_and_repeat_seed_is_frozen(self):
        profile = RunProfile("dev", (1000,), (2000,), (3200,), 20)
        meta = resolved_repeat_metadata(row_specs()[0], repeat_index=2, profile=profile)
        self.assertFalse(meta["formal_result_eligible"])
        self.assertEqual(meta["training_seed"], POLICY_SEEDS[1])

    def test_validation_selection_rejects_test_data_and_tie_breaks(self):
        sha_a, sha_b = "a" * 64, "b" * 64
        selected = select_validation_checkpoint([
            {"epoch": 2, "validation_score": 1, "checkpoint_sha256": sha_a},
            {"epoch": 1, "validation_score": 1, "checkpoint_sha256": sha_b},
        ])
        self.assertEqual(selected["epoch"], 1)
        with self.assertRaisesRegex(ValueError, "test fields"):
            select_validation_checkpoint([{"epoch": 1, "validation_score": 1,
                                           "checkpoint_sha256": sha_a, "test_score": 9}])

    def test_fail_only_gate_stays_blocked_without_approved_clean_provenance(self):
        root = Path(__file__).resolve().parents[1]
        specs = {(x.table_id, x.row_id): x for x in row_specs()}
        prior = root / "outputs/priorrl_cc4/prototypes/frozen_prototypes.json"
        report = preflight(
            project_root=root,
            profile=RunProfile(),
            selected_spec=specs[("table3", "fail_only")],
            offline_prior_artifact=prior,
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("no clean-commit approved" in item for item in report["errors"]))
        with self.assertRaisesRegex(RuntimeError, "no clean-commit approved"):
            _derive_fail_only_training_contract(root)

    def test_fail_only_contract_uses_validated_final_sidecar_bindings(self):
        source_root = Path(__file__).resolve().parents[1]
        source_dir = source_root / "outputs/formal_v3/table3_reward_models/fail_only"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "outputs/formal_v3/table3_reward_models/fail_only"
            target.mkdir(parents=True)
            for name in ("frozen_manifest.json", "response_reward_predictor.pt"):
                shutil.copyfile(source_dir / name, target / name)
            sidecar = target / "provenance_sidecar.json"
            sidecar.write_text(json.dumps({"status": "final"}), encoding="utf-8")
            reward_binding = {
                "manifest_sha256": sha256_file(target / "frozen_manifest.json"),
                "checkpoint_sha256": sha256_file(target / "response_reward_predictor.pt"),
            }
            with patch(
                "formal_experiments.ours.run_table23_formal.validate_fail_only_provenance",
                return_value={"reward_artifact": reward_binding},
            ):
                contract = _derive_fail_only_training_contract(root)
            self.assertEqual(contract["manifest_sha256"], reward_binding["manifest_sha256"])
            self.assertEqual(contract["checkpoint_sha256"], reward_binding["checkpoint_sha256"])
            self.assertEqual(contract["provenance_sidecar_sha256"], sha256_file(sidecar))
            with patch(
                "formal_experiments.ours.run_table23_formal.validate_fail_only_provenance",
                return_value={"reward_artifact": {**reward_binding, "checkpoint_sha256": "f" * 64}},
            ):
                with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                    _derive_fail_only_training_contract(root)

    def test_selected_jobs_are_not_blocked_by_unrelated_gates(self):
        root = Path(__file__).resolve().parents[1]
        specs = {(x.table_id, x.row_id): x for x in row_specs()}
        with patch("baselines.rsmbrl_cc4.artifact_preflight.run_preflight",
                   return_value={"status": "PASS", "eligible": True, "errors": []}):
            for key in (("table2", "rl_only"), ("table2", "wm_rl")):
                report = preflight(project_root=root, profile=RunProfile(), selected_spec=specs[key])
                self.assertEqual(report["scope"], "selected_job")
                self.assertEqual(report["status"], "PASS", report["errors"])
                self.assertFalse(report["selected_job"]["dependencies"]["offline_prior"])
            prior = root / "outputs/priorrl_cc4/prototypes/frozen_prototypes.json"
            delay = preflight(project_root=root, profile=RunProfile(),
                              selected_spec=specs[("table3", "delay_only")],
                              offline_prior_artifact=prior)
            self.assertEqual(delay["status"], "PASS", delay["errors"])

    def test_world_model_jobs_fail_closed_on_strict_artifact_gate(self):
        root = Path(__file__).resolve().parents[1]
        specs = {(x.table_id, x.row_id): x for x in row_specs()}
        with patch("baselines.rsmbrl_cc4.artifact_preflight.run_preflight",
                   return_value={"status": "FAIL", "eligible": False,
                                 "errors": ["missing frozen normalizer sidecar"]}):
            blocked = preflight(project_root=root, profile=RunProfile(),
                                selected_spec=specs[("table2", "wm_rl")], device="cuda")
            allowed = preflight(project_root=root, profile=RunProfile(),
                                selected_spec=specs[("table2", "rl_only")], device="cuda")
        self.assertEqual(blocked["status"], "FAIL")
        self.assertTrue(any("normalizer sidecar" in item for item in blocked["errors"]))
        self.assertEqual(allowed["status"], "PASS", allowed["errors"])

    def test_prior_and_fail_only_jobs_fail_closed_on_own_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        specs = {(x.table_id, x.row_id): x for x in row_specs()}
        llm = preflight(project_root=root, profile=RunProfile(),
                        selected_spec=specs[("table2", "llm_rl")])
        self.assertEqual(llm["status"], "FAIL")
        self.assertTrue(any("offline-prior-artifact" in item for item in llm["errors"]))
        fail = preflight(project_root=root, profile=RunProfile(),
                         selected_spec=specs[("table3", "fail_only")])
        self.assertEqual(fail["status"], "FAIL")
        self.assertTrue(any("offline-prior-artifact" in item for item in fail["errors"]))
        self.assertTrue(any("no clean-commit approved" in item for item in fail["errors"]))

    def test_ofox_audit_is_derived_from_final_provenance_and_rejects_tampering(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "outputs/priorrl_cc4/prototypes"
        audit = _load_verified_ofox_cache_audit(source / "frozen_prototypes.json")
        self.assertEqual((audit["provider_calls"], audit["generated"], audit["verified_entries"]), (6, 6, 6))
        self.assertTrue(audit["coverage_pass"]); self.assertFalse(audit["online_calls_allowed"])
        specs = {(x.table_id, x.row_id): x for x in row_specs()}
        with patch("baselines.rsmbrl_cc4.artifact_preflight.run_preflight",
                   return_value={"status": "PASS", "eligible": True, "errors": []}):
            for row_id in ("llm_rl", "lwm_rl"):
                report = preflight(project_root=root, profile=RunProfile(),
                                   selected_spec=specs[("table2", row_id)],
                                   offline_prior_artifact=source / "frozen_prototypes.json")
                self.assertEqual(report["status"], "PASS", report["errors"])
                self.assertEqual(report["gates"]["offline_prior"]["ofox_cache_audit"], audit)
        for mutation in (
            "artifact", "coverage", "provider_calls", "duplicate_entry", "online",
            "radius", "threshold_source", "base_migration", "semantic_missing",
            "freeze_provider_calls",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary)
                for name in ("frozen_prototypes.json", "frozen_prototype_coverage.json",
                             "frozen_prototype_provenance.json"):
                    shutil.copyfile(source / name, target / name)
                provenance = target / "frozen_prototype_provenance.json"
                if mutation == "artifact": (target / "frozen_prototypes.json").write_text("{}", encoding="utf-8")
                elif mutation == "coverage": (target / "frozen_prototype_coverage.json").write_text("{}", encoding="utf-8")
                elif mutation == "freeze_provider_calls":
                    coverage_path = target / "frozen_prototype_coverage.json"
                    coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
                    coverage_payload["provider_calls_during_freeze"] = 1
                    coverage_path.write_text(json.dumps(coverage_payload, sort_keys=True), encoding="utf-8")
                    payload = json.loads(provenance.read_text(encoding="utf-8"))
                    payload["coverage"]["file_sha256"] = sha256_file(coverage_path)
                    provenance.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                else:
                    payload = json.loads(provenance.read_text(encoding="utf-8"))
                    if mutation == "provider_calls": payload["provider_calls"] = 7
                    elif mutation == "duplicate_entry": payload["ofox_entries"][1] = payload["ofox_entries"][0]
                    elif mutation == "online": payload["online_allowed"] = True
                    elif mutation == "radius": payload["radius"] = 1.9091052783574496
                    elif mutation == "threshold_source": payload["threshold_source"] = "validation_post_supplement"
                    elif mutation == "base_migration": payload["base_artifact"]["migration"] = "v2_refit"
                    else: payload["base_artifact"]["semantic_missing"] = 1
                    provenance.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                with patch("formal_experiments.ours.run_table23_formal.FROZEN_PROTOTYPE_PROVENANCE_SHA256",
                           sha256_file(provenance)):
                    with self.assertRaisesRegex(RuntimeError, "BLOCKED"):
                        _load_verified_ofox_cache_audit(target / "frozen_prototypes.json")

    def test_dependency_matrix_and_shared_full_are_stable(self):
        specs = {(x.table_id, x.row_id): x for x in row_specs()}
        self.assertEqual(job_dependencies(specs[("table2", "rl_only")]), {
            "offline_prior": False, "world_model": False, "ablation_reward_gate": False})
        self.assertTrue(job_dependencies(specs[("table2", "lwm_rl")])["offline_prior"])
        self.assertEqual(specs[("table2", "lwm_rl")].canonical_id,
                         specs[("table3", "full_reward")].canonical_id)

    def test_formal_batch_plan_is_exact_and_rejects_missing_or_duplicate(self):
        plan = formal_batch_plan()
        self.assertEqual(len(plan), 30)
        self.assertEqual(len({(x["canonical_id"], x["repeat_index"]) for x in plan}), 30)
        self.assertFalse(any(x["table_id"] == "table3" and x["row_id"] == "full_reward" for x in plan))
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_formal_batch_plan(plan[:-1])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_formal_batch_plan((*plan[:-1], plan[0]))

    def test_production_train_incrementally_writes_curve_and_test_does_not_touch_it(self):
        class FakeTorch:
            @staticmethod
            def save(_state, path): Path(path).write_bytes(b"checkpoint")
        class FakePolicy:
            @staticmethod
            def state_dict(): return {"weight": 1}
        callbacks = RealCC4ProductionCallbacks.__new__(RealCC4ProductionCallbacks)
        with tempfile.TemporaryDirectory() as temporary:
            callbacks.torch = FakeTorch(); callbacks.run_dir = Path(temporary)
            callbacks.runtime = SimpleNamespace(policy=FakePolicy(), split=None)
            callbacks.profile = RunProfile("dev", (1000, 1001), (2000,), (3200,), 20)
            callbacks.repeat_index = 2; callbacks.checkpoint_stride = 2
            callbacks.reward_mode = "Full-Reward"; callbacks.candidates = []
            callbacks.episodes = []; callbacks.decisions = []; callbacks._test_checkpoint_loaded = True
            transition = lambda reward: SimpleNamespace(reward=reward)
            def fake_episode(_runtime, *, episode_seed, split, **_kwargs):
                if split == "train": return SimpleNamespace(transitions=[transition(episode_seed / 1000)]), []
                return (None, [{"tick": 1}], {"episode_seed": episode_seed})
            with patch("formal_experiments.ours.run_table2_real_smoke.run_episode", side_effect=fake_episode), \
                 patch("formal_experiments.ours.variant_training.ppo_update", return_value={"loss": 0.25}):
                callbacks.train((1000, 1001))
                curve = callbacks.run_dir / "training_curve.jsonl"
                records = [json.loads(line) for line in curve.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([x["environment_steps"] for x in records], [20, 40])
                self.assertTrue(all(x["schema"] == TRAINING_CURVE_SCHEMA for x in records))
                self.assertEqual(records[0]["training_objective_reward"], 1.0)
                self.assertEqual(records[1]["policy_seed"], POLICY_SEEDS[1])
                checkpoint = callbacks.run_dir / records[1]["checkpoint"]["path"]
                self.assertEqual(records[1]["checkpoint"]["sha256"], sha256_file(checkpoint))
                declaration = _training_curve_figure_artifact(curve)
                self.assertEqual(declaration["record_count"], 2)
                self.assertEqual((declaration["split"], declaration["x_field"], declaration["y_field"]),
                                 ("train", "environment_steps", "training_objective_reward"))
                before = curve.read_bytes(); callbacks.test((3200,)); self.assertEqual(curve.read_bytes(), before)

    def test_training_curve_manifest_eligibility_triple_binding_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); curve = root / "training_curve.jsonl"
            curve.write_text(json.dumps({"schema": TRAINING_CURVE_SCHEMA,
                                         "environment_steps": 20,
                                         "training_objective_reward": 1.0}) + "\n", encoding="utf-8")
            digest = sha256_file(curve)
            manifest = {"figure_artifacts": {"training_curve": {
                "schema": TRAINING_CURVE_SCHEMA, "path": "training_curve.jsonl", "sha256": digest,
                "split": "train", "x_field": "environment_steps",
                "y_field": "training_objective_reward", "record_count": 1}},
                "artifact_sha256": {"training_curve.jsonl": digest}}
            eligibility = {"input_sha256": {"training_curve.jsonl": digest}}
            self.assertEqual(verify_training_curve_bindings(root, manifest, eligibility), digest)
            for location in ("figure", "artifact", "eligibility"):
                broken_manifest = json.loads(json.dumps(manifest)); broken_eligibility = json.loads(json.dumps(eligibility))
                if location == "figure": broken_manifest["figure_artifacts"]["training_curve"]["sha256"] = "f" * 64
                elif location == "artifact": broken_manifest["artifact_sha256"]["training_curve.jsonl"] = "f" * 64
                else: broken_eligibility["input_sha256"]["training_curve.jsonl"] = "f" * 64
                with self.assertRaisesRegex(RuntimeError, "triple binding"):
                    verify_training_curve_bindings(root, broken_manifest, broken_eligibility)
            curve.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "triple binding"):
                verify_training_curve_bindings(root, manifest, eligibility)

    def test_stage_runner_split_isolation_resume_and_immutable_test(self):
        calls = []
        hashes = {"policy": "a" * 64, "world": "b" * 64, "reward": "c" * 64, "scaler": "d" * 64}
        callbacks = {
            "preflight": lambda: {"status": "PASS"},
            "train": lambda seeds: calls.append(("train", tuple(seeds))) or {"checkpoint_count": 2},
            "validation": lambda seeds: calls.append(("validation", tuple(seeds))) or [
                {"epoch": 1, "validation_score": 2.0, "checkpoint_sha256": "e" * 64}],
            "test": lambda seeds: calls.append(("test", tuple(seeds))) or {"decisions_audited": True},
            "hashes": lambda: hashes,
            "integrity": lambda _: {"passed": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            runner = FormalStageRunner(spec=row_specs()[0], repeat_index=1,
                                       profile=RunProfile(), run_dir=Path(temporary), callbacks=callbacks)
            result = runner.run(); runner.run()
            self.assertEqual([x[0] for x in calls], ["train", "validation", "test"])
            self.assertEqual(calls[0][1], TRAIN_SEEDS); self.assertEqual(calls[1][1], VALIDATION_SEEDS)
            self.assertEqual(calls[2][1], TEST_SEEDS)
            self.assertEqual(set(result["completed"]), set(ResumeJournal.STAGES))

    def test_production_callback_preflight_is_dependency_aware(self):
        callback = object.__new__(RealCC4ProductionCallbacks)
        callback.spec = physical_jobs()[0]
        callback.project_root = Path("project")
        callback.offline_prior_artifact = Path("prior.json")
        callback.device = "cpu"
        callback.frozen_artifacts = {"artifact_sha256": "a" * 64}
        callback.startup_preflight_report = None
        expected = {"status": "FAIL", "errors": ["blocked"],
                    "canonical_id": callback.spec.canonical_id}
        with patch(
            "formal_experiments.ours.run_table23_formal._job_preflight",
            return_value=expected,
        ) as gate:
            report = callback.preflight()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["frozen_artifacts"], callback.frozen_artifacts)
        gate.assert_called_once_with(
            project_root=callback.project_root, spec=callback.spec,
            offline_prior_artifact=callback.offline_prior_artifact, device="cpu",
        )

        callback.startup_preflight_report = {
            "status": "PASS", "selected_job": {"canonical_id": callback.spec.canonical_id},
        }
        with patch(
            "formal_experiments.ours.run_table23_formal._job_preflight",
            side_effect=AssertionError("startup report must be reused"),
        ):
            self.assertEqual(callback.preflight()["status"], "PASS")

    def test_test_hash_mutation_fails(self):
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            assert_immutable({"policy": "a"}, {"policy": "b"})


if __name__ == "__main__": unittest.main()
