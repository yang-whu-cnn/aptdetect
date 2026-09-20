"""Resumable, fail-closed execution contract for the v3 Table 2/3 runs.

This module deliberately separates orchestration from the expensive CC4 episode
loop.  ``run_table2_real_smoke.run_episode`` is the reference callback used by
the formal launcher; dry-run/preflight never creates an environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from formal_experiments.common.prior_provenance import validate_frozen_prior_provenance
from formal_experiments.common.run_manifest import POLICY_SEEDS
from formal_experiments.ours.reward_ablation import RewardMode
from formal_experiments.ours.reward_artifact_contract import validate_frozen_reward_artifact
from formal_experiments.ours.reward_provenance import validate_fail_only_provenance
from formal_experiments.ours.table2_variants import Table2Variant

TRAIN_SEEDS = tuple(range(1000, 1032))
VALIDATION_SEEDS = tuple(range(2000, 2008))
TEST_SEEDS = tuple(range(4000, 4100))
FORMAL_EPISODE_TICKS = 500
ORCHESTRATOR_VERSION = "table23_formal_runner_v2"
FULL_REWARD_SEMANTICS = "cc4_v3_full_reward"
TRAINING_CURVE_SCHEMA = "cc4_v3_training_curve_v1"
FROZEN_PROTOTYPE_PROVENANCE_SHA256 = "464b306274a4123acf39384d289eca632e02960f3dfb9a381e9cf7613d471363"
OFOX_ENTRY_SET_SHA256 = "7941f6f11d47989265955bdc6caff950de97a9f5b9562afa8f7cb5e0209a7e10"
OFOX_SUPPLEMENT_MANIFEST_SHA256 = "ece590f5bc01ea6b6cf06458cd86096914e85d886d50686b2c7c9d365e909d44"
OFOX_LOGICAL_MANIFEST_SHA256 = "5ddc621817a902d3c8f70d76353f80ba2ab0c37e1947d8e855a5f4518443e49f"


@dataclass(frozen=True)
class RunProfile:
    mode: str = "formal"
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS
    test_seeds: tuple[int, ...] = TEST_SEEDS
    episode_ticks: int = FORMAL_EPISODE_TICKS

    def __post_init__(self):
        if self.mode not in {"formal", "dev"}: raise ValueError("mode must be formal or dev")
        for name in ("train_seeds", "validation_seeds", "test_seeds"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)): raise ValueError(f"{name} must be nonempty and unique")
        if set(self.train_seeds) & (set(self.validation_seeds) | set(self.test_seeds)) or set(self.validation_seeds) & set(self.test_seeds):
            raise ValueError("train/validation/test seed leakage")
        if self.episode_ticks <= 0: raise ValueError("episode_ticks must be positive")
        if self.mode == "formal" and (self.train_seeds != TRAIN_SEEDS or self.validation_seeds != VALIDATION_SEEDS or
                                      self.test_seeds != TEST_SEEDS or self.episode_ticks != FORMAL_EPISODE_TICKS):
            raise ValueError("formal profile is frozen to 32/8/100 seeds and 500 ticks")

    @property
    def formal_result_eligible(self) -> bool: return self.mode == "formal"


@dataclass(frozen=True)
class RowSpec:
    table_id: str
    row_id: str
    component_variant: str
    reward_mode: str
    reward_semantics: str
    canonical_id: str


def row_specs() -> tuple[RowSpec, ...]:
    table2 = tuple(RowSpec("table2", v.value.lower().replace("-", "_"), v.value.lower().replace("-", "_"),
                           "full_reward", FULL_REWARD_SEMANTICS,
                           "lwm_full" if v == Table2Variant.LWM_RL else f"table2_{v.value}")
                   for v in Table2Variant)
    table3 = tuple(RowSpec("table3", m.value.lower().replace("-", "_"), "lwm_rl",
                           m.value.lower().replace("-", "_"), (FULL_REWARD_SEMANTICS if m == RewardMode.FULL_REWARD else
                           "cc4_v3_delay_only" if m == RewardMode.DELAY_ONLY else
                           "cc4_v3_fail_only"),
                           "lwm_full" if m == RewardMode.FULL_REWARD else f"table3_{m.value}")
                   for m in RewardMode)
    return table2 + table3


def physical_jobs() -> tuple[RowSpec, ...]:
    """Return six physical jobs; Table2 LWM Full and Table3 Full are one run."""
    selected: dict[str, RowSpec] = {}
    for spec in row_specs(): selected.setdefault(spec.canonical_id, spec)
    return tuple(selected.values())


def formal_batch_plan() -> tuple[dict[str, Any], ...]:
    """Return the frozen 30-run plan (six physical jobs x five repeats)."""
    plan = tuple(
        {"canonical_id": spec.canonical_id, "table_id": spec.table_id,
         "row_id": spec.row_id, "repeat_index": repeat_index,
         "training_seed": POLICY_SEEDS[repeat_index - 1]}
        for spec in physical_jobs() for repeat_index in range(1, 6)
    )
    validate_formal_batch_plan(plan)
    return plan


def validate_formal_batch_plan(plan: Iterable[Mapping[str, Any]]) -> None:
    """Fail closed on missing/duplicate physical jobs or logical alias runs."""
    records = [dict(record) for record in plan]
    expected_ids = {spec.canonical_id for spec in physical_jobs()}
    expected = {(canonical_id, repeat_index)
                for canonical_id in expected_ids for repeat_index in range(1, 6)}
    actual = [(str(record.get("canonical_id")), record.get("repeat_index"))
              for record in records]
    if len(records) != 30 or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("formal batch plan must contain each of six physical jobs at repeats 1..5 exactly once")
    if any(record.get("table_id") == "table3" and record.get("row_id") == "full_reward"
           for record in records):
        raise ValueError("Table3 full_reward is a logical alias and must not be a physical job")


def job_dependencies(spec: RowSpec) -> dict[str, bool]:
    """Declare only the frozen inputs consumed by one physical job."""
    return {
        "offline_prior": spec.component_variant in {"llm_rl", "lwm_rl"},
        "world_model": spec.component_variant in {"wm_rl", "lwm_rl"},
        "ablation_reward_gate": spec.reward_mode in {"delay_only", "fail_only"},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def verify_training_curve_bindings(run_dir: Path, manifest: Mapping[str, Any],
                                   eligibility_report: Mapping[str, Any]) -> str:
    """Return the curve digest only when all three integrity bindings agree."""
    curve_path = Path(run_dir) / "training_curve.jsonl"
    if not curve_path.is_file(): raise RuntimeError("missing training_curve.jsonl")
    digest = sha256_file(curve_path)
    figure = manifest.get("figure_artifacts", {}).get("training_curve", {})
    declared = (figure.get("sha256"),
                manifest.get("artifact_sha256", {}).get("training_curve.jsonl"),
                eligibility_report.get("input_sha256", {}).get("training_curve.jsonl"))
    if figure.get("schema") != TRAINING_CURVE_SCHEMA or figure.get("path") != "training_curve.jsonl":
        raise RuntimeError("invalid training curve figure declaration")
    record_count = len(curve_path.read_text(encoding="utf-8").splitlines())
    if (figure.get("split"), figure.get("x_field"), figure.get("y_field"), figure.get("record_count")) != (
            "train", "environment_steps", "training_objective_reward", record_count):
        raise RuntimeError("invalid training curve figure semantics")
    if any(item != digest for item in declared):
        raise RuntimeError("training curve SHA256 triple binding mismatch")
    return digest


def _training_curve_figure_artifact(curve_path: Path) -> dict[str, Any]:
    curve_path = Path(curve_path)
    records = [line for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records: raise RuntimeError("training curve must contain at least one record")
    return {"schema": TRAINING_CURVE_SCHEMA, "path": "training_curve.jsonl",
            "sha256": sha256_file(curve_path), "split": "train",
            "x_field": "environment_steps", "y_field": "training_objective_reward",
            "record_count": len(records)}


def _load_verified_ofox_cache_audit(artifact_path: Path) -> dict[str, Any]:
    """Verify the final frozen prototype provenance and derive its public audit."""
    audit = validate_frozen_prior_provenance(
        Path(artifact_path),
        expected_provenance_sha256=FROZEN_PROTOTYPE_PROVENANCE_SHA256,
        expected_entry_set_sha256=OFOX_ENTRY_SET_SHA256,
        expected_supplement_manifest_sha256=OFOX_SUPPLEMENT_MANIFEST_SHA256,
        expected_logical_manifest_sha256=OFOX_LOGICAL_MANIFEST_SHA256,
    )
    audit["provenance_sha256"] = audit.pop("prototype_provenance_sha256")
    for private in ("prototype_sha256", "prototype_file_sha256",
                    "prototype_coverage_sha256", "supplement_manifest_sha256"):
        audit.pop(private)
    return audit


def select_validation_checkpoint(records: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select solely by validation score, then deterministic epoch/SHA tie-break."""
    candidates = []
    for record in records:
        forbidden = {key for key in record if key.startswith("test")}
        if forbidden: raise ValueError(f"checkpoint selection received test fields: {sorted(forbidden)}")
        score = float(record["validation_score"]); epoch = int(record["epoch"])
        digest = str(record["checkpoint_sha256"])
        if not digest or len(digest) != 64: raise ValueError("checkpoint SHA256 required")
        candidates.append((score, -epoch, digest, dict(record)))
    if not candidates: raise ValueError("no validation checkpoints")
    return max(candidates, key=lambda item: item[:3])[3]


def assert_immutable(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    if dict(before) != dict(after):
        changed = sorted(set(before) | set(after), key=str)
        changed = [key for key in changed if before.get(key) != after.get(key)]
        raise RuntimeError(f"test mutated frozen policy/model/scaler artifacts: {changed}")


def _load_gate(path: Path, mode: RewardMode) -> dict[str, Any]:
    if mode == RewardMode.FULL_REWARD: return {"pass": True, "source": "frozen_full_reward_preflight"}
    project_root = Path(__file__).resolve().parents[2]
    provenance_sha256 = None
    if mode == RewardMode.FAIL_ONLY:
        provenance_path = path.with_name("provenance_sidecar.json")
        provenance = validate_fail_only_provenance(provenance_path, project_root=project_root)
        provenance_sha256 = sha256_file(provenance_path)
    _, _, audit = validate_frozen_reward_artifact(
        path, mode=mode, project_root=project_root
    )
    if provenance_sha256 is not None:
        reward_binding = provenance.get("reward_artifact", {})
        if (reward_binding.get("manifest_sha256") != audit.get("manifest_sha256")
                or reward_binding.get("checkpoint_sha256") != audit.get("checkpoint_sha256")):
            raise RuntimeError("BLOCKED: Fail-Only provenance/reward artifact binding mismatch")
        audit = {**audit, "provenance_sidecar_sha256": provenance_sha256}
    return audit


def _derive_fail_only_training_contract(project_root: Path) -> dict[str, Any]:
    """Derive the paper-facing contract from the helper-verified frozen artifact."""
    path = (Path(project_root) / "outputs/formal_v3/table3_reward_models/fail_only/frozen_manifest.json")
    provenance_path = path.with_name("provenance_sidecar.json")
    provenance = validate_fail_only_provenance(provenance_path, project_root=Path(project_root))
    payload, _, audit = validate_frozen_reward_artifact(
        path, mode=RewardMode.FAIL_ONLY, project_root=Path(project_root))
    reward_binding = provenance.get("reward_artifact", {})
    if (reward_binding.get("manifest_sha256") != audit.get("manifest_sha256")
            or reward_binding.get("checkpoint_sha256") != audit.get("checkpoint_sha256")):
        raise RuntimeError("BLOCKED: Fail-Only provenance/reward artifact binding mismatch")
    loss = payload["training_loss"]
    return {"loss_name": loss["name"], "beta": float(loss["beta"]),
            "legacy_mse": ("diagnostic_only" if not payload["diagnostic_hurdle_used_for_formal_artifact"]
                           else "used_for_formal_artifact"),
            "gate_pass": bool(payload["quality_gate"]["pass"]),
            "train_seed_count": len(TRAIN_SEEDS), "validation_seed_count": len(VALIDATION_SEEDS),
            "train_validation_overlap": len(set(TRAIN_SEEDS) & set(VALIDATION_SEEDS)),
            "test_leak_count": 0 if payload["test_seeds_used"] is False else len(TEST_SEEDS),
            "checkpoint_sha256": reward_binding["checkpoint_sha256"],
            "manifest_sha256": reward_binding["manifest_sha256"],
            "provenance_sidecar_sha256": sha256_file(provenance_path)}


def _job_preflight(*, project_root: Path, spec: RowSpec,
                   offline_prior_artifact: Path | None,
                   device: str = "cpu") -> dict[str, Any]:
    dependencies = job_dependencies(spec)
    gates: dict[str, Any] = {}
    errors: list[str] = []
    if dependencies["ablation_reward_gate"]:
        mode = RewardMode({"delay_only": "Delay-Only", "fail_only": "Fail-Only"}[spec.reward_mode])
        path = project_root / "outputs/formal_v3/table3_reward_models" / spec.reward_mode / "frozen_manifest.json"
        try:
            gates["reward"] = _load_gate(path, mode)
        except Exception as exc:
            errors.append(str(exc))
    else:
        gates["reward"] = {"pass": True, "source": "frozen_full_reward_preflight"}
    if dependencies["offline_prior"]:
        if offline_prior_artifact is None:
            errors.append("BLOCKED: this job requires --offline-prior-artifact")
        else:
            path = Path(offline_prior_artifact)
            if not path.is_absolute(): path = project_root / path
            if not path.is_file():
                errors.append(f"BLOCKED: offline prior artifact not found: {path}")
            else:
                try:
                    from baselines.priorrl_cc4.prototype_retrieval import FrozenPrototypePriorAdapter
                    FrozenPrototypePriorAdapter.load_artifact(path)
                    audit = _load_verified_ofox_cache_audit(path)
                    gates["offline_prior"] = {"pass": True, "sha256": sha256_file(path),
                                               "path": str(path), "ofox_cache_audit": audit}
                except Exception as exc:
                    errors.append(f"BLOCKED: invalid frozen K6/H4 offline prior artifact: {exc}")
    else:
        gates["offline_prior"] = {"pass": True, "required": False}
    if dependencies["world_model"]:
        try:
            from baselines.rsmbrl_cc4.artifact_preflight import run_preflight as run_wm_preflight
            wm_gate = run_wm_preflight(device=device)
            if not wm_gate.get("eligible"):
                raise RuntimeError("; ".join(wm_gate.get("errors", ())))
            gates["world_model"] = wm_gate
        except Exception as exc:
            errors.append(f"BLOCKED: frozen WM/Full-Reward preflight failed: {exc}")
    else:
        gates["world_model"] = {"pass": True, "required": False}
    return {"canonical_id": spec.canonical_id, "table_id": spec.table_id, "row_id": spec.row_id,
            "dependencies": dependencies, "status": "PASS" if not errors else "FAIL",
            "errors": errors, "gates": gates}


def preflight(*, project_root: Path, profile: RunProfile, selected_spec: RowSpec | None = None,
              offline_prior_artifact: Path | None = None,
              device: str = "cpu") -> dict[str, Any]:
    """Preflight either one physical job or the complete six-job plan.

    A selected job is never blocked by an unrelated row.  With no selection,
    the report remains a conservative whole-plan readiness summary.
    """
    scoped_specs = (selected_spec,) if selected_spec is not None else physical_jobs()
    jobs = (_job_preflight(project_root=project_root, spec=spec,
                           offline_prior_artifact=offline_prior_artifact,
                           device=device)
            for spec in scoped_specs)
    job_reports = {item["canonical_id"]: item for item in jobs}
    active = job_reports[selected_spec.canonical_id] if selected_spec is not None else None
    errors = list(active["errors"]) if active is not None else [
        f"{job_id}: {error}" for job_id, item in job_reports.items() for error in item["errors"]]
    gates = dict(active["gates"]) if active is not None else {
        job_id: item["gates"] for job_id, item in job_reports.items()}
    return {"version": ORCHESTRATOR_VERSION, "status": "PASS" if not errors else "FAIL",
            "errors": errors, "gates": gates, "formal_result_eligible": profile.formal_result_eligible,
            "scope": "selected_job" if selected_spec is not None else "complete_plan",
            "selected_job": active, "jobs": job_reports,
            "profile": {**asdict(profile), "train_seeds": list(profile.train_seeds),
                        "validation_seeds": list(profile.validation_seeds), "test_seeds": list(profile.test_seeds)},
            "policy_seeds": list(POLICY_SEEDS), "physical_job_count": len(physical_jobs()),
            "logical_row_count": len(row_specs()), "formal_batch_plan": list(formal_batch_plan())}


class ResumeJournal:
    STAGES = ("preflight", "train", "validation", "test", "integrity")
    def __init__(self, path: Path): self.path = Path(path)
    def load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {"version": 1, "completed": {}}
    def complete(self, stage: str, payload: Mapping[str, Any]) -> None:
        if stage not in self.STAGES: raise ValueError("unknown stage")
        state = self.load(); done = state["completed"]
        expected = self.STAGES[:self.STAGES.index(stage)]
        if any(item not in done for item in expected): raise RuntimeError("stage ordering violation")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        entry = {"payload": dict(payload), "sha256": hashlib.sha256(encoded.encode()).hexdigest()}
        if stage in done and done[stage] != entry: raise RuntimeError("resume payload mismatch")
        done[stage] = entry; self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp"); temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"); temporary.replace(self.path)


class FormalStageRunner:
    """Small injectable stage machine; production callbacks own the real loop.

    The callback interface makes stage isolation testable without starting
    CybORG. A callback receives only the seeds for its declared split.
    """
    def __init__(self, *, spec: RowSpec, repeat_index: int, profile: RunProfile,
                 run_dir: Path, callbacks: Mapping[str, Any]):
        self.spec, self.repeat_index, self.profile = spec, repeat_index, profile
        self.run_dir, self.callbacks = Path(run_dir), dict(callbacks)
        self.journal = ResumeJournal(self.run_dir / "resume_state.json")

    def run(self) -> dict[str, Any]:
        required = {"train", "validation", "test", "hashes"}
        if required - set(self.callbacks): raise ValueError(f"missing callbacks: {sorted(required-set(self.callbacks))}")
        state = self.journal.load()["completed"]
        if "preflight" not in state:
            report = self.callbacks.get("preflight", lambda: {"status": "PASS"})()
            if report.get("status") != "PASS": raise RuntimeError("formal preflight FAIL")
            self.journal.complete("preflight", report)
        if "train" not in state:
            trained = self.callbacks["train"](self.profile.train_seeds)
            self.journal.complete("train", trained)
        if "validation" not in state:
            # Only validation records enter selection; test callback is not in scope.
            records = self.callbacks["validation"](self.profile.validation_seeds)
            selected = select_validation_checkpoint(records)
            self.journal.complete("validation", {"selected": dict(selected), "candidate_count": len(records)})
        if "test" not in state:
            if "prepare_test" in self.callbacks: self.callbacks["prepare_test"]()
            before = dict(self.callbacks["hashes"]())
            result = self.callbacks["test"](self.profile.test_seeds)
            after = dict(self.callbacks["hashes"]()); assert_immutable(before, after)
            if not result.get("decisions_audited"):
                raise RuntimeError("test result missing decisions audit")
            self.journal.complete("test", {"result": dict(result), "immutable_sha256": before})
        if "integrity" not in state:
            result = self.callbacks.get("integrity", lambda _path: {"passed": True})(self.run_dir)
            if not result.get("passed"): raise RuntimeError("formal integrity validator FAIL")
            self.journal.complete("integrity", result)
        return self.journal.load()


class RealCC4ProductionCallbacks:
    """Production implementation backed by the audited CC4 policy loop."""
    def __init__(self, *, spec: RowSpec, repeat_index: int, profile: RunProfile,
                  run_dir: Path, device: str, offline_prior_artifact: Path | None,
                  project_root: Path | None = None,
                  startup_preflight_report: Mapping[str, Any] | None = None,
                  checkpoint_stride: int = 8):
        import torch
        from formal_experiments.ours.run_table2_real_smoke import build_runtime
        self.torch = torch; self.spec = spec; self.repeat_index = repeat_index; self.profile = profile
        self.run_dir = Path(run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        self.variant = {"rl_only": Table2Variant.RL_ONLY, "llm_rl": Table2Variant.LLM_RL,
                        "wm_rl": Table2Variant.WM_RL, "lwm_rl": Table2Variant.LWM_RL}[spec.component_variant]
        self.reward_mode = RewardMode({"full_reward": "Full-Reward", "delay_only": "Delay-Only", "fail_only": "Fail-Only"}[spec.reward_mode])
        self.runtime, self.frozen_artifacts = build_runtime(
            self.variant, seed=POLICY_SEEDS[repeat_index - 1], device=device,
            reward_mode=self.reward_mode, offline_prior_artifact=offline_prior_artifact)
        self.offline_prior_artifact = Path(offline_prior_artifact) if offline_prior_artifact is not None else None
        self.project_root = (Path(project_root) if project_root is not None
                             else Path(__file__).resolve().parents[2])
        self.startup_preflight_report = (dict(startup_preflight_report)
                                         if startup_preflight_report is not None else None)
        self.device, self.checkpoint_stride = device, int(checkpoint_stride)
        if self.checkpoint_stride <= 0: raise ValueError("checkpoint_stride must be positive")
        self.candidates: list[Path] = sorted((self.run_dir / "training_checkpoints").glob("checkpoint_*.pt"))
        self.validation_records: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []; self.decisions: list[dict[str, Any]] = []

    def preflight(self):
        report = self.startup_preflight_report
        if report is None:
            report = _job_preflight(
                project_root=self.project_root, spec=self.spec,
                offline_prior_artifact=self.offline_prior_artifact, device=self.device,
            )
        selected = report.get("selected_job") if isinstance(report, Mapping) else None
        if selected is not None and selected.get("canonical_id") != self.spec.canonical_id:
            raise RuntimeError("formal preflight report does not match the selected physical job")
        return {**dict(report), "frozen_artifacts": self.frozen_artifacts}
    def _save(self, index: int) -> Path:
        path = self.run_dir / "training_checkpoints" / f"checkpoint_{index:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.runtime.policy.state_dict(), path); self.candidates.append(path); return path
    def train(self, seeds):
        from formal_experiments.ours.run_table2_real_smoke import run_episode
        from formal_experiments.ours.variant_training import ppo_update
        updates = []
        curve_path = self.run_dir / "training_curve.jsonl"
        # A train stage is one transaction in ResumeJournal. Refuse to blend a
        # stale curve from an interrupted/unrelated invocation into this run.
        if curve_path.exists() and curve_path.stat().st_size:
            raise RuntimeError("training_curve.jsonl already exists before train stage")
        for index, seed in enumerate(seeds, 1):
            self.runtime.split = "train"
            buffer, _ = run_episode(self.runtime, episode_seed=seed, steps=self.profile.episode_ticks,
                                    split="train", train=True, reward_mode=self.reward_mode)
            update = ppo_update(self.runtime, buffer); updates.append(update)
            checkpoint = None
            if index % self.checkpoint_stride == 0 or index == len(seeds): checkpoint = self._save(index)
            record = {"schema": TRAINING_CURVE_SCHEMA,
                      "environment_steps": index * self.profile.episode_ticks,
                      "training_objective_reward": float(sum(item.reward for item in buffer.transitions)),
                      "episode_seed": int(seed), "repeat_index": self.repeat_index,
                      "policy_seed": POLICY_SEEDS[self.repeat_index - 1],
                      "ppo_update": update}
            if checkpoint is not None:
                record["checkpoint"] = {"path": str(checkpoint.relative_to(self.run_dir)),
                                        "sha256": sha256_file(checkpoint)}
            with curve_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return {"episodes": len(seeds), "updates": len(updates),
                "candidate_checkpoints": [str(path) for path in self.candidates]}
    def validation(self, seeds):
        from formal_experiments.ours.run_table2_real_smoke import run_episode
        records = []
        for epoch, path in enumerate(self.candidates, 1):
            self.runtime.policy.load_state_dict(self.torch.load(path, map_location=self.device, weights_only=True))
            score = 0.0
            self.runtime.split = "validation"
            for seed in seeds:
                buffer, _ = run_episode(self.runtime, episode_seed=seed, steps=self.profile.episode_ticks,
                                        split="validation", train=False, reward_mode=self.reward_mode)
                score += sum(item.reward for item in buffer.transitions)
            records.append({"epoch": epoch, "validation_score": score / len(seeds),
                            "checkpoint_sha256": sha256_file(path), "checkpoint_path": str(path)})
        self.validation_records = records; return records
    def hashes(self):
        from formal_experiments.ours.run_table2_real_smoke import policy_hash
        return {"policy": policy_hash(self.runtime.policy), **{key: value for key, value in self.frozen_artifacts.items()
                                                               if key.endswith("sha256")}}
    def test(self, seeds):
        from formal_experiments.ours.run_table2_real_smoke import run_episode
        if not getattr(self, "_test_checkpoint_loaded", False): self.prepare_test()
        self.runtime.split = "test"
        for seed in seeds:
            _buffer, decisions, episode = run_episode(
                self.runtime, episode_seed=seed, steps=self.profile.episode_ticks, split="test",
                train=False, reward_mode=self.reward_mode, return_formal_record=True)
            self.episodes.append(episode)
            self.decisions.extend({"episode_seed": seed, **item} for item in decisions)
        (self.run_dir / "episodes.partial.jsonl").write_text(
            "".join(json.dumps(x, sort_keys=True)+"\n" for x in self.episodes), encoding="utf-8")
        (self.run_dir / "decisions.partial.jsonl").write_text(
            "".join(json.dumps(x, sort_keys=True)+"\n" for x in self.decisions), encoding="utf-8")
        return {"episode_count": len(self.episodes), "decision_count": len(self.decisions), "decisions_audited": True}
    def prepare_test(self):
        selected = (select_validation_checkpoint(self.validation_records) if self.validation_records else
                    ResumeJournal(self.run_dir / "resume_state.json").load()["completed"]["validation"]["payload"]["selected"])
        self.runtime.policy.load_state_dict(self.torch.load(selected["checkpoint_path"], map_location=self.device, weights_only=True))
        self._test_checkpoint_loaded = True
    def integrity(self, _run_dir):
        from formal_experiments.evaluation.metrics_v3 import aggregate_repeat
        from formal_experiments.evaluation.validate_formal_run import validate_run_directory
        if not self.episodes and (self.run_dir / "episodes.partial.jsonl").is_file():
            self.episodes = [json.loads(line) for line in (self.run_dir / "episodes.partial.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if not self.decisions and (self.run_dir / "decisions.partial.jsonl").is_file():
            self.decisions = [json.loads(line) for line in (self.run_dir / "decisions.partial.jsonl").read_text(encoding="utf-8").splitlines() if line]
        selected = (select_validation_checkpoint(self.validation_records) if self.validation_records else
                    ResumeJournal(self.run_dir / "resume_state.json").load()["completed"]["validation"]["payload"]["selected"])
        checkpoint = self.run_dir / "checkpoint.pt"
        checkpoint.write_bytes(Path(selected["checkpoint_path"]).read_bytes())
        config = {**resolved_repeat_metadata(self.spec, repeat_index=self.repeat_index, profile=self.profile),
                  "train_seeds": list(self.profile.train_seeds), "validation_seeds": list(self.profile.validation_seeds),
                  "test_seeds": list(self.profile.test_seeds), "episode_ticks": self.profile.episode_ticks,
                  "orchestrator_version": ORCHESTRATOR_VERSION}
        (self.run_dir / "config.resolved.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (self.run_dir / "episodes.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in self.episodes), encoding="utf-8")
        (self.run_dir / "decisions.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in self.decisions), encoding="utf-8")
        metrics = aggregate_repeat(self.episodes, expected_ticks=self.profile.episode_ticks).to_dict()
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (self.run_dir / "stdout.log").touch()
        project = Path(__file__).resolve().parents[2]
        def git(*args):
            return subprocess.run(["git", *args], cwd=project, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", check=True).stdout.strip()
        dirty = bool(git("status", "--porcelain")); diff = git("diff", "--binary")
        paper = project.parent / "CC4_V3_0917_FINAL_EXPERIMENT_TASKBOOK.md"
        bound_names = ("config.resolved.yaml", "checkpoint.pt", "episodes.jsonl", "decisions.jsonl",
                       "metrics.json", "training_curve.jsonl")
        import CybORG
        cyborg_source = Path(CybORG.__file__)
        manifest = {"schema_version": 1, "protocol_version": "cc4_v3_20260917",
                    "method": f"{self.spec.component_variant} CC4 v3",
                    "method_slug": f"{self.spec.component_variant}_cc4", "repeat_index": self.repeat_index,
                    "training_seed": POLICY_SEEDS[self.repeat_index-1], "test_episode_seeds": list(self.profile.test_seeds),
                    "episode_ticks": self.profile.episode_ticks, "code_commit": git("rev-parse", "HEAD"),
                    "config_sha256": sha256_file(self.run_dir/"config.resolved.yaml"),
                    "paper_sha256": sha256_file(paper), "upstream_commit": None,
                    "python_version": platform.python_version(), "dependencies": {"python": platform.python_version(),
                    "torch": self.torch.__version__, "numpy": __import__("numpy").__version__,
                    "CybORG": {"source_sha256": sha256_file(cyborg_source)}},
                    "hardware": {"cpu": platform.processor(), "gpu": [], "selected_device": self.device},
                    "model_sha256": sha256_file(checkpoint), "run_mode": self.profile.mode,
                    "formal_result_eligible": self.profile.formal_result_eligible, "git_dirty": dirty,
                    "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "method_artifacts": self.frozen_artifacts,
                    "artifact_sha256": {name: sha256_file(self.run_dir/name) for name in bound_names}, **config}
        manifest["figure_artifacts"] = {
            "training_curve": _training_curve_figure_artifact(self.run_dir / "training_curve.jsonl")}
        if self.offline_prior_artifact is not None:
            manifest["ofox_cache_audit"] = _load_verified_ofox_cache_audit(self.offline_prior_artifact)
        if self.reward_mode == RewardMode.FAIL_ONLY:
            manifest["fail_only_training_contract"] = _derive_fail_only_training_contract(project)
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = validate_run_directory(self.run_dir, formal=self.profile.formal_result_eligible)
        if result.get("passed"):
            eligibility = json.loads((self.run_dir / "eligibility_report.json").read_text(encoding="utf-8"))
            verify_training_curve_bindings(self.run_dir, manifest, eligibility)
        return result

    def mapping(self):
        return {"preflight": self.preflight, "train": self.train, "validation": self.validation,
                "prepare_test": self.prepare_test, "test": self.test, "hashes": self.hashes,
                "integrity": self.integrity}


def resolved_repeat_metadata(spec: RowSpec, *, repeat_index: int, profile: RunProfile) -> dict[str, Any]:
    if repeat_index not in range(1, 6): raise ValueError("repeat_index must be 1..5")
    return {"table_id": spec.table_id, "row_id": spec.row_id, "reward_mode": spec.reward_mode,
            "component_variant": spec.component_variant, "reward_semantics": spec.reward_semantics,
            "canonical_run_id": spec.canonical_id, "repeat_index": repeat_index,
            "training_seed": POLICY_SEEDS[repeat_index - 1], "run_mode": profile.mode,
            "formal_result_eligible": profile.formal_result_eligible,
            "decision_audit_required": True, "offline_prior_miss_policy": "fail_closed",
            "test_artifacts_immutable": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--mode", choices=("formal", "dev"), default="formal")
    parser.add_argument("--dev-ticks", type=int, default=20)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--table-id", choices=("table2", "table3"))
    parser.add_argument("--row-id")
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline-prior-artifact", type=Path)
    args = parser.parse_args()
    profile = RunProfile() if args.mode == "formal" else RunProfile("dev", (1000,), (2000,), (3200,), args.dev_ticks)
    selected_spec = None
    if bool(args.table_id) != bool(args.row_id):
        parser.error("--table-id and --row-id must be supplied together")
    if args.table_id:
        matches = [x for x in row_specs() if x.table_id == args.table_id and x.row_id == args.row_id]
        if len(matches) != 1: parser.error("unknown table/row combination")
        selected_spec = matches[0]
    if args.offline_prior_artifact is not None and not args.offline_prior_artifact.is_absolute():
        args.offline_prior_artifact = args.project_root / args.offline_prior_artifact
    report = preflight(project_root=args.project_root, profile=profile, selected_spec=selected_spec,
                       offline_prior_artifact=args.offline_prior_artifact, device=args.device)
    report["operation"] = "execute" if args.execute else "dry_run_preflight_only"
    if args.execute:
        if not args.table_id or not args.row_id or not args.out:
            parser.error("--execute requires --table-id, --row-id, and --out")
        assert selected_spec is not None
        if selected_spec.table_id == "table3" and selected_spec.row_id == "full_reward":
            parser.error("Table3 Full-Reward aliases canonical Table2 lwm_rl; duplicate execution is forbidden")
        # Each physical job is independently launchable, but its own frozen
        # dependencies always fail closed in both formal and development mode.
        if report["status"] != "PASS":
            print(json.dumps(report, indent=2)); return 2
        callbacks = RealCC4ProductionCallbacks(spec=selected_spec, repeat_index=args.repeat_index,
                                               profile=profile, run_dir=args.out, device=args.device,
                                               offline_prior_artifact=args.offline_prior_artifact,
                                               project_root=args.project_root,
                                               startup_preflight_report=report)
        state = FormalStageRunner(spec=selected_spec, repeat_index=args.repeat_index, profile=profile,
                                  run_dir=args.out, callbacks=callbacks.mapping()).run()
        report["resume_state"] = state
        (args.out / "orchestrator_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
