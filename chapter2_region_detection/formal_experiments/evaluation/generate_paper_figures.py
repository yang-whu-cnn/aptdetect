"""Fail-closed, provenance-bound paper figures for formal CC4-v3 results only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from formal_experiments.common.run_manifest import load_jsonl
from formal_experiments.evaluation.metrics_v3 import aggregate_repeat, compute_episode_metrics
from formal_experiments.evaluation.validate_formal_run import validate_run_directory


BANNED_PATH_MARKERS = ("pilot", "development", "smoke", "provisional", "blocked")
ELIGIBILITY_SCHEMA = "cc4_v3_eligibility_report_v1"
ELIGIBILITY_VALIDATOR = "formal_experiments.evaluation.validate_formal_run"
FULL_REWARD_SEMANTICS = "cc4_v3_full_reward"
FULL_REWARD_ALIAS_SOURCE = ("table2", "lwm_rl")
FULL_REWARD_ALIAS_TARGETS = (
    ("table1", "lwm_rl"),
    ("table3", "full_reward"),
)
FULL_REWARD_PHYSICAL_REJECTION = (
    "table1/lwm_rl and table3/full_reward must be read-only aliases of table2/lwm_rl; "
    "a separate physical run is forbidden"
)
TRAINING_CURVE_SCHEMA = "cc4_v3_training_curve_v1"
FAIL_ONLY_CHECKPOINT_SHA256 = "8800a65d4cd4f540ca6d5fa5bcc0f7e87a913403c4cded5beed931e91bb0b3b2"
FAIL_ONLY_MANIFEST_SHA256 = "47c2644c6b5948157d049c4cbc035a16762000b4991385f9172780552ab056ef"
OFOX_ENTRY_SET_SHA256 = "7941f6f11d47989265955bdc6caff950de97a9f5b9562afa8f7cb5e0209a7e10"
OFOX_LOGICAL_MANIFEST_SHA256 = "b981a125f162e8d59bd3c9c5b0f4d53871f0b3d9938305f2e5a6a60c2d2fe0e0"
TABLE_ROWS = {
    "table1": (
        "uamcts_cc4", "rsmbrl_cc4", "carl_cc4", "dca_cc4",
        "priorrl_ppo_cc4", "terla_a4", "lwm_rl",
    ),
    "table2": ("rl_only", "llm_rl", "wm_rl", "lwm_rl"),
    "table3": ("delay_only", "fail_only", "full_reward"),
}
ROW_LABELS = {
    "uamcts_cc4": "UAMCTS-CC4 (adapted)", "rsmbrl_cc4": "RSMBRL-CC4",
    "carl_cc4": "CARL-CC4 (adapted)", "dca_cc4": "DCA-CC4 (adapted)",
    "priorrl_ppo_cc4": "PriorRL-PPO-CC4", "terla_a4": "TERLA-A4", "lwm_rl": "LWM-RL",
    "rl_only": "RL-Only", "llm_rl": "LLM-RL", "wm_rl": "WM-RL",
    "delay_only": "Delay-Only", "fail_only": "Fail-Only", "full_reward": "Full-Reward",
}
TABLE1_IDENTITIES = {
    "uamcts_cc4": ("full_reward", "cc4_v3_full_reward"),
    "rsmbrl_cc4": ("full_reward", "cc4_v3_full_reward"),
    "carl_cc4": ("caics_reward", "standard_caics_cc4_mapping"),
    "dca_cc4": ("not_applicable", "fixed_response_mapping_no_learning_reward"),
    "priorrl_ppo_cc4": ("full_reward", "cc4_v3_full_reward"),
    "terla_a4": ("terla_cyber_reward", "negative_red_sessions_plus_service_unreliability_ot"),
}
FIGURE_ORDER = (
    "table1_method_comparison",
    "training_learning_curve",
    "recovery_survival",
    "component_reward_ablation",
    "efficiency_tradeoff",
)
COLORS = ("#2864DC", "#E4572E", "#3A9D23", "#8E5BB7", "#D89B00", "#437C90", "#AA4465")


class FigureInputError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise FigureInputError(f"{field} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureInputError(f"{field} must be finite numeric") from exc
    if not isfinite(number):
        raise FigureInputError(f"{field} must be finite numeric")
    return number


def _contains_banned_marker(path: Path) -> bool:
    for part in path.parts:
        lowered = part.lower()
        if any(marker in lowered for marker in BANNED_PATH_MARKERS):
            return True
        if lowered == "dev" or lowered.startswith("dev_") or lowered.endswith("_dev"):
            return True
    return False


def _eligibility_report(run_dir: Path) -> tuple[dict[str, Any], Path]:
    path = run_dir / "eligibility_report.json"
    if not path.is_file():
        raise FigureInputError("missing validator-generated eligibility_report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FigureInputError("eligibility_report.json must be an object")
    if payload.get("schema") != ELIGIBILITY_SCHEMA:
        raise FigureInputError("eligibility report schema mismatch")
    if payload.get("validator") != ELIGIBILITY_VALIDATOR:
        raise FigureInputError("eligibility report validator identity mismatch")
    statuses = [payload[key] for key in ("status", "eligibility") if key in payload]
    if not statuses or any(str(value).upper() != "PASS" for value in statuses):
        raise FigureInputError("validator eligibility status must be PASS")
    if payload.get("passed") is not True:
        raise FigureInputError("validator eligibility report passed must be true")
    bindings = payload.get("input_sha256", payload.get("inputs"))
    if not isinstance(bindings, Mapping):
        raise FigureInputError("eligibility report requires input_sha256 bindings")
    for filename in ("manifest.json", "episodes.jsonl", "metrics.json"):
        expected = bindings.get(filename)
        target = run_dir / filename
        if not isinstance(expected, str) or not target.is_file() or _sha256(target) != expected:
            raise FigureInputError(f"eligibility input binding mismatch for {filename}")
    return dict(payload), path


def _table_identity(manifest: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    required = ("table_id", "row_id", "reward_mode", "component_variant", "reward_semantics")
    missing = [field for field in required if not isinstance(manifest.get(field), str) or not manifest[field].strip()]
    if missing:
        raise FigureInputError(f"missing/non-string frozen figure schema fields: {missing}")
    table_id = manifest["table_id"].lower(); row_id = manifest["row_id"].lower()
    reward_mode = manifest["reward_mode"].lower(); component = manifest["component_variant"].lower()
    semantics = manifest["reward_semantics"].lower()
    if table_id not in TABLE_ROWS or row_id not in TABLE_ROWS[table_id]:
        raise FigureInputError(f"row {table_id}/{row_id} is outside frozen whitelist")
    if (table_id, row_id) in FULL_REWARD_ALIAS_TARGETS:
        raise FigureInputError(FULL_REWARD_PHYSICAL_REJECTION)
    if table_id == "table1":
        expected = TABLE1_IDENTITIES.get(row_id)
        if (expected is None or manifest.get("method_slug") != row_id
                or component != row_id or (reward_mode, semantics) != expected):
            raise FigureInputError(
                "Table1 row_id, method_slug, component_variant, reward_mode, and reward_semantics "
                "must match the frozen method identity"
            )
    if table_id == "table2" and (reward_mode != "full_reward" or component != row_id or semantics != FULL_REWARD_SEMANTICS):
        raise FigureInputError("Table2 requires row-matched component, full_reward, and frozen Full-Reward semantics")
    if table_id == "table3" and (component != "lwm_rl" or reward_mode != row_id):
        raise FigureInputError("Table3 requires component_variant=lwm_rl and row-matched reward_mode")
    if table_id == "table3" and row_id in ("delay_only", "fail_only"):
        expected_semantics = f"cc4_v3_{row_id}"
        if semantics != expected_semantics:
            raise FigureInputError(
                f"Table3 {row_id} requires reward_semantics={expected_semantics}"
            )
    if (table_id, row_id) == ("table3", "fail_only"):
        _validate_fail_only_contract(manifest)
    return table_id, row_id, reward_mode, component, semantics


def _validate_fail_only_contract(manifest: Mapping[str, Any]) -> None:
    training = manifest.get("fail_only_training_contract")
    expected_training = {
        "loss_name": "smooth_l1",
        "beta": 1.0,
        "legacy_mse": "diagnostic_only",
        "gate_pass": True,
        "train_seed_count": 32,
        "validation_seed_count": 8,
        "train_validation_overlap": 0,
        "test_leak_count": 0,
        "checkpoint_sha256": FAIL_ONLY_CHECKPOINT_SHA256,
        "manifest_sha256": FAIL_ONLY_MANIFEST_SHA256,
    }
    if not isinstance(training, Mapping) or any(
        training.get(field) != value for field, value in expected_training.items()
    ):
        raise FigureInputError("Table3 fail_only approved Huber training contract mismatch")
    cache = manifest.get("ofox_cache_audit")
    expected_cache = {
        "provider": "ofox",
        "provider_calls": 6,
        "generated": 6,
        "verified_entries": 6,
        "failed": 0,
        "d27_exact_count": 6,
        "split": "train_only",
        "coverage_pass": True,
        "online_calls_allowed": False,
        "entry_set_sha256": OFOX_ENTRY_SET_SHA256,
        "logical_manifest_sha256": OFOX_LOGICAL_MANIFEST_SHA256,
    }
    if not isinstance(cache, Mapping) or any(
        cache.get(field) != value for field, value in expected_cache.items()
    ):
        raise FigureInputError("Table3 fail_only frozen OFOX cache contract mismatch")


def _bound_optional_jsonl(
    run_dir: Path,
    manifest: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    name: str,
) -> Path | None:
    declarations = manifest.get("figure_artifacts", {})
    if not isinstance(declarations, Mapping):
        raise FigureInputError(f"{run_dir}: figure_artifacts must be an object")
    declaration = declarations.get(name)
    if declaration is None:
        return None
    if not isinstance(declaration, Mapping):
        raise FigureInputError(f"{run_dir}: figure_artifacts.{name} must be an object")
    relative = declaration.get("path")
    expected = declaration.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative.strip()
        or not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise FigureInputError(f"{run_dir}: {name} needs path and sha256")
    path = (run_dir / relative).resolve()
    if run_dir.resolve() not in path.parents or not path.is_file():
        raise FigureInputError(f"{run_dir}: {name} path escapes repeat or is missing")
    if _sha256(path) != expected:
        raise FigureInputError(f"{run_dir}: {name} sha256 mismatch")
    artifact_bindings = manifest.get("artifact_sha256")
    if not isinstance(artifact_bindings, Mapping) or artifact_bindings.get(relative) != expected:
        raise FigureInputError(
            f"{run_dir}: {name} must have the same sha256 binding in artifact_sha256"
        )
    eligibility_bindings = eligibility.get("input_sha256", eligibility.get("inputs"))
    if not isinstance(eligibility_bindings, Mapping) or eligibility_bindings.get(relative) != expected:
        raise FigureInputError(
            f"{run_dir}: {name} must have the same sha256 binding in eligibility_report"
        )
    if name == "training_curve":
        contract = {
            "schema": TRAINING_CURVE_SCHEMA,
            "split": "train",
            "x_field": "environment_steps",
            "y_field": "training_objective_reward",
        }
        mismatches = [field for field, value in contract.items() if declaration.get(field) != value]
        count = declaration.get("record_count")
        if mismatches or isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise FigureInputError(
                f"{run_dir}: training_curve declaration violates {TRAINING_CURVE_SCHEMA}"
            )
    return path


def discover_formal_runs(input_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(input_root).resolve()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        run_dir = manifest_path.parent
        reason = None
        claimed_formal = False
        claimed_identity: str | None = None
        try:
            if _contains_banned_marker(run_dir.relative_to(root)):
                raise FigureInputError("path contains a non-formal marker")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise FigureInputError("manifest must be an object")
            claimed_formal = (
                manifest.get("run_mode") == "formal"
                and manifest.get("formal_result_eligible") is True
            )
            raw_table = str(manifest.get("table_id", "")).lower()
            raw_row = str(manifest.get("row_id", "")).lower()
            if raw_table in TABLE_ROWS and raw_row in TABLE_ROWS[raw_table]:
                claimed_identity = f"{raw_table}/{raw_row}"
            if manifest.get("run_mode") != "formal" or manifest.get("formal_result_eligible") is not True:
                raise FigureInputError("manifest is development or formal_result_eligible is not true")
            table_id, row_id, reward_mode, component, reward_semantics = _table_identity(manifest)
            report, eligibility_path = _eligibility_report(run_dir)
            validation = validate_run_directory(run_dir, formal=True)
            if not validation["passed"]:
                raise FigureInputError("formal validation failed: " + "; ".join(validation["errors"]))
            episodes_path = run_dir / "episodes.jsonl"
            metrics_path = run_dir / "metrics.json"
            episodes = load_jsonl(episodes_path)
            curve_path = _bound_optional_jsonl(run_dir, manifest, report, "training_curve")
            accepted.append({
                "run_dir": run_dir,
                "manifest": dict(manifest),
                "episodes": episodes,
                "metrics": validation["metrics"],
                "training_curve_path": curve_path,
                "table_id": table_id,
                "row_id": row_id,
                "reward_mode": reward_mode,
                "component_variant": component,
                "reward_semantics": reward_semantics,
                "inputs": {
                    str(manifest_path.relative_to(root)): _sha256(manifest_path),
                    str(episodes_path.relative_to(root)): _sha256(episodes_path),
                    str(metrics_path.relative_to(root)): _sha256(metrics_path),
                    str(eligibility_path.relative_to(root)): _sha256(eligibility_path),
                    **({str(curve_path.relative_to(root)): _sha256(curve_path)} if curve_path else {}),
                },
            })
        except (FigureInputError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reason = str(exc)
        if reason is not None:
            rejected.append({
                "manifest": str(manifest_path.relative_to(root)),
                "reason": reason,
                "claimed_formal": claimed_formal,
                "claimed_identity": claimed_identity,
            })
    return accepted, rejected


def _full_reward_alias(
    source_rows: list[dict[str, Any]], target: tuple[str, str],
) -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    for run in source_rows:
        manifest = run["manifest"]
        identity = (
            str(manifest.get("table_id", "")).lower(),
            str(manifest.get("row_id", "")).lower(),
            str(manifest.get("reward_mode", "")).lower(),
            str(manifest.get("component_variant", "")).lower(),
            str(manifest.get("reward_semantics", "")).lower(),
        )
        if identity != (
            "table2", "lwm_rl", "full_reward", "lwm_rl", FULL_REWARD_SEMANTICS
        ):
            raise FigureInputError(
                "LWM-RL alias source must be the frozen table2/lwm_rl Full-Reward row"
            )
        canonical_run_id = manifest.get("canonical_run_id")
        if canonical_run_id != "lwm_full":
            raise FigureInputError(
                "LWM-RL alias source canonical_run_id must be lwm_full"
            )
        target_table, target_row = target
        alias = dict(run)
        alias.update({
            "table_id": target_table,
            "row_id": target_row,
            "reward_mode": "full_reward",
            "component_variant": "lwm_rl",
            "reward_semantics": FULL_REWARD_SEMANTICS,
            "alias_of": "table2/lwm_rl",
        })
        aliases.append(alias)
    return aliases


def group_formal_rows(
    runs: Iterable[dict[str, Any]],
    rejected: Iterable[Mapping[str, Any]] = (),
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["table_id"], run["row_id"])].append(run)
    complete: dict[tuple[str, str], list[dict[str, Any]]] = {}; issues = []
    for identity, rows in grouped.items():
        indices = [row["manifest"]["repeat_index"] for row in rows]
        if indices == [1, 2, 3, 4, 5]:
            complete[identity] = rows
        elif sorted(indices) == [1, 2, 3, 4, 5]:
            complete[identity] = sorted(rows, key=lambda row: row["manifest"]["repeat_index"])
        else:
            issues.append(f"{identity[0]}/{identity[1]} repeat indices must be exactly 1..5 once; got {sorted(indices)}")
    tainted = {
        tuple(str(item["claimed_identity"]).split("/", 1))
        for item in rejected
        if item.get("claimed_formal") and item.get("claimed_identity")
    }
    for identity in sorted(tainted):
        complete.pop(identity, None)
        issues.append(
            f"{identity[0]}/{identity[1]} has a rejected formal candidate; row selection is forbidden"
        )
    source = complete.get(FULL_REWARD_ALIAS_SOURCE)
    for target in FULL_REWARD_ALIAS_TARGETS:
        if target in tainted:
            issues.append(FULL_REWARD_PHYSICAL_REJECTION)
        elif source is not None:
            try:
                complete[target] = _full_reward_alias(source, target)
            except FigureInputError as exc:
                issues.append(str(exc))
    return dict(sorted(complete.items())), issues


def _group_rows(runs: Iterable[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[str]]:
    """Backward-compatible wrapper for callers that have no rejection list."""
    return group_formal_rows(runs)


def _require_table(rows, table_id: str) -> dict[str, list[dict[str, Any]]]:
    missing = [row for row in TABLE_ROWS[table_id] if (table_id, row) not in rows]
    if missing:
        raise FigureInputError(f"{table_id} missing complete frozen rows: {missing}")
    return {row: rows[(table_id, row)] for row in TABLE_ROWS[table_id]}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "svg.hashsalt": "cc4-v3-paper-figures",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out: Path, stem: str) -> dict[str, str]:
    svg = out / f"{stem}.svg"; png = out / f"{stem}.png"
    metadata = {"Creator": "CC4-v3 deterministic formal figure pipeline", "Date": None}
    fig.savefig(svg, bbox_inches="tight", metadata=metadata)
    fig.savefig(png, bbox_inches="tight", metadata={"Software": metadata["Creator"]})
    return {"svg": svg.name, "svg_sha256": _sha256(svg),
            "png": png.name, "png_sha256": _sha256(png)}


def _method_metric_values(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([_finite(row["metrics"][name], name) for row in rows], dtype=np.float64)


def plot_table1(rows: Mapping[tuple[str, str], list[dict[str, Any]]], out: Path) -> dict[str, str]:
    methods = _require_table(rows, "table1")
    plt = _style()
    metrics = (
        ("cc4_official_reward", "Official reward", False),
        ("operation_failure_penalty", "Operation-failure penalty", True),
        ("recovery_precision", "Recovery precision", False),
        ("recovery_time_censored_mean", "Recovery time (ticks)", True),
    )
    names = list(methods); labels = [ROW_LABELS[name] for name in names]; y = np.arange(len(names)); fig, axes = plt.subplots(1, 4, figsize=(12.4, max(2.3, .42*len(names)+1.2)))
    for ax, (field, label, lower_better) in zip(axes, metrics):
        values = [_method_metric_values(methods[name], field) for name in names]
        means = [float(v.mean()) for v in values]; stds = [float(v.std(ddof=1)) for v in values]
        ax.errorbar(means, y, xerr=stds, fmt="o", color=COLORS[0], capsize=3, lw=1.2)
        ax.set_title(label + (" ↓" if lower_better else " ↑")); ax.grid(axis="x", alpha=.22)
        ax.set_yticks(y, labels if ax is axes[0] else [""]*len(names)); ax.invert_yaxis()
    fig.suptitle("CC4-v3 formal method comparison (mean ± sample SD; n=5)", y=1.02)
    result = _save(fig, out, "table1_method_comparison"); plt.close(fig); return result


def _load_curves(methods: Mapping[str, list[dict[str, Any]]]):
    curves = {}
    reference_steps: tuple[int, ...] | None = None
    for method, runs in methods.items():
        rows = []
        for run in runs:
            path = run["training_curve_path"]
            if path is None:
                raise FigureInputError(f"{method}: missing hash-bound training_curve artifact")
            try:
                curve = load_jsonl(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise FigureInputError(f"{method}: invalid training curve JSONL: {exc}") from exc
            declaration = run["manifest"]["figure_artifacts"]["training_curve"]
            if len(curve) != declaration["record_count"]:
                raise FigureInputError(f"{method}: training curve record_count mismatch")
            points = []
            for index, item in enumerate(curve):
                step = item.get("environment_steps")
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise FigureInputError(
                        f"{method}: training curve step[{index}] must be a non-negative integer"
                    )
                points.append((step, _finite(
                    item.get("training_objective_reward"),
                    f"{method}.training_objective_reward[{index}]",
                )))
            if len(points) < 2 or any(
                points[index][0] >= points[index + 1][0]
                for index in range(len(points) - 1)
            ):
                raise FigureInputError(f"{method}: training curve steps must be strictly increasing")
            rows.append(dict(points))
        steps = tuple(rows[0])
        if any(tuple(row) != steps for row in rows[1:]):
            raise FigureInputError(f"{method}: repeat training curves must use the same step grid")
        if reference_steps is None:
            reference_steps = steps
        elif steps != reference_steps:
            raise FigureInputError("Table2 methods must use the same training curve step grid")
        curves[method] = (np.asarray(steps), np.asarray([[row[x] for x in steps] for row in rows]))
    return curves


def plot_learning_curves(methods, out):
    comparable = _require_table(methods, "table2")
    semantics = {run["reward_semantics"] for runs in comparable.values() for run in runs}
    if semantics != {FULL_REWARD_SEMANTICS}:
        raise FigureInputError(f"learning curves require one shared Full-Reward semantics, got {sorted(semantics)}")
    curves = _load_curves(comparable); plt = _style(); fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for index, (method, (steps, values)) in enumerate(curves.items()):
        mean = values.mean(0); half = 2.776 * values.std(0, ddof=1) / sqrt(5)
        ax.plot(steps, mean, label=ROW_LABELS[method], color=COLORS[index % len(COLORS)], lw=1.7)
        ax.fill_between(steps, mean-half, mean+half, color=COLORS[index % len(COLORS)], alpha=.18)
    ax.set(xlabel="Environment transitions", ylabel="Training-objective reward", title="Formal training learning curves (mean and 95% t-CI; n=5)")
    ax.grid(alpha=.22); ax.legend(frameon=False, ncol=2)
    result = _save(fig, out, "training_learning_curve"); plt.close(fig); return result


def _kaplan_meier(episodes: Iterable[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    observations = []
    for episode in episodes:
        end = int(episode["episode_end_tick"])
        for incident in episode["incidents"]:
            start = int(incident["t_compromise"]); recovered = incident.get("t_recovered")
            observations.append(((int(recovered)-start) if recovered is not None else (end-start), recovered is not None))
    if not observations:
        raise FigureInputError("recovery survival requires at least one incident")
    at_risk = len(observations); survival = 1.0; xs = [0.0]; ys = [1.0]
    for time in sorted({x[0] for x in observations}):
        events = sum(t == time and event for t, event in observations)
        censored = sum(t == time and not event for t, event in observations)
        if events:
            survival *= 1.0 - events / at_risk; xs.extend([float(time), float(time)]); ys.extend([ys[-1], survival])
        at_risk -= events + censored
    return np.asarray(xs), np.asarray(ys)


def plot_recovery_survival(methods, out):
    methods = _require_table(methods, "table1")
    plt = _style(); fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for index, (method, runs) in enumerate(methods.items()):
        x, y = _kaplan_meier(ep for run in runs for ep in run["episodes"])
        ax.step(x, y, where="post", label=ROW_LABELS[method], color=COLORS[index % len(COLORS)], lw=1.7)
    ax.set(xlabel="Ticks since compromise", ylabel="Probability not yet recovered", ylim=(0, 1.02), title="Recovery-time survival curves (right-censored)")
    ax.grid(alpha=.22); ax.legend(frameon=False, ncol=2)
    result = _save(fig, out, "recovery_survival"); plt.close(fig); return result


def plot_ablation(methods, out):
    table2 = _require_table(methods, "table2"); table3 = _require_table(methods, "table3")
    plt = _style(); fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for ax, table, title in ((axes[0], table2, "Component ablation"), (axes[1], table3, "Reward ablation")):
        row_ids = list(table); labels = [ROW_LABELS[row] for row in row_ids]
        values = [_method_metric_values(table[row], "cc4_official_reward") for row in row_ids]
        means = [v.mean() for v in values]; errors = [v.std(ddof=1) for v in values]
        ax.bar(np.arange(len(labels)), means, yerr=errors, capsize=3, color=COLORS[:len(labels)])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right"); ax.set_title(title); ax.set_ylabel("Official reward ↑"); ax.grid(axis="y", alpha=.22)
    result = _save(fig, out, "component_reward_ablation"); plt.close(fig); return result


def _episode_latency(episode: Mapping[str, Any]) -> float:
    values = episode.get("decision_latency_ms")
    if isinstance(values, list) and values:
        return float(np.mean([_finite(x, "decision_latency_ms") for x in values]))
    total = episode.get("decision_latency_ms_sum"); count = episode.get("decision_latency_count")
    if total is not None and isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return _finite(total, "decision_latency_ms_sum") / count
    raise FigureInputError("episode lacks decision latency samples or sum/count")


def plot_efficiency(methods, out):
    methods = _require_table(methods, "table1")
    plt = _style(); fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for index, (method, runs) in enumerate(methods.items()):
        repeat_latency = np.asarray([np.mean([_episode_latency(ep) for ep in run["episodes"]]) for run in runs])
        reward = _method_metric_values(runs, "cc4_official_reward")
        ax.errorbar(repeat_latency.mean(), reward.mean(), xerr=repeat_latency.std(ddof=1), yerr=reward.std(ddof=1), fmt="o", capsize=3, label=ROW_LABELS[method], color=COLORS[index % len(COLORS)])
    ax.set(xlabel="Decision latency (ms; lower is better)", ylabel="Official reward (higher is better)", title="Decision latency–effect trade-off (mean ± sample SD; n=5)")
    ax.grid(alpha=.22); ax.legend(frameon=False, ncol=2)
    result = _save(fig, out, "efficiency_tradeoff"); plt.close(fig); return result


PLOTTERS = {
    "table1_method_comparison": plot_table1,
    "training_learning_curve": plot_learning_curves,
    "recovery_survival": plot_recovery_survival,
    "component_reward_ablation": plot_ablation,
    "efficiency_tradeoff": plot_efficiency,
}


def generate(input_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(input_root).resolve(); out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    # A blocked re-run must not leave an older eligible-looking figure behind.
    for name in FIGURE_ORDER:
        for suffix in (".svg", ".png"):
            stale = out / f"{name}{suffix}"
            if stale.is_file():
                stale.unlink()
    runs, rejected = discover_formal_runs(source); methods, row_issues = group_formal_rows(runs, rejected)
    figures = {}
    expected_rows = {(table_id, row_id) for table_id, row_ids in TABLE_ROWS.items() for row_id in row_ids}
    publication_ready = set(methods) == expected_rows and not row_issues
    if publication_ready:
        for name in FIGURE_ORDER:
            try:
                artifacts = PLOTTERS[name](methods, out)
                figures[name] = {"status": "PASS", "artifacts": artifacts}
            except FigureInputError as exc:
                figures[name] = {"status": "BLOCKED", "reason": str(exc)}
        failures = {
            name: row["reason"] for name, row in figures.items()
            if row["status"] != "PASS"
        }
        if failures:
            for name in FIGURE_ORDER:
                for suffix in (".svg", ".png"):
                    artifact = out / f"{name}{suffix}"
                    if artifact.is_file():
                        artifact.unlink()
            reason = f"atomic five-figure publication blocked: {failures}"
            figures = {name: {"status": "BLOCKED", "reason": reason} for name in FIGURE_ORDER}
    else:
        missing = [f"{table_id}/{row_id}" for table_id, row_id in sorted(expected_rows - set(methods))]
        reason = "all three formal tables must pass before any paper figure is generated"
        if missing:
            reason += f"; missing rows: {missing}"
        if row_issues:
            reason += f"; row issues: {row_issues}"
        figures = {name: {"status": "BLOCKED", "reason": reason} for name in FIGURE_ORDER}
    inputs = {}
    for run in runs: inputs.update(run["inputs"])
    passed_count = sum(row["status"] == "PASS" for row in figures.values())
    if passed_count == len(FIGURE_ORDER):
        status = "PASS"
    elif methods:
        status = "PARTIAL"
    else:
        status = "BLOCKED"
    report = {
        "schema": "cc4_v3_formal_paper_figures_v1",
        "status": status,
        "formal_inputs_only": True,
        "input_root": str(source),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_repeat_count": len(runs),
        "logical_repeat_count": sum(len(group) for group in methods.values()),
        "complete_rows": [f"{table_id}/{row_id}" for table_id, row_id in methods],
        "row_completeness_issues": row_issues,
        "rejected_inputs": rejected,
        "input_sha256": dict(sorted(inputs.items())),
        "figures": figures,
        "style": {"colors": COLORS, "font": "DejaVu Sans", "png_dpi": 300, "svg_hashsalt": "cc4-v3-paper-figures"},
    }
    path = out / "figure_manifest.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); report = generate(args.input_root, args.output_dir)
    print(json.dumps({"status": report["status"], "figures": {k: v["status"] for k, v in report["figures"].items()}}, sort_keys=True))
    if report["status"] != "PASS": raise SystemExit(2)


if __name__ == "__main__": main()
