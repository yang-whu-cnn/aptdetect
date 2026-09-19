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
TABLE_ROWS = {
    "table1": (
        "uamcts_cc4", "rsmbrl_cc4", "carl_cc4", "dca_cc4",
        "priorrl_cc4", "terla_a4", "lwm_rl",
    ),
    "table2": ("rl_only", "llm_rl", "wm_rl", "lwm_rl"),
    "table3": ("delay_only", "fail_only", "full_reward"),
}
ROW_LABELS = {
    "uamcts_cc4": "UAMCTS-CC4", "rsmbrl_cc4": "RSMBRL-CC4",
    "carl_cc4": "CARL-CC4", "dca_cc4": "DCA-CC4 (adapted)",
    "priorrl_cc4": "PriorRL-CC4", "terla_a4": "TERLA-A4", "lwm_rl": "LWM-RL",
    "rl_only": "RL-Only", "llm_rl": "LLM-RL", "wm_rl": "WM-RL",
    "delay_only": "Delay-Only", "fail_only": "Fail-Only", "full_reward": "Full-Reward",
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
    if table_id == "table2" and (reward_mode != "full_reward" or component != row_id or semantics != FULL_REWARD_SEMANTICS):
        raise FigureInputError("Table2 requires row-matched component, full_reward, and frozen Full-Reward semantics")
    if table_id == "table3" and (component != "lwm_rl" or reward_mode != row_id):
        raise FigureInputError("Table3 requires component_variant=lwm_rl and row-matched reward_mode")
    return table_id, row_id, reward_mode, component, semantics


def _bound_optional_jsonl(run_dir: Path, manifest: Mapping[str, Any], name: str) -> Path | None:
    declaration = manifest.get("figure_artifacts", {}).get(name)
    if declaration is None:
        return None
    if not isinstance(declaration, Mapping):
        raise FigureInputError(f"{run_dir}: figure_artifacts.{name} must be an object")
    relative = declaration.get("path")
    expected = declaration.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise FigureInputError(f"{run_dir}: {name} needs path and sha256")
    path = (run_dir / relative).resolve()
    if run_dir.resolve() not in path.parents or not path.is_file():
        raise FigureInputError(f"{run_dir}: {name} path escapes repeat or is missing")
    if _sha256(path) != expected:
        raise FigureInputError(f"{run_dir}: {name} sha256 mismatch")
    return path


def discover_formal_runs(input_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(input_root).resolve()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        run_dir = manifest_path.parent
        reason = None
        try:
            if _contains_banned_marker(run_dir.relative_to(root)):
                raise FigureInputError("path contains a non-formal marker")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise FigureInputError("manifest must be an object")
            if manifest.get("run_mode") != "formal" or manifest.get("formal_result_eligible") is not True:
                raise FigureInputError("manifest is development or formal_result_eligible is not true")
            table_id, row_id, reward_mode, component, reward_semantics = _table_identity(manifest)
            _report, eligibility_path = _eligibility_report(run_dir)
            validation = validate_run_directory(run_dir, formal=True)
            if not validation["passed"]:
                raise FigureInputError("formal validation failed: " + "; ".join(validation["errors"]))
            episodes_path = run_dir / "episodes.jsonl"
            metrics_path = run_dir / "metrics.json"
            episodes = load_jsonl(episodes_path)
            curve_path = _bound_optional_jsonl(run_dir, manifest, "training_curve")
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
            rejected.append({"manifest": str(manifest_path.relative_to(root)), "reason": reason})
    return accepted, rejected


def _group_rows(runs: Iterable[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[str]]:
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
    return dict(sorted(complete.items())), issues


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
    for method, runs in methods.items():
        rows = []
        for run in runs:
            path = run["training_curve_path"]
            if path is None:
                raise FigureInputError(f"{method}: missing hash-bound training_curve artifact")
            curve = load_jsonl(path)
            points = [(int(item["environment_steps"]), _finite(item["training_objective_reward"], "training_objective_reward")) for item in curve]
            if not points or any(step < 0 for step, _ in points) or len({step for step, _ in points}) != len(points):
                raise FigureInputError(f"{method}: invalid training curve steps")
            rows.append(dict(points))
        common = sorted(set.intersection(*(set(row) for row in rows)))
        if len(common) < 2:
            raise FigureInputError(f"{method}: fewer than two common training curve steps")
        curves[method] = (np.asarray(common), np.asarray([[row[x] for x in common] for row in rows]))
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
    runs, rejected = discover_formal_runs(source); methods, row_issues = _group_rows(runs)
    figures = {}
    for name in FIGURE_ORDER:
        try:
            artifacts = PLOTTERS[name](methods, out)
            figures[name] = {"status": "PASS", "artifacts": artifacts}
        except FigureInputError as exc:
            figures[name] = {"status": "BLOCKED", "reason": str(exc)}
    inputs = {}
    for run in runs: inputs.update(run["inputs"])
    passed_count = sum(row["status"] == "PASS" for row in figures.values())
    status = "PASS" if passed_count == len(FIGURE_ORDER) else ("PARTIAL" if passed_count else "BLOCKED")
    report = {
        "schema": "cc4_v3_formal_paper_figures_v1",
        "status": status,
        "formal_inputs_only": True,
        "input_root": str(source),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_repeat_count": len(runs),
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
