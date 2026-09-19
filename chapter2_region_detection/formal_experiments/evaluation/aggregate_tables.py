"""Five-repeat aggregation and auditable formal paper-table artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from formal_experiments.common.run_manifest import POLICY_SEEDS


TABLE_METRICS = (
    "cc4_official_reward",
    "operation_failure_penalty",
    "recovery_precision",
    "recovery_time_censored_mean",
)
AUXILIARY_METRICS = (
    "recovery_time_completed_only_mean",
    "recovery_unrecovered_rate",
)
AGGREGATION_SCHEMA = "cc4_v3_formal_table_aggregation_v1"
AGGREGATOR_IDENTITY = "formal_experiments.evaluation.aggregate_tables"
AGGREGATOR_VERSION = "formal_table_aggregation_v1"


def aggregate_five_repeats(repeats: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(repeats)
    if len(rows) != 5:
        raise ValueError("formal aggregation requires exactly five repeat-level rows")
    identities: list[tuple[int, int]] = []
    for row in rows:
        if row.get("run_mode") != "formal" or row.get("formal_result_eligible") is not True:
            raise ValueError("formal aggregation rejects development or ineligible repeat rows")
        try:
            repeat_index = int(row["repeat_index"])
            training_seed = int(row["training_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every repeat row needs integer repeat_index and training_seed") from exc
        identities.append((repeat_index, training_seed))
    expected_identities = [(index, POLICY_SEEDS[index - 1]) for index in range(1, 6)]
    if sorted(identities) != expected_identities:
        raise ValueError(
            "repeat identities must cover 1..5 exactly once with their frozen training seeds"
        )
    method_values = [row.get("method") for row in rows]
    if any(value is not None for value in method_values):
        if any(not isinstance(value, str) or not value.strip() for value in method_values):
            raise ValueError("method must be present and non-empty on every repeat row")
        if len(set(method_values)) != 1:
            raise ValueError("all repeat rows must use the same method")
    result: dict[str, Any] = {"repeat_count": 5, "ddof": 1, "metrics": {}}
    for name in TABLE_METRICS + AUXILIARY_METRICS:
        values = [row.get(name) for row in rows]
        if any(value is None for value in values):
            raise ValueError(f"{name} contains NA; formal review required")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (5,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain five finite repeat-level values")
        result["metrics"][name] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1)),
            "values": [float(value) for value in array],
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _aggregate_run_group(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    repeat_rows = []
    for run in runs:
        manifest = run["manifest"]
        repeat_rows.append({
            **run["metrics"],
            "run_mode": manifest["run_mode"],
            "formal_result_eligible": manifest["formal_result_eligible"],
            "repeat_index": manifest["repeat_index"],
            "training_seed": manifest["training_seed"],
            "method": manifest["method"],
        })
    return aggregate_five_repeats(repeat_rows)


def _csv_text(
    table_id: str,
    row_ids: Iterable[str],
    aggregations: Mapping[tuple[str, str], Mapping[str, Any]],
    groups: Mapping[tuple[str, str], list[dict[str, Any]]],
    row_labels: Mapping[str, str],
) -> str:
    fields = [
        "table_id", "row_id", "row_label", "source_table_id", "source_row_id",
        "is_alias", "repeat_count", "ddof",
    ]
    for metric in TABLE_METRICS + AUXILIARY_METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_sample_sd"))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row_id in row_ids:
        aggregation = aggregations.get((table_id, row_id))
        if aggregation is None:
            continue
        alias_of = groups[(table_id, row_id)][0].get("alias_of")
        source_table_id, source_row_id = (
            str(alias_of).split("/", 1) if alias_of else (table_id, row_id)
        )
        row: dict[str, Any] = {
            "table_id": table_id,
            "row_id": row_id,
            "row_label": row_labels[row_id],
            "source_table_id": source_table_id,
            "source_row_id": source_row_id,
            "is_alias": bool(alias_of),
            "repeat_count": aggregation["repeat_count"],
            "ddof": aggregation["ddof"],
        }
        for metric in TABLE_METRICS + AUXILIARY_METRICS:
            values = aggregation["metrics"][metric]
            row[f"{metric}_mean"] = values["mean"]
            row[f"{metric}_sample_sd"] = values["std"]
        writer.writerow(row)
    return buffer.getvalue()


def _format_metric(metric: Mapping[str, Any]) -> str:
    return f"{float(metric['mean']):.10g} ± {float(metric['std']):.10g}"


def _markdown_text(
    table_rows: Mapping[str, Iterable[str]],
    aggregations: Mapping[tuple[str, str], Mapping[str, Any]],
    row_labels: Mapping[str, str],
    statuses: Mapping[str, str],
) -> str:
    labels = {
        "table1": "Table 1: Main comparison",
        "table2": "Table 2: Component ablation",
        "table3": "Table 3: Reward ablation",
    }
    lines = [
        "# CC4-v3 formal tables",
        "",
        "Values are mean ± sample SD across five independent training repeats (ddof=1).",
        "",
    ]
    headers = (
        "Method",
        "Official reward ↑",
        "Operation-failure penalty ↓",
        "Recovery precision ↑",
        "Censored recovery time ↓",
    )
    for table_id, row_ids in table_rows.items():
        lines.extend((f"## {labels[table_id]} [{statuses[table_id]}]", ""))
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(("---", "---:", "---:", "---:", "---:")) + "|")
        found = False
        for row_id in row_ids:
            aggregation = aggregations.get((table_id, row_id))
            if aggregation is None:
                continue
            found = True
            metrics = aggregation["metrics"]
            lines.append("| " + " | ".join((
                row_labels[row_id],
                _format_metric(metrics["cc4_official_reward"]),
                _format_metric(metrics["operation_failure_penalty"]),
                _format_metric(metrics["recovery_precision"]),
                _format_metric(metrics["recovery_time_censored_mean"]),
            )) + " |")
        if not found:
            lines.append("| No eligible complete rows |  |  |  |  |")
        lines.append("")
    return "\n".join(lines)


def generate_table_artifacts(input_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Validate, aggregate, and write deterministic table artifacts.

    Discovery is shared with the paper-figure pipeline so a repeat cannot enter a
    table without passing the same eligibility, hash-binding, and metric
    recomputation gates.  The Table 3 Full-Reward logical row is supplied by the
    read-only Table 2/LWM-RL alias created by ``group_formal_rows``; no second
    physical result set is selected or copied.
    """
    from formal_experiments.evaluation.generate_paper_figures import (
        ROW_LABELS,
        TABLE_ROWS,
        discover_formal_runs,
        group_formal_rows,
    )

    source = Path(input_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    runs, rejected = discover_formal_runs(source)
    grouped, row_issues = group_formal_rows(runs, rejected)
    aggregations = {
        identity: _aggregate_run_group(group)
        for identity, group in grouped.items()
    }
    table_statuses: dict[str, str] = {}
    missing_rows: dict[str, list[str]] = {}
    for table_id, row_ids in TABLE_ROWS.items():
        missing = [row_id for row_id in row_ids if (table_id, row_id) not in grouped]
        present_count = len(row_ids) - len(missing)
        missing_rows[table_id] = missing
        table_statuses[table_id] = (
            "PASS" if not missing else ("PARTIAL" if present_count else "BLOCKED")
        )

    complete_count = len(grouped)
    expected_count = sum(len(rows) for rows in TABLE_ROWS.values())
    status = "PASS" if complete_count == expected_count else ("PARTIAL" if complete_count else "BLOCKED")
    artifact_names = tuple(f"{table_id}.csv" for table_id in TABLE_ROWS) + ("tables.md",)
    for name in artifact_names:
        stale = destination / name
        if stale.is_file():
            stale.unlink()
    output_paths: dict[str, Path] = {}
    if status == "PASS":
        for table_id, row_ids in TABLE_ROWS.items():
            path = destination / f"{table_id}.csv"
            _atomic_text(path, _csv_text(table_id, row_ids, aggregations, grouped, ROW_LABELS))
            output_paths[path.name] = path
        markdown_path = destination / "tables.md"
        _atomic_text(markdown_path, _markdown_text(TABLE_ROWS, aggregations, ROW_LABELS, table_statuses))
        output_paths[markdown_path.name] = markdown_path

    input_hashes: dict[str, str] = {}
    for run in runs:
        input_hashes.update(run["inputs"])
    serializable_aggregations = {
        f"{table_id}/{row_id}": value
        for (table_id, row_id), value in aggregations.items()
    }
    alias_rows = {
        f"{table_id}/{row_id}": {
            "source": group[0]["alias_of"],
            "mode": "read_only",
            "physical_repeat_count": len(group),
            "selection": "all source repeats 1..5 exactly once",
            "repeat_bindings": [
                {
                    "repeat_index": run["manifest"]["repeat_index"],
                    "run_dir": str(run["run_dir"]),
                    "input_sha256": dict(sorted(run["inputs"].items())),
                }
                for run in group
            ],
        }
        for (table_id, row_id), group in grouped.items()
        if group and group[0].get("alias_of")
    }
    report = {
        "schema": AGGREGATION_SCHEMA,
        "validator": AGGREGATOR_IDENTITY,
        "validator_version": AGGREGATOR_VERSION,
        "status": status,
        "passed": status == "PASS",
        "formal_inputs_only": True,
        "formal_contract": {
            "protocol_version": "cc4_v3_20260917",
            "repeats_per_row": 5,
            "episodes_per_repeat": 100,
            "ticks_per_episode": 500,
            "sample_sd_ddof": 1,
        },
        "input_root": str(source),
        "physical_accepted_repeat_count": len(runs),
        "logical_aggregated_repeat_count": sum(len(group) for group in grouped.values()),
        "complete_rows": [f"{table_id}/{row_id}" for table_id, row_id in grouped],
        "missing_rows": missing_rows,
        "table_status": table_statuses,
        "tables": {
            table_id: {
                "status": table_statuses[table_id],
                "passed": table_statuses[table_id] == "PASS",
                "required_rows": list(row_ids),
                "complete_rows": [
                    row_id for row_id in row_ids if (table_id, row_id) in grouped
                ],
                "missing_rows": missing_rows[table_id],
            }
            for table_id, row_ids in TABLE_ROWS.items()
        },
        "row_completeness_issues": row_issues,
        "rejected_inputs": rejected,
        "aliases": alias_rows,
        "input_sha256": dict(sorted(input_hashes.items())),
        "output_sha256": {name: _sha256(path) for name, path in sorted(output_paths.items())},
        "aggregations": serializable_aggregations,
        "ddof": 1,
    }
    report_path = destination / "eligibility_report.json"
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = generate_table_artifacts(args.input_root, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "table_status": report["table_status"],
        "complete_rows": report["complete_rows"],
    }, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
