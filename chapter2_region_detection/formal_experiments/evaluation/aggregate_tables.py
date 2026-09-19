"""Five-repeat aggregation for the formal paper tables."""

from __future__ import annotations

from math import isfinite
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
