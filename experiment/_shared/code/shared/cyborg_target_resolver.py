from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import (
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# ============================================================
# Frozen targeted action families
# ============================================================

TARGETED_FAMILIES: Tuple[
    str,
    ...,
] = (
    "Analyse",
    "Remove",
    "Restore",
)


# ============================================================
# Valid action record
# ============================================================

@dataclass(frozen=True)
class ValidCyborgAction:
    """
    BlueFixedActionWrapper 中一个
    当前真正可执行的 low-level action。

    index:
        wrapper 当前 raw action index。

    label:
        wrapper action label。

    family:
        Sleep / Analyse / Remove / Restore。

    target_host:
        host-specific action 的目标主机。
        Sleep 为 None。
    """

    index: int
    label: str
    family: str
    target_host: Optional[str]


# ============================================================
# Internal validation / parsing
# ============================================================

def _validate_labels_and_mask(
    labels: Sequence[object],
    mask: Sequence[object],
) -> None:
    if len(labels) != len(mask):
        raise ValueError(
            "action_labels and action_mask "
            "must have the same length"
        )

    if len(labels) == 0:
        raise ValueError(
            "action_labels must not be empty"
        )


def _parse_label(
    label: object,
) -> tuple[
    str,
    Optional[str],
    bool,
]:
    """
    解析一个 CC4 wrapper label。

    返回：
        family
        target_host
        explicitly_invalid

    这里只使用精确 action family 前缀，
    不使用模糊 substring 匹配。
    """

    text = str(
        label
    ).strip()

    invalid_prefix = (
        "[Invalid] "
    )

    explicitly_invalid = (
        text.startswith(
            invalid_prefix
        )
    )

    if explicitly_invalid:
        text = text[
            len(
                invalid_prefix
            ):
        ].strip()

    if text == "Sleep":
        return (
            "Sleep",
            None,
            explicitly_invalid,
        )

    for family in TARGETED_FAMILIES:
        prefix = (
            family
            + " "
        )

        if text.startswith(
            prefix
        ):
            host = text[
                len(
                    prefix
                ):
            ].strip()

            if not host:
                return (
                    "other",
                    None,
                    explicitly_invalid,
                )

            return (
                family,
                host,
                explicitly_invalid,
            )

    return (
        "other",
        None,
        explicitly_invalid,
    )


# ============================================================
# Four-action valid candidate extraction
# ============================================================

def valid_actions(
    labels: Sequence[object],
    mask: Sequence[object],
) -> Tuple[
    ValidCyborgAction,
    ...,
]:
    """
    从 wrapper 当前 action space
    提取真正合法的四动作候选。

    规则：

    1. action_mask=False 排除；
    2. [Invalid] 排除；
    3. 只允许：
       Sleep / Analyse / Remove / Restore；
    4. Monitor / traffic / decoy 等排除。
    """

    _validate_labels_and_mask(
        labels,
        mask,
    )

    out = []

    for index, (
        raw_label,
        raw_valid,
    ) in enumerate(
        zip(
            labels,
            mask,
        )
    ):
        if not bool(
            raw_valid
        ):
            continue

        (
            family,
            target_host,
            explicitly_invalid,
        ) = _parse_label(
            raw_label
        )

        if explicitly_invalid:
            continue

        if (
            family != "Sleep"
            and family
            not in TARGETED_FAMILIES
        ):
            continue

        out.append(
            ValidCyborgAction(
                index=int(
                    index
                ),

                label=str(
                    raw_label
                ).strip(),

                family=family,

                target_host=(
                    target_host
                ),
            )
        )

    return tuple(
        out
    )


# ============================================================
# Sleep resolver
# ============================================================

def resolve_sleep_action(
    labels: Sequence[object],
    mask: Sequence[object],
) -> ValidCyborgAction:
    """
    动态寻找真正有效的 Sleep。

    禁止依赖 probe 观察到的固定 raw index。
    """

    candidates = [
        item
        for item
        in valid_actions(
            labels,
            mask,
        )
        if (
            item.family
            == "Sleep"
            and item.target_host
            is None
        )
    ]

    if not candidates:
        raise RuntimeError(
            "no valid Sleep action "
            "is available"
        )

    return min(
        candidates,
        key=lambda item:
            item.index,
    )


# ============================================================
# Observable host-score validation
# ============================================================

def _validate_host_scores(
    host_scores: Mapping[
        str,
        Real,
    ],
) -> dict[
    str,
    float,
]:
    """
    验证 planner-visible host score。

    不读取 hidden truth。
    """

    checked: dict[
        str,
        float,
    ] = {}

    for (
        raw_host,
        raw_score,
    ) in host_scores.items():

        host = str(
            raw_host
        ).strip()

        if not host:
            raise ValueError(
                "host score key "
                "must not be empty"
            )

        if (
            isinstance(
                raw_score,
                bool,
            )
            or not isinstance(
                raw_score,
                Real,
            )
        ):
            raise ValueError(
                "host score must be "
                "a real number: "
                f"{raw_host!r} -> "
                f"{raw_score!r}"
            )

        score = float(
            raw_score
        )

        if not isfinite(
            score
        ):
            raise ValueError(
                "host score must be finite: "
                f"{host!r} -> "
                f"{score!r}"
            )

        checked[
            host
        ] = score

    return checked


# ============================================================
# A4.1c
# Planner-visible targeted action availability
# ============================================================

def observable_target_family_availability(
    labels: Sequence[object],
    mask: Sequence[object],
    observable_host_scores: Mapping[
        str,
        Real,
    ] | None,
) -> dict[
    str,
    bool,
]:
    """
    判断 Analyse / Remove / Restore
    当前是否存在真正合法且可观察的 target。

    A4.6a 已证明：

        observable evidence exists

    并不等价于：

        valid observable action target exists

    因此这里判断的是：

        observable scored host
              ∩
        wrapper current valid targets
              != empty

    输入只来自：

        action_labels
        action_mask
        observable_host_scores

    不读取：

        controller true state
        Red sessions
        compromise truth
        incident truth
        reward
        future information
    """

    _validate_labels_and_mask(
        labels,
        mask,
    )

    if not observable_host_scores:
        return {
            family: False
            for family
            in TARGETED_FAMILIES
        }

    scores = (
        _validate_host_scores(
            observable_host_scores
        )
    )

    candidates = valid_actions(
        labels,
        mask,
    )

    result: dict[
        str,
        bool,
    ] = {}

    for family in TARGETED_FAMILIES:
        result[
            family
        ] = any(
            (
                item.family
                == family

                and item.target_host
                is not None

                and item.target_host
                in scores
            )

            for item
            in candidates
        )

    return result


# ============================================================
# Targeted action resolver
# ============================================================

def resolve_target_action(
    family: str,
    labels: Sequence[object],
    mask: Sequence[object],
    observable_host_scores: Mapping[
        str,
        Real,
    ] | None,
) -> Optional[
    ValidCyborgAction
]:
    """
    为 Analyse / Remove / Restore
    选择一个 host-specific low-level action。

    规则：

    1. action_mask=True；
    2. 非 [Invalid]；
    3. family 必须匹配；
    4. target host 必须存在于 observable scores；
    5. score 最大优先；
    6. 同分 hostname 字典序；
    7. 再按 raw index；
    8. 无合法 target 返回 None。

    Adapter 收到 None 后 fallback Sleep。
    """

    if family not in TARGETED_FAMILIES:
        raise ValueError(
            "target family must be one of "
            f"{TARGETED_FAMILIES}; "
            f"got {family!r}"
        )

    _validate_labels_and_mask(
        labels,
        mask,
    )

    if not observable_host_scores:
        return None

    scores = (
        _validate_host_scores(
            observable_host_scores
        )
    )

    candidates = [
        item
        for item
        in valid_actions(
            labels,
            mask,
        )
        if (
            item.family
            == family

            and item.target_host
            is not None

            and item.target_host
            in scores
        )
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda item: (
            -scores[
                item.target_host
            ],
            item.target_host,
            item.index,
        ),
    )