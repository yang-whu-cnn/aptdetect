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


TARGETED_FAMILIES: Tuple[
    str,
    ...,
] = (
    "Analyse",
    "Remove",
    "Restore",
)


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

    注意：
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
            len(invalid_prefix):
        ].strip()

    # no-op
    if text == "Sleep":
        return (
            "Sleep",
            None,
            explicitly_invalid,
        )

    # host-specific actions
    for family in TARGETED_FAMILIES:
        prefix = (
            family
            + " "
        )

        if text.startswith(
            prefix
        ):
            host = text[
                len(prefix):
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

    # Monitor / traffic / decoy / others
    return (
        "other",
        None,
        explicitly_invalid,
    )


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
    1. action_mask=False 一律排除；
    2. label 带 [Invalid] 一律排除；
    3. 只允许：
       - Sleep
       - Analyse
       - Remove
       - Restore
    4. Monitor / traffic / decoy
       不进入当前论文四动作空间。
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
        # A3.1 已真实确认：
        # invalid padded slots 的底层 action
        # 可能实际变成 Sleep。
        #
        # 所以第一层必须先看 mask。
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

        # 防御性检查：
        # 即使未来 wrapper 异常地把
        # [Invalid] slot mask 成 True，
        # 也禁止使用。
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


def resolve_sleep_action(
    labels: Sequence[object],
    mask: Sequence[object],
) -> ValidCyborgAction:
    """
    动态寻找当前真正有效的 Sleep。

    禁止依赖 A3.1 probe 中观察到的：
        blue_agent_0..3 -> 49
        blue_agent_4    -> 145

    这些 index 只能用于 probe，
    不能进入正式 adapter 逻辑。
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

    # 正常 CC4 当前只有一个 valid Sleep。
    #
    # 若未来 wrapper 返回多个，
    # 使用最小当前 raw index
    # 作为 deterministic tie-break。
    #
    # 这不是固定 index。
    return min(
        candidates,
        key=lambda item:
            item.index,
    )


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
    验证来自当前 Blue observation
    的 host score。

    score 的具体构造方式不属于 A3.3；
    A4 正式 state/evidence 阶段再冻结。

    当前这里只保证 resolver：
    - 不读取 hidden truth；
    - 输入是有限实数；
    - 输出完全 deterministic。
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
    为：
        Analyse
        Remove
        Restore

    选择一个 host-specific low-level action。

    observable_host_scores 必须由调用方
    根据“当前 Blue 可观察信息”产生。

    本 resolver 自己：
    - 不读取 Red agent；
    - 不读取真实 compromise state；
    - 不读取 future state；
    - 不读取 test label；
    - 不读取攻击脚本真值。

    选择规则：

    1. 先验证 labels + mask；
    2. 只保留 action_mask=True；
    3. 只保留 requested family；
    4. 只考虑 observable_host_scores
       中实际出现的 host；
    5. score 最高者优先；
    6. score 相同：
       host 名字字典序优先；
    7. 同 host 再按 raw index；
    8. 没有合法 observable target
       时返回 None。

    Adapter 收到 None 后统一 fallback Sleep。
    """

    if family not in TARGETED_FAMILIES:
        raise ValueError(
            "target family must be one of "
            f"{TARGETED_FAMILIES}; "
            f"got {family!r}"
        )

    # 即使 host score 为空，
    # labels/mask contract 也必须有效。
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

    # min + -score：
    # 等价于 score 最大优先，
    # 然后 deterministic host/index tie-break。
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