from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_experiments.evaluation.run_b4_provisional_stage import (
    run_strict_stage,
    DEFAULT_FINAL_WORLD_MODEL,
    DEFAULT_FINAL_REWARD_MODEL,
    DEFAULT_FINAL_CACHE_ROOT,
    DEFAULT_FINAL_OUT_ROOT,
)


def run_lwm_rl(
    *,
    model_alias: str,
    stage_target: int,
    device: str,
    seed: int,
    output_root: str,
):

    # 当前B4协议固定seed
    if seed != 20260917:
        raise ValueError(
            "current frozen protocol only supports seed=20260917"
        )

    report = run_strict_stage(
        model_alias=model_alias,
        stage_target=stage_target,
        device=device,
        resume=False,
        world_model_path=DEFAULT_FINAL_WORLD_MODEL,
        reward_model_path=DEFAULT_FINAL_REWARD_MODEL,
        cache_root=DEFAULT_FINAL_CACHE_ROOT,
        out_root=output_root,
    )

    return {
        "method": "LWM-RL",
        "run_mode": "provisional",
        "formal_result_eligible": False,
        "model_alias": model_alias,
        "stage_target": stage_target,
        "report": report,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Legacy provisional LWM-RL stage entrypoint (not a formal result)"
    )

    parser.add_argument(
        "--method",
        default="lwm_rl",
        choices=[
            "lwm_rl",
        ],
    )

    parser.add_argument(
        "--model-alias",
        required=True,
    )

    parser.add_argument(
        "--stage-target",
        type=int,
        default=2000,
        choices=[
            2000,
            5000,
            10000,
            20000,
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260917,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--output",
        default="outputs/provisional_results",
    )

    args = parser.parse_args()


    if args.method == "lwm_rl":

        result = run_lwm_rl(
            model_alias=args.model_alias,
            stage_target=args.stage_target,
            device=args.device,
            seed=args.seed,
            output_root=args.output,
        )


    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    result_file = (
        out /
        f"{args.method}_{args.model_alias}.json"
    )

    result_file.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print("=" * 80)
    print("[PROVISIONAL EXPERIMENT - NOT FORMAL-RESULT ELIGIBLE]")
    print("method:", args.method)
    print("model:", args.model_alias)
    print("output:", result_file)
    print("=" * 80)


if __name__ == "__main__":
    main()
