"""Table-3 reward ablation contracts, training, validation, and freezing."""

from __future__ import annotations

import argparse
from dataclasses import replace
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from formal_experiments.training.response_reward_predictor import (
    ResponseRewardDataset, ResponseRewardPredictor, ResponseRewardPredictorConfig,
    build_response_reward_dataset,
)

GAMMA_TICK = 0.99
HORIZON = 4
FAIL_ONLY_FORMAL_LOSS = "smooth_l1"
FAIL_ONLY_FORMAL_BETA = 1.0


class RewardMode(str, Enum):
    DELAY_ONLY = "Delay-Only"
    FAIL_ONLY = "Fail-Only"
    FULL_REWARD = "Full-Reward"


def formal_predictor_config(mode: RewardMode) -> ResponseRewardPredictorConfig:
    """Return the frozen optimization contract for one Table-3 predictor."""
    if mode == RewardMode.FAIL_ONLY:
        return ResponseRewardPredictorConfig(
            loss=FAIL_ONLY_FORMAL_LOSS,
            smooth_l1_beta=FAIL_ONLY_FORMAL_BETA,
        )
    return ResponseRewardPredictorConfig()


class HurdleFailurePredictor:
    """Diagnostic only; forbidden for formal Table-3 by the architecture lock."""
    FORMAT = "fail_only_hurdle_v1"

    def __init__(self, occurrence, severity):
        self.occurrence = occurrence; self.severity = severity
        self.config = severity.config; self.device = severity.device

    def predict_tensor(self, states, actions, next_states):
        probability = self.occurrence.predict_tensor(states, actions, next_states).clamp(0.0, 1.0)
        severity = self.severity.predict_tensor(states, actions, next_states).clamp(max=0.0)
        return probability * severity

    def predict(self, states, actions, next_states):
        with torch.no_grad():
            return self.predict_tensor(torch.as_tensor(states, dtype=torch.float32, device=self.device),
                                       torch.as_tensor(actions, dtype=torch.long, device=self.device),
                                       torch.as_tensor(next_states, dtype=torch.float32, device=self.device)).cpu().numpy()

    def save_checkpoint(self, path: Path):
        path = Path(path); occurrence_path = path.with_name("occurrence_predictor.pt")
        severity_path = path.with_name("severity_predictor.pt")
        self.occurrence.save_checkpoint(occurrence_path); self.severity.save_checkpoint(severity_path)
        torch.save({"format": self.FORMAT, "occurrence": occurrence_path.name,
                    "occurrence_sha256": hashlib.sha256(occurrence_path.read_bytes()).hexdigest(),
                    "severity": severity_path.name,
                    "severity_sha256": hashlib.sha256(severity_path.read_bytes()).hexdigest()}, path)

    @classmethod
    def load_checkpoint(cls, path: Path, *, device="cpu"):
        path = Path(path); payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != cls.FORMAT: raise ValueError("invalid Fail-Only hurdle checkpoint")
        occurrence_path = path.with_name(payload["occurrence"]); severity_path = path.with_name(payload["severity"])
        if hashlib.sha256(occurrence_path.read_bytes()).hexdigest() != payload.get("occurrence_sha256"):
            raise ValueError("Fail-Only occurrence component SHA mismatch")
        if hashlib.sha256(severity_path.read_bytes()).hexdigest() != payload.get("severity_sha256"):
            raise ValueError("Fail-Only severity component SHA mismatch")
        return cls(ResponseRewardPredictor.load_checkpoint(occurrence_path, device=device),
                   ResponseRewardPredictor.load_checkpoint(severity_path, device=device))


def _dataset_with_rewards(dataset: ResponseRewardDataset, rewards) -> ResponseRewardDataset:
    return ResponseRewardDataset(states=dataset.states, actions=dataset.actions,
                                 next_states=dataset.next_states,
                                 rewards=np.asarray(rewards, dtype=np.float32))


def train_hurdle_failure(dataset: ResponseRewardDataset, *, device: str):
    nonzero = dataset.rewards < 0
    if nonzero.sum() < 2 or (~nonzero).sum() < 2:
        raise RuntimeError("Fail-Only hurdle requires both zero and nonzero train labels")
    occurrence = ResponseRewardPredictor(ResponseRewardPredictorConfig(), device=device)
    occurrence_summary = occurrence.fit(_dataset_with_rewards(dataset, nonzero.astype(np.float32)))
    severity_data = ResponseRewardDataset(states=dataset.states[nonzero], actions=dataset.actions[nonzero],
                                          next_states=dataset.next_states[nonzero], rewards=dataset.rewards[nonzero])
    severity = ResponseRewardPredictor(ResponseRewardPredictorConfig(model_seed=20260918), device=device)
    severity_summary = severity.fit(severity_data)
    return HurdleFailurePredictor(occurrence, severity), {
        "zero_count": int((~nonzero).sum()), "nonzero_count": int(nonzero.sum()),
        "nonzero_fraction": float(nonzero.mean()),
        "occurrence_training": vars(occurrence_summary), "severity_training": vars(severity_summary),
    }


def transition_reward(item, mode: RewardMode) -> float:
    delay = float(item.incident_delay_penalty)
    failure = float(item.incident_host_lwf_raw_penalty)
    if not np.isfinite(delay) or delay < 0 or not np.isfinite(failure) or failure > 0:
        raise ValueError("invalid delay/failure reward components")
    expected = -delay + failure
    if not np.isclose(float(item.response_reward), expected, rtol=0, atol=1e-5):
        raise ValueError("Full-Reward label does not equal delay + failure components")
    if mode == RewardMode.DELAY_ONLY: return -delay
    if mode == RewardMode.FAIL_ONLY: return failure
    return expected


def tick_reward(accounting, mode: RewardMode) -> float:
    delay = float(accounting.incident_delay_penalty)
    failure = float(accounting.incident_host_lwf_raw_penalty)
    if not np.isfinite(delay) or delay < 0 or not np.isfinite(failure) or failure > 0:
        raise ValueError("invalid runtime delay/failure reward components")
    expected = -delay + failure
    if not np.isclose(float(accounting.response_reward), expected, rtol=0, atol=1e-5):
        raise ValueError("runtime Full-Reward accounting mismatch")
    if mode == RewardMode.DELAY_ONLY: return -delay
    if mode == RewardMode.FAIL_ONLY: return failure
    return expected


def derive(transitions, mode: RewardMode):
    if not transitions: raise ValueError("empty replay")
    return [replace(item, response_reward=transition_reward(item, mode)) for item in transitions]


def _rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def baseline_metrics(transitions, train_mean: float) -> dict:
    from formal_experiments.evaluation.validate_bootstrap_world_model import spearman
    completed = [item for item in transitions if item.action_completed]
    truth = np.asarray([item.response_reward for item in completed], dtype=np.float64)
    previous = {}; persistence = []
    for item in completed:
        key = (item.episode_seed, item.agent_name)
        persistence.append(previous.get(key, train_mean)); previous[key] = item.response_reward
    constant = np.full_like(truth, train_mean)
    return {"constant_rmse": _rmse(truth, constant), "persistence_rmse": _rmse(truth, persistence),
            "persistence_spearman": spearman(truth, np.asarray(persistence))}


def train_mode(*, mode: RewardMode, train, validation, out_dir: Path, device="cpu",
               world_model_path: Path = Path("outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt")) -> dict:
    from formal_experiments.evaluation.validate_bootstrap_world_model import validate_frozen_split
    from formal_experiments.evaluation.validate_response_reward_predictor import (
        build_reward_rollout_windows, one_step_metrics, oracle_state_h4_metrics,
        wm_h4_metrics, discount_weights,
    )
    from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
    if mode == RewardMode.FULL_REWARD: raise ValueError("Full-Reward uses the already frozen artifact")
    validate_frozen_split(train, "train"); validate_frozen_split(validation, "validation")
    if {x.episode_seed for x in train} & {x.episode_seed for x in validation}:
        raise RuntimeError("train/validation leakage")
    train_m = derive(train, mode); validation_m = derive(validation, mode)
    train_dataset = build_response_reward_dataset(train_m)
    # Section 5.3 freezes architecture and optimization to the Full-Reward
    # predictor except for the approved Fail-Only loss substitution. Hurdle,
    # weighting, and resampling alternatives are diagnostic only and must never
    # enter this formal path.
    predictor_config = formal_predictor_config(mode)
    predictor = ResponseRewardPredictor(predictor_config, device=device)
    summary = predictor.fit(train_dataset); training_report = vars(summary)
    training_report["zero_label_count"] = int(np.count_nonzero(train_dataset.rewards == 0))
    training_report["nonzero_label_count"] = int(np.count_nonzero(train_dataset.rewards != 0))
    training_report["nonzero_label_fraction"] = float(np.mean(train_dataset.rewards != 0))
    architecture = "single Full-Reward-compatible MLP"
    h1 = one_step_metrics(predictor, train_dataset, validation_m)
    train_mean = float(train_dataset.rewards.mean())
    baselines = baseline_metrics(validation_m, train_mean)
    windows = build_reward_rollout_windows(validation_m, horizon=HORIZON)
    h4 = oracle_state_h4_metrics(predictor, windows, train_mean, GAMMA_TICK)
    world_model = BootstrapProbabilisticWorldModel.load_checkpoint(world_model_path, device=device)
    if world_model.config.target_mode != "absolute" or world_model.config.ensemble_size != 5:
        raise RuntimeError("Table-3 requires the frozen five-member absolute world model")
    wm_h4 = wm_h4_metrics(world_model=world_model, predictor=predictor, windows=windows,
                          train_reward_mean=train_mean, gamma_tick=GAMMA_TICK)
    # Persistence H4 repeats the first true reward. This is a deliberately strong,
    # label-only validation baseline and is never available to the policy.
    h4_truth = np.asarray([float(np.dot(discount_weights(w.durations, GAMMA_TICK), w.rewards)) for w in windows])
    h4_persistence = np.asarray([float(discount_weights(w.durations, GAMMA_TICK).sum() * w.rewards[0]) for w in windows])
    h4["persistence_baseline_rmse"] = _rmse(h4_truth, h4_persistence)
    wm_h4["persistence_baseline_rmse"] = h4["persistence_baseline_rmse"]
    gate = {
        "h1_beats_constant": h1["rmse"] < baselines["constant_rmse"],
        "h1_beats_persistence": h1["rmse"] < baselines["persistence_rmse"],
        "h1_positive_rank": h1["spearman"] > 0,
        "h4_beats_constant": wm_h4["rmse"] < wm_h4["constant_baseline_rmse"],
        "h4_beats_persistence": wm_h4["rmse"] < wm_h4["persistence_baseline_rmse"],
        "h4_positive_rank": wm_h4["spearman"] > 0,
    }
    gate["pass"] = all(gate.values())
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "response_reward_predictor.pt"
    predictor.save_checkpoint(checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loss_contract = {
        "name": predictor_config.loss,
        "target_space": "standardized_reward_label",
        "reduction": "mean",
        "sample_weighting": "none",
        "resampling": "none",
    }
    if predictor_config.loss == "smooth_l1":
        loss_contract["beta"] = predictor_config.smooth_l1_beta
    manifest = {"format": "table3_reward_ablation_v2", "mode": mode.value,
                "reward_model_eligible_for_policy_training": bool(gate["pass"]),
                "formal_result_eligible": False, "test_seeds_used": False,
                "architecture": {"description": architecture, "component_config": vars(predictor_config)},
                "formal_contract_architecture_match": True,
                "formal_contract_training_match": True,
                "diagnostic_hurdle_used_for_formal_artifact": False,
                "training_loss": loss_contract,
                "training_policy": ("approved Fail-Only minimum change: unweighted SmoothL1(beta=1.0) on "
                                    "standardized labels; otherwise matches frozen Full-Reward: 50 fixed epochs; "
                                    "no validation checkpoint selection" if mode == RewardMode.FAIL_ONLY else
                                    "matches frozen Full-Reward: MSE on standardized labels; 50 fixed epochs; "
                                    "no validation checkpoint selection"),
                "training": training_report, "h1": h1, "h1_baselines": baselines,
                "h4_oracle_state": h4, "h4_imagined_world_model": wm_h4,
                "world_model_sha256": hashlib.sha256(world_model_path.read_bytes()).hexdigest(),
                "quality_gate": gate,
                "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                "frozen": True}
    (out_dir / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    from formal_experiments.evaluation.validate_bootstrap_world_model import load_replay_jsonl
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=[RewardMode.DELAY_ONLY.value, RewardMode.FAIL_ONLY.value], required=True)
    parser.add_argument("--train", default="outputs/formal_replay_final_20260917/train.jsonl")
    parser.add_argument("--validation", default="outputs/formal_replay_final_20260917/validation.jsonl")
    parser.add_argument("--world-model", type=Path, default=Path("outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt"))
    parser.add_argument("--out-dir", required=True, type=Path); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); train = load_replay_jsonl(args.train); validation = load_replay_jsonl(args.validation)
    print(json.dumps(train_mode(mode=RewardMode(args.mode), train=train, validation=validation,
                                out_dir=args.out_dir, device=args.device,
                                world_model_path=args.world_model), indent=2))


if __name__ == "__main__": main()
