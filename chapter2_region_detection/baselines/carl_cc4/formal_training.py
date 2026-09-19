"""Formal, CPU-only CARL-CC4 training orchestration.

The environment and frozen shared world model are injected deliberately: this
module owns protocol enforcement and PPO updates, but never silently substitutes
a toy environment or an unvalidated transition model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable, Iterable

import torch
from torch.nn.utils import clip_grad_norm_

from .augmentation import ImaginationGate
from .buffers import CARLTransition, RealSyntheticBuffer
from .policy import CARLActorCritic


TRAIN_SEEDS = tuple(range(1000, 1032))
VALIDATION_SEEDS = tuple(range(2000, 2008))
TEST_SEEDS = tuple(range(4000, 4100))
POLICY_SEEDS = (51001, 51002, 51003, 51004, 51005)
TICKS = 500
METHOD = "carl_cc4"


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma_tick: float = .99
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    max_grad_norm: float = .5


@dataclass(frozen=True)
class GitState:
    code_commit: str
    git_dirty: bool
    git_diff_sha256: str


def inspect_git_state() -> GitState:
    """Return auditable repository identity; formal training requires clean Git."""
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, check=True,
                          capture_output=True).stdout.encode("utf-8")
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                               cwd=root, check=True, capture_output=True,
                               text=True).stdout.strip()
    return GitState(commit, bool(diff or untracked), hashlib.sha256(diff).hexdigest())


def validate_git_state(state: GitState) -> None:
    if state.git_dirty:
        raise RuntimeError("formal CARL training requires a clean Git worktree")
    if len(state.code_commit) != 40 or len(state.git_diff_sha256) != 64:
        raise RuntimeError("invalid Git provenance")


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_gate(gate: ImaginationGate) -> None:
    if gate.status == "BLOCKED" or gate.effective_horizon is None:
        raise RuntimeError(gate.disclosure)
    if gate.requested_horizon != 256:
        raise ValueError("formal CARL must request imagination horizon H=256")
    if gate.effective_horizon != 256 and gate.status != "ADAPTED_TRUNCATED":
        raise RuntimeError("horizon reduction must be explicitly disclosed")


class CARLPPOTrainer:
    """Small duration-aware actor-critic update over the fixed 1:8 mixture."""
    def __init__(self, policy: CARLActorCritic, config: PPOConfig):
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    def update(self, buffer: RealSyntheticBuffer) -> dict[str, float]:
        real, synthetic = buffer.fixed_ratio_batch(
            real_count=len(buffer.real), synthetic_per_real=8)
        rows = real + synthetic
        states = torch.tensor([row.state for row in rows], dtype=torch.float32)
        actions = torch.tensor([row.action for row in rows], dtype=torch.long)
        rewards = torch.tensor([row.reward for row in rows], dtype=torch.float32)
        next_states = torch.tensor([row.next_state for row in rows], dtype=torch.float32)
        durations = torch.tensor([row.duration for row in rows], dtype=torch.float32)
        with torch.no_grad():
            _, next_values = self.policy(next_states)
        logits, values = self.policy(states)
        log_prob = torch.log_softmax(logits, dim=-1).gather(1, actions[:, None]).squeeze(1)
        target = rewards + torch.pow(self.config.gamma_tick, durations) * next_values
        advantage = (target - values).detach()
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        policy_loss = -(log_prob * advantage).mean()
        value_loss = (values - target).square().mean()
        loss = policy_loss + self.config.value_coefficient * value_loss \
            - self.config.entropy_coefficient * entropy
        self.optimizer.zero_grad(); loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return {"loss": float(loss.detach()), "real_samples": len(real),
                "synthetic_samples": len(synthetic)}


def select_validation_candidate(candidates: Iterable[dict], *, output: Path) -> dict:
    """Select solely by all eight validation seeds; test provenance is rejected."""
    rows = list(candidates)
    if not rows:
        raise ValueError("validation selection requires candidates")
    for row in rows:
        seeds = tuple(int(x) for x in row.get("validation_episode_seeds", ()))
        if seeds != VALIDATION_SEEDS or row.get("test_seeds_used") is not False:
            raise ValueError("candidate violates frozen validation/test isolation")
        if tuple(int(x) for x in row.get("training_episode_seeds", ())) != TRAIN_SEEDS:
            raise ValueError("candidate violates frozen training split")
        if len(str(row.get("world_model_sha256", ""))) != 64:
            raise ValueError("candidate is missing frozen WM provenance")
        if (len(str(row.get("code_commit", ""))) != 40
                or row.get("git_dirty") is not False
                or len(str(row.get("git_diff_sha256", ""))) != 64):
            raise ValueError("candidate is missing clean Git provenance")
    wm_hashes = {str(row["world_model_sha256"]) for row in rows}
    if len(wm_hashes) != 1:
        raise ValueError("validation candidates do not share one frozen WM")
    git_identities = {(str(row["code_commit"]), str(row["git_diff_sha256"])) for row in rows}
    if len(git_identities) != 1:
        raise ValueError("validation candidates do not share one clean Git identity")
    chosen = sorted(rows, key=lambda row: (-float(row["validation_score"]), str(row["candidate_id"])))[0]
    payload = {"schema": "carl_cc4_validation_selection_v1", "method": METHOD,
               "training_episode_seeds": list(TRAIN_SEEDS),
               "validation_episode_seeds": list(VALIDATION_SEEDS), "test_seeds_used": False,
               "selection_metric": "validation_score", "tie_break": "candidate_id",
               "world_model_sha256": chosen["world_model_sha256"],
               "code_commit": chosen["code_commit"], "git_dirty": False,
               "git_diff_sha256": chosen["git_diff_sha256"],
               "selected_candidate_id": chosen["candidate_id"], "candidates": rows}
    _atomic_json(output, payload)
    return payload


def train_repeat(*, repeat_index: int, output: Path, gate: ImaginationGate,
                 world_model_sha256: str, selection_path: Path,
                 collect_real: Callable[[int, CARLActorCritic, int], list[CARLTransition]],
                 generate_synthetic: Callable[[int, list[CARLTransition], CARLActorCritic, int, object], list[list[CARLTransition]]],
                 fit_seed_scm: Callable[[int, list[CARLTransition]], object],
                 resume: bool = False, config: PPOConfig = PPOConfig(),
                 git_state: Callable[[], GitState] = inspect_git_state) -> dict:
    """Train one repeat. Callbacks must wrap real CC4 and the frozen shared WM."""
    if repeat_index not in range(1, 6):
        raise ValueError("repeat_index must be 1..5")
    validate_gate(gate)
    repository = git_state(); validate_git_state(repository)
    if len(world_model_sha256) != 64:
        raise ValueError("a frozen shared-world-model sha256 is required")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("method") != METHOD or selection.get("test_seeds_used") is not False
            or tuple(selection.get("training_episode_seeds", ())) != TRAIN_SEEDS
            or tuple(selection.get("validation_episode_seeds", ())) != VALIDATION_SEEDS
            or selection.get("world_model_sha256") != world_model_sha256
            or selection.get("code_commit") != repository.code_commit
            or selection.get("git_dirty") is not False
            or selection.get("git_diff_sha256") != repository.git_diff_sha256):
        raise ValueError("invalid CARL validation-selection artifact")
    selection_copy = output.with_suffix(".selection.json")
    selection_copy.parent.mkdir(parents=True, exist_ok=True)
    temporary_selection = selection_copy.with_suffix(selection_copy.suffix + ".tmp")
    temporary_selection.write_bytes(selection_path.read_bytes())
    temporary_selection.replace(selection_copy)
    selection_sha256 = _sha256(selection_copy)
    policy_seed = POLICY_SEEDS[repeat_index - 1]
    torch.manual_seed(policy_seed)
    policy = CARLActorCritic().cpu(); trainer = CARLPPOTrainer(policy, config)
    progress = output.with_suffix(output.suffix + ".progress")
    completed: list[int] = []
    if resume and progress.is_file():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        meta = saved.get("metadata", {})
        if (meta.get("method") != METHOD or meta.get("policy_seed") != policy_seed
                or meta.get("world_model_sha256") != world_model_sha256
                or meta.get("validation_selection_sha256") != selection_sha256
                or meta.get("code_commit") != repository.code_commit
                or meta.get("git_dirty") is not False
                or meta.get("git_diff_sha256") != repository.git_diff_sha256):
            raise RuntimeError("CARL resume identity mismatch")
        completed = [int(x) for x in meta.get("training_episode_seeds", ())]
        if tuple(completed) != TRAIN_SEEDS[:len(completed)]:
            raise RuntimeError("CARL resume seed prefix is invalid")
        policy.load_state_dict(saved["state_dict"])
        trainer.optimizer.load_state_dict(saved["optimizer_state_dict"])
    logs = []
    for episode_seed in TRAIN_SEEDS[len(completed):]:
        real = collect_real(episode_seed, policy, TICKS)
        if not real or any(row.episode_seed != episode_seed or row.source != "real" for row in real):
            raise RuntimeError("real buffer seed/source isolation violation")
        scm = fit_seed_scm(episode_seed, real)  # one SCM fit boundary per episode seed
        synthetic = generate_synthetic(
            episode_seed, real, policy, int(gate.effective_horizon), scm)
        buffer = RealSyntheticBuffer()
        for row in real: buffer.add_real(row)
        buffer.add_synthetic_rollouts(origin_episode_seed=episode_seed, rollouts=synthetic)
        metrics = trainer.update(buffer); completed.append(episode_seed)
        logs.append({"episode_seed": episode_seed, **metrics})
        _atomic_torch_save(progress, {"state_dict": policy.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(), "metadata": {
                "method": METHOD, "policy_seed": policy_seed,
                "training_episode_seeds": completed, "world_model_sha256": world_model_sha256,
                "validation_selection_sha256": selection_sha256,
                "code_commit": repository.code_commit, "git_dirty": False,
                "git_diff_sha256": repository.git_diff_sha256}})
    metadata = {"schema": "carl_cc4_formal_checkpoint_v1",
        "method": METHOD, "policy_seed": policy_seed,
        "formal_training_complete": completed == list(TRAIN_SEEDS), "training_split": "train",
        "training_episode_seeds": completed, "ticks_per_episode": TICKS,
        "validation_episode_seeds": list(VALIDATION_SEEDS), "test_seeds_used": False,
        "world_model_frozen": True, "world_model_sha256": world_model_sha256,
        "synthetic_rollouts_per_real_rollout": 8, "requested_imagination_horizon": 256,
        "effective_imagination_horizon": gate.effective_horizon,
        "imagination_gate_status": gate.status, "imagination_disclosure": gate.disclosure,
        "hidden_truth_policy_input": False, "decision_time_model_use": False,
        "validation_selection_sha256": selection_sha256,
        "validation_selection_file": selection_copy.name, "ppo": asdict(config),
        "code_commit": repository.code_commit, "git_dirty": False,
        "git_diff_sha256": repository.git_diff_sha256,
        "training_log": logs}
    _atomic_torch_save(output, {"state_dict": policy.state_dict(), **metadata})
    return metadata
