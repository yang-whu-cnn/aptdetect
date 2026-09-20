"""Train-only progress ensemble and validation-only uncertainty calibration."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from shared.formal_state import BLUE_AGENTS, FORMAL_STATE_DIM
from shared.d27_projection import PROJECTION_SHA256, PROJECTION_VERSION

TRAIN_SEEDS = frozenset(range(1000, 1032))
VALIDATION_SEEDS = frozenset(range(2000, 2008))
ENSEMBLE_SIZE = 5
UNCERTAINTY_KEYS = ("world_model", "progress", "prior_entropy")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_REPLAY = "outputs/formal_replay_final_20260917/train.jsonl"
DEFAULT_VALIDATION_REPLAY = "outputs/formal_replay_final_20260917/validation.jsonl"
DEFAULT_PROGRESS_SIDECAR = (
    "outputs/uamcts_cc4/progress/progress_ensemble_train_only.sidecar.json"
)
DEFAULT_PRIOR_ENTROPY_SIDECAR = (
    "outputs/uamcts_cc4/calibration/validation_prior_entropy.sidecar.json"
)
DEFAULT_NORMALIZER_SIDECAR = (
    "outputs/uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.sidecar.json"
)
PROGRESS_SIDECAR_SCHEMA = "uamcts_progress_sidecar_v1"
PRIOR_ENTROPY_SIDECAR_SCHEMA = "uamcts_validation_prior_entropy_sidecar_v1"
NORMALIZER_SIDECAR_SCHEMA = "uamcts_uncertainty_normalizers_sidecar_v1"
PROGRESS_LABEL = "normalized_future_response_return"
PROGRESS_DERIVATION_VERSION = "uamcts_progress_derivation_v1"


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sidecar_value(payload: dict, *paths):
    """Read a canonical field while accepting the nested spelling used by reports."""
    for path in paths:
        value = payload
        try:
            for key in path if isinstance(path, tuple) else (path,):
                value = value[key]
        except (KeyError, TypeError):
            continue
        return value
    return None


def _read_json_mapping(path: Path, label: str) -> tuple[dict | None, list[str]]:
    if not path.is_file():
        return None, [f"missing {label}: {path}"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"{label} cannot be loaded: {exc}"]
    if not isinstance(value, dict):
        return None, [f"{label} must be a JSON object"]
    return value, []


def _validate_replay_binding(
    *, payload: dict, label: str, expected_seeds: frozenset[int],
    expected_sha: str | None = None, expected_record_count: int | None = None,
    replay_path: str | Path | None = None,
) -> list[str]:
    errors: list[str] = []
    prefix = label.replace(" ", "_")
    entry = payload.get(prefix)
    if not isinstance(entry, dict):
        entry = payload.get(label)
    sha_paths = [f"{prefix}_sha256", f"{label}_sha256",
                 (prefix, "sha256"), (label, "sha256")]
    if label == "canonical_train":
        # The sidecar's canonical spelling is deliberately explicit: this is
        # the replay used to fit the train-only progress ensemble.  Keep the
        # older nested/report spellings as compatibility aliases, but do not
        # make ``canonical_train_replay_sha256`` an implicit/unused argument.
        sha_paths.extend([
            "canonical_train_replay_sha256", "train_replay_sha256",
            ("canonical_train_replay", "sha256"),
            ("train_replay", "sha256"), ("train", "sha256"),
        ])
    elif label == "validation":
        sha_paths.extend([
            "validation_replay_sha256", ("validation_replay", "sha256"),
        ])
    sha = _sidecar_value(payload, *sha_paths)
    if not _is_sha256(sha):
        errors.append(f"progress sidecar {label} replay SHA256 is invalid")
    if expected_sha is not None and sha != expected_sha:
        errors.append(f"progress sidecar {label} replay SHA256 mismatch")
    seed_paths = [f"{prefix}_seeds", f"{label}_seeds",
                  (prefix, "seeds"), (label, "seeds")]
    if label == "canonical_train":
        seed_paths.extend(["train_seeds", ("train_replay", "seeds"),
                           ("train", "seeds")])
    seeds = _sidecar_value(payload, *seed_paths)
    try:
        actual_seeds = tuple(int(seed) for seed in seeds)
    except (TypeError, ValueError):
        actual_seeds = ()
    if actual_seeds != tuple(sorted(expected_seeds)):
        errors.append(f"progress sidecar {label} seeds mismatch")
    count_paths = [f"{prefix}_record_count", f"{label}_record_count",
                   (prefix, "record_count"), (label, "record_count")]
    if label == "canonical_train":
        count_paths.extend(["train_record_count", ("train_replay", "record_count"),
                            ("train", "record_count")])
    count = _sidecar_value(payload, *count_paths)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        errors.append(f"progress sidecar {label} record_count is invalid")
    elif expected_record_count is not None and count != expected_record_count:
        errors.append(f"progress sidecar {label} record_count mismatch")
    if replay_path is not None:
        resolved = _resolve_project_path(replay_path)
        if not resolved.is_file():
            errors.append(f"progress sidecar {label} replay is missing: {resolved}")
        elif _is_sha256(sha) and sha256_file(resolved) != sha:
            errors.append(f"progress sidecar {label} replay file SHA256 mismatch")
    return errors


def validate_progress_sidecar(
    path: str | Path, *, progress_sha256: str,
    canonical_train_replay_sha256: str | None = None,
    canonical_train_record_count: int | None = None,
    validation_replay_sha256: str | None = None,
    train_replay_path: str | Path | None = None,
    validation_replay_path: str | Path | None = None,
) -> tuple[dict | None, list[str]]:
    """Validate the immutable provenance paired with a progress checkpoint.

    The checkpoint deliberately does not contain validation metrics.  Keeping
    them in this JSON sidecar makes the quality decision auditable without
    changing a binary model and lets a loader reject a sidecar copied from a
    different checkpoint or replay namespace.
    """
    payload, errors = _read_json_mapping(_resolve_project_path(path), "progress sidecar")
    if payload is None:
        return None, errors
    expected = {
        "schema": PROGRESS_SIDECAR_SCHEMA,
        "checkpoint_sha256": progress_sha256,
        "ensemble_size": ENSEMBLE_SIZE,
        "test_seeds_used": False,
        "provider_calls": 0,
        "validation_gate_pass": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"progress sidecar {key} mismatch")
    if payload.get("progress_sha256") not in (None, progress_sha256):
        errors.append("progress sidecar progress_sha256 mismatch")
    label = payload.get("label_contract")
    if not isinstance(label, dict):
        if payload.get("label") != PROGRESS_LABEL:
            errors.append("progress sidecar label_contract.label mismatch")
        if payload.get("derivation_version") != PROGRESS_DERIVATION_VERSION:
            errors.append("progress sidecar derivation_version mismatch")
    else:
        if label.get("label") != PROGRESS_LABEL:
            errors.append("progress sidecar label_contract.label mismatch")
        if label.get("derivation_version") != PROGRESS_DERIVATION_VERSION:
            errors.append("progress sidecar label_contract.derivation_version mismatch")
    rmse = payload.get("validation_rmse")
    baseline = payload.get("constant_baseline_rmse")
    if (isinstance(rmse, bool) or not isinstance(rmse, (int, float))
            or not np.isfinite(float(rmse)) or float(rmse) < 0):
        errors.append("progress sidecar validation_rmse is invalid")
    if (isinstance(baseline, bool) or not isinstance(baseline, (int, float))
            or not np.isfinite(float(baseline)) or float(baseline) <= 0):
        errors.append("progress sidecar constant_baseline_rmse is invalid")
    elif isinstance(rmse, (int, float)) and float(rmse) >= float(baseline):
        errors.append("progress sidecar validation RMSE does not beat constant baseline")
    train_sha = canonical_train_replay_sha256
    if train_sha is None:
        train_sha = _sidecar_value(payload, "canonical_train_replay_sha256", "train_replay_sha256",
                                   ("train_replay", "sha256"))
    validation_sha = validation_replay_sha256
    if validation_sha is None:
        validation_sha = _sidecar_value(payload, "validation_replay_sha256",
                                        ("validation_replay", "sha256"))
    errors.extend(_validate_replay_binding(
        payload=payload, label="canonical_train", expected_seeds=TRAIN_SEEDS,
        expected_sha=train_sha, expected_record_count=canonical_train_record_count,
        replay_path=train_replay_path,
    ))
    errors.extend(_validate_replay_binding(
        payload=payload, label="validation", expected_seeds=VALIDATION_SEEDS,
        expected_sha=validation_sha, replay_path=validation_replay_path,
    ))
    return payload, errors

class UAMCTSPrototypeRetriever:
    """Read-only adapter for the explicitly frozen prototype artifact identity."""
    def __init__(self,path):
        self.path=Path(path); self.file_sha256=sha256_file(path); self.payload=json.loads(self.path.read_text(encoding="utf-8"))
        material=dict(self.payload); checksum=material.pop("prototype_sha256",None)
        actual=hashlib.sha256(json.dumps(material,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        if checksum!=actual or self.payload.get("fit_split")!="train" or self.payload.get("radius_selection_split")!="validation":
            raise ValueError("invalid frozen prototype provenance")
    def lookup(self,state,*,agent_name):
        item=self.payload["agents"].get(agent_name); q=np.asarray(state,np.float64)
        if q.shape!=(27,) or not item: raise RuntimeError("prototype miss")
        scale=np.asarray(item["scaler_scale"],np.float64); weights=np.asarray(self.payload["weights"],np.float64)
        ranked=[]
        for row in item["prototypes"]:
            p=np.asarray(row["state"],np.float64); d=float(np.sqrt(((((p-q)/scale)**2)*weights).sum()/weights.sum()))
            ranked.append((d,row["state_sha256"],row["cache_key"],row))
        d,_,_,winner=min(ranked,key=lambda x:(x[0],x[1],x[2]))
        if d>float(self.payload["radius"]): raise RuntimeError(f"nearest prototype distance {d:.9g} exceeds frozen radius (fail closed)")
        return torch.as_tensor(winner["action_prior"],dtype=torch.float32)


def sha256_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_prior_entropy_sidecar(
    path: str | Path, *, entropy_sha256: str, prototype_artifact_sha256: str,
    prototype_sha256: str, validation_replay_sha256: str | None = None,
) -> tuple[dict | None, list[str]]:
    payload, errors = _read_json_mapping(
        _resolve_project_path(path), "validation prior entropy sidecar"
    )
    if payload is None:
        return None, errors
    expected = {
        "schema": PRIOR_ENTROPY_SIDECAR_SCHEMA,
        "prototype_artifact_sha256": prototype_artifact_sha256,
        "prototype_sha256": prototype_sha256,
        "split": "validation",
        "validation_seeds": sorted(VALIDATION_SEEDS),
        "provider_calls": 0,
        "test_seeds_used": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"validation prior entropy sidecar {key} mismatch")
    if payload.get("artifact_sha256", payload.get("prior_entropy_sha256")) != entropy_sha256:
        errors.append("validation prior entropy sidecar artifact SHA256 mismatch")
    if validation_replay_sha256 is not None and payload.get("validation_replay_sha256") != validation_replay_sha256:
        errors.append("validation prior entropy sidecar replay SHA256 mismatch")
    count = payload.get("aligned_unique_records")
    hits, misses = payload.get("hits"), payload.get("misses")
    if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
            or hits != count or misses != 0):
        errors.append("validation prior entropy sidecar coverage is incomplete")
    return payload, errors


def validate_normalizer_sidecar(
    path: str | Path, *, normalizer_sha256: str, world_model_sha256: str,
    reward_model_sha256: str, progress_sha256: str, prior_entropy_sha256: str,
    validation_replay_sha256: str | None = None,
) -> tuple[dict | None, list[str]]:
    payload, errors = _read_json_mapping(
        _resolve_project_path(path), "UAMCTS normalizer sidecar"
    )
    if payload is None:
        return None, errors
    expected = {
        "schema": NORMALIZER_SIDECAR_SCHEMA,
        "world_model_sha256": world_model_sha256,
        "reward_model_sha256": reward_model_sha256,
        "progress_sha256": progress_sha256,
        "prior_entropy_sha256": prior_entropy_sha256,
        "split": "validation",
        "validation_seeds": sorted(VALIDATION_SEEDS),
        "test_updates_allowed": False,
        "test_seeds_used": False,
        "provider_calls": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"UAMCTS normalizer sidecar {key} mismatch")
    if payload.get("artifact_sha256", payload.get("normalizer_sha256")) != normalizer_sha256:
        errors.append("UAMCTS normalizer sidecar artifact SHA256 mismatch")
    if validation_replay_sha256 is not None and payload.get("validation_replay_sha256") != validation_replay_sha256:
        errors.append("UAMCTS normalizer sidecar replay SHA256 mismatch")
    sources = payload.get("uncertainty_sources", payload.get("sources"))
    if not (sources == list(UNCERTAINTY_KEYS)
            or (isinstance(sources, dict) and set(sources) == set(UNCERTAINTY_KEYS))):
        errors.append("UAMCTS normalizer sidecar must bind all three uncertainty sources")
    count = payload.get("validation_aligned_total")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        errors.append("UAMCTS normalizer sidecar validation_aligned_total is invalid")
    return payload, errors


def validate_prototype_chain(
    *, prototype_path: str | Path, coverage_path: str | Path,
    provenance_path: str | Path,
) -> tuple[dict, list[str]]:
    """Bind the prototype file to its coverage and provenance reports."""
    errors: list[str] = []
    artifacts: dict[str, str] = {}
    prototype_file = _resolve_project_path(prototype_path)
    coverage_file = _resolve_project_path(coverage_path)
    provenance_file = _resolve_project_path(provenance_path)
    try:
        prototype = UAMCTSPrototypeRetriever(prototype_file)
        artifacts["prototype_sha256"] = prototype.file_sha256
        artifacts["prototype_payload_sha256"] = str(prototype.payload.get("prototype_sha256"))
    except Exception as exc:
        return artifacts, [f"prototype artifact: {exc}"]
    coverage, coverage_errors = _read_json_mapping(coverage_file, "prototype coverage artifact")
    provenance, provenance_errors = _read_json_mapping(provenance_file, "prototype provenance artifact")
    errors.extend(coverage_errors); errors.extend(provenance_errors)
    if coverage is not None:
        artifacts["prototype_coverage_sha256"] = sha256_file(coverage_file)
        if coverage.get("artifact_file_sha256") != prototype.file_sha256:
            errors.append("prototype coverage does not bind the current prototype file")
        if coverage.get("prototype_sha256") != prototype.payload.get("prototype_sha256"):
            errors.append("prototype coverage payload SHA256 mismatch")
        if coverage.get("online_allowed") is not False or coverage.get("runtime_mode") != "offline_read_only":
            errors.append("prototype coverage is not offline read-only")
        validation = coverage.get("splits", {}).get("validation", {})
        if (not isinstance(validation, dict) or validation.get("misses") != 0
                or validation.get("hits") != validation.get("total")):
            errors.append("prototype validation coverage is incomplete")
    if provenance is not None:
        artifacts["prototype_provenance_sha256"] = sha256_file(provenance_file)
        if provenance.get("schema") != "priorrl_frozen_prototype_provenance_v1":
            errors.append("prototype provenance schema mismatch")
        artifact = provenance.get("artifact")
        if not isinstance(artifact, dict) or artifact.get("file_sha256") != prototype.file_sha256:
            errors.append("prototype provenance does not bind the current prototype file")
        if provenance.get("status") != "PASS" or provenance.get("online_allowed") is not False:
            errors.append("prototype provenance is not PASS/offline")
        if provenance.get("coverage", {}).get("file_sha256") != artifacts.get("prototype_coverage_sha256"):
            errors.append("prototype provenance coverage SHA256 mismatch")
    return artifacts, errors


def _load_rows(path, *, allowed_seeds):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); seed = int(row["episode_seed"])
            if seed not in allowed_seeds or row["agent_name"] not in BLUE_AGENTS:
                raise ValueError("replay violates frozen seed/agent split")
            state = np.asarray(row["state"], np.float32)
            if state.shape != (FORMAL_STATE_DIM,) or not np.isfinite(state).all():
                raise ValueError("progress replay state must be finite D27")
            rows.append(row)
    if not rows: raise ValueError("progress replay is empty")
    return rows


def build_future_progress_dataset(path):
    """Train labels are normalized future response returns within each trajectory."""
    rows = _load_rows(path, allowed_seeds=TRAIN_SEEDS)
    grouped = {}
    for row in rows: grouped.setdefault((int(row["episode_seed"]), row["agent_name"]), []).append(row)
    states, returns = [], []
    for key in sorted(grouped):
        trajectory = sorted(grouped[key], key=lambda x: int(x["decision_index"])); future = 0.0
        local = []
        for row in reversed(trajectory):
            future = float(row["response_reward"]) + (.99 ** int(row["decision_dt"])) * future
            local.append((np.asarray(row["state"], np.float32), future))
        for state, value in reversed(local): states.append(state); returns.append(value)
    values = np.asarray(returns, np.float32); lo, hi = float(values.min()), float(values.max())
    targets = np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)
    return np.stack(states), targets, {"return_min": lo, "return_max": hi, "gamma_tick": .99,
                                      "label": "normalized_future_response_return"}


class ProgressNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(27,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)


def train_progress_ensemble(*, train_replay, output, epochs=3, batch_size=512,
                            device=None, seed=72001):
    states, targets, label = build_future_progress_dataset(train_replay)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = torch.as_tensor(states, device=dev); y = torch.as_tensor(targets, device=dev)
    members = []
    for member in range(ENSEMBLE_SIZE):
        torch.manual_seed(seed + member); model = ProgressNet().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(seed + 100 + member)
        bootstrap = torch.randint(len(x), (len(x),), generator=generator)
        for _ in range(int(epochs)):
            for start in range(0, len(x), int(batch_size)):
                ix = bootstrap[start:start+batch_size].to(dev); loss = ((model(x[ix])-y[ix])**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        members.append({k:v.detach().cpu() for k,v in model.state_dict().items()})
    payload = {"schema":"uamcts_progress_ensemble_v1", "split":"train",
        "allowed_seeds":sorted(TRAIN_SEEDS), "ensemble_size":ENSEMBLE_SIZE, "D":27,
        "architecture":"27-64-1-sigmoid", "bootstrap_seed":seed, "epochs":int(epochs),
        "record_count":len(states), "train_replay_sha256":sha256_file(train_replay),
        "label_contract":label, "members":members}
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,out)
    return {"path":str(out),"sha256":sha256_file(out),"record_count":len(states)}


class FrozenProgressEnsemble:
    def __init__(self, path, *, device="cpu", sidecar_path=None,
                 train_replay_path=None, validation_replay_path=None,
                 require_sidecar=False):
        self.path=Path(path); self.sha256=sha256_file(path); self.device=torch.device(device)
        p=torch.load(path,map_location=self.device,weights_only=False)
        if (p.get("schema")!="uamcts_progress_ensemble_v1" or p.get("split")!="train"
                or p.get("allowed_seeds")!=sorted(TRAIN_SEEDS)
                or p.get("ensemble_size")!=ENSEMBLE_SIZE):
            raise ValueError("invalid train-only progress artifact")
        if require_sidecar and (
                p.get("D") != 27 or not isinstance(p.get("record_count"), int)
                or p.get("record_count") <= 0 or not _is_sha256(p.get("train_replay_sha256"))):
            raise ValueError("progress checkpoint provenance is incomplete")
        contract = p.get("label_contract")
        if (require_sidecar and not isinstance(contract, dict)) or (
                contract is not None and (not isinstance(contract, dict)
                or contract.get("label") != PROGRESS_LABEL or contract.get("gamma_tick") != .99)):
            raise ValueError("invalid progress label contract")
        self.payload = p
        self.sidecar = None
        resolved_sidecar = sidecar_path
        if resolved_sidecar is None and require_sidecar:
            resolved_sidecar = self.path.with_suffix(".sidecar.json")
        if resolved_sidecar is not None:
            sidecar, errors = validate_progress_sidecar(
                resolved_sidecar, progress_sha256=self.sha256,
                canonical_train_replay_sha256=p.get("train_replay_sha256"),
                canonical_train_record_count=p.get("record_count"),
                train_replay_path=train_replay_path,
                validation_replay_path=validation_replay_path,
            )
            if errors:
                raise ValueError("progress sidecar validation failed: " + "; ".join(errors))
            self.sidecar = sidecar
            self.sidecar_path = _resolve_project_path(resolved_sidecar)
            self.sidecar_sha256 = sha256_file(self.sidecar_path)
        self.models=[]
        members = p.get("members")
        if not isinstance(members, list) or len(members) != ENSEMBLE_SIZE:
            raise ValueError("progress ensemble member count mismatch")
        for state in members:
            m=ProgressNet().to(self.device); m.load_state_dict(state); m.eval(); self.models.append(m)
    @torch.no_grad()
    def score(self,state):
        x=torch.as_tensor(np.asarray(state,np.float32),device=self.device)
        if x.shape!=(27,): raise ValueError("progress input must be D27")
        v=torch.stack([m(x) for m in self.models]); return float(v.mean().cpu()),float(v.var(unbiased=False).cpu())


def freeze_uncertainty_normalizers(*, validation_replay, values: dict[str, list[float]],
                                   progress_sha256: str, world_model_sha256: str,
                                   prior_artifact_sha256: str, output,
                                   expected_record_count: int | None = None):
    rows=_load_rows(validation_replay,allowed_seeds=VALIDATION_SEEDS)
    if set(values)!=set(UNCERTAINTY_KEYS): raise ValueError("all three uncertainty sources are required")
    total = int(expected_record_count if expected_record_count is not None else len(rows))
    arrays={key:np.asarray(values[key],np.float64) for key in UNCERTAINTY_KEYS}
    if any(len(x)!=total for x in arrays.values()):
        raise ValueError("uncertainty calibration lengths must match aligned validation records")
    if not np.isfinite(arrays["world_model"]).all() or not np.isfinite(arrays["progress"]).all():
        raise ValueError("WM/progress validation uncertainty must be complete and finite")
    valid=np.isfinite(arrays["prior_entropy"])
    if not valid.any(): raise ValueError("prior entropy has zero validation coverage")
    normalizers={}
    for key in UNCERTAINTY_KEYS:
        x=arrays[key][valid]
        if not np.isfinite(x).all() or np.any(x<0):
            raise ValueError(f"invalid validation-only {key} calibration values")
        normalizers[key]={"mean":float(x.mean()),"std":float(max(x.std(),1e-8)),"count":len(x)}
    payload={"schema":"uamcts_uncertainty_normalizers_v1","split":"validation",
        "allowed_seeds":sorted(VALIDATION_SEEDS),"record_count":int(valid.sum()),
        "validation_aligned_total":total,"prior_coverage_hits":int(valid.sum()),
        "prior_coverage_misses":int((~valid).sum()),
        "validation_replay_sha256":sha256_file(validation_replay),
        "progress_sha256":progress_sha256,"world_model_sha256":world_model_sha256,
        "prior_artifact_sha256":prior_artifact_sha256,"normalizers":normalizers,
        "test_updates_allowed":False,"projection_version":PROJECTION_VERSION,
        "projection_sha256":PROJECTION_SHA256}
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,out)
    return {"path":str(out),"sha256":sha256_file(out),"record_count":int(valid.sum()),
            "aligned_total":total,"prior_misses":int((~valid).sum())}


class FrozenUncertaintyNormalizers:
    def __init__(self,path):
        self.path=Path(path); self.sha256=sha256_file(path); self.payload=torch.load(path,weights_only=False)
        p=self.payload
        if p.get("schema")!="uamcts_uncertainty_normalizers_v1" or p.get("split")!="validation" or p.get("allowed_seeds")!=sorted(VALIDATION_SEEDS) or p.get("test_updates_allowed") is not False:
            raise ValueError("invalid frozen validation normalizers")
        if p.get("projection_sha256") != PROJECTION_SHA256:
            raise ValueError("normalizer semantic projection provenance mismatch")
        if set(p.get("normalizers",{}))!=set(UNCERTAINTY_KEYS): raise ValueError("missing uncertainty normalizer")
    def normalize(self,key,value):
        if key not in UNCERTAINTY_KEYS: raise KeyError(key)
        item=self.payload["normalizers"][key]; return max(0.0,(float(value)-item["mean"])/item["std"])
