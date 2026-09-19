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
    def __init__(self, path, *, device="cpu"):
        self.path=Path(path); self.sha256=sha256_file(path); self.device=torch.device(device)
        p=torch.load(path,map_location=self.device,weights_only=False)
        if p.get("schema")!="uamcts_progress_ensemble_v1" or p.get("split")!="train" or p.get("allowed_seeds")!=sorted(TRAIN_SEEDS) or p.get("ensemble_size")!=5:
            raise ValueError("invalid train-only progress artifact")
        self.models=[]
        for state in p["members"]:
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
