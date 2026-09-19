"""Strict UAMCTS real-run artifact gate (read-only)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .artifacts import FrozenProgressEnsemble, FrozenUncertaintyNormalizers, sha256_file

ROOT=Path(__file__).resolve().parents[2]
def resolve(v): p=Path(v); return p if p.is_absolute() else ROOT/p

def run_preflight(*, world_model, reward_model, progress, normalizers, offline_prior_artifact=None):
    errors=[]; artifacts={}
    for key,value in (("world_model",world_model),("reward_model",reward_model)):
        p=resolve(value)
        if not p.is_file(): errors.append(f"missing {key}: {p}")
        else: artifacts[key+"_sha256"]=sha256_file(p)
    try:
        model=FrozenProgressEnsemble(resolve(progress)); artifacts["progress_sha256"]=model.sha256
    except Exception as exc: errors.append(f"progress artifact: {exc}")
    try:
        norm=FrozenUncertaintyNormalizers(resolve(normalizers)); artifacts["normalizers_sha256"]=norm.sha256
        expected={"progress_sha256":artifacts.get("progress_sha256"),
                  "world_model_sha256":artifacts.get("world_model_sha256")}
        for key,value in expected.items():
            if norm.payload.get(key)!=value: errors.append(f"normalizer {key} provenance mismatch")
    except Exception as exc: errors.append(f"normalizer artifact: {exc}")
    if offline_prior_artifact is None:
        errors.append("missing frozen offline-prior coverage/entropy artifact")
    else:
        p=resolve(offline_prior_artifact)
        if not p.is_file(): errors.append(f"missing offline prior artifact: {p}")
        else:
            artifacts["offline_prior_artifact_sha256"]=sha256_file(p)
            if 'norm' in locals() and norm.payload.get("prior_artifact_sha256")!=artifacts["offline_prior_artifact_sha256"]:
                errors.append("normalizer prior entropy provenance mismatch")
    return {"schema":"uamcts_real_gate_v1","eligible":not errors,
            "formal_result_eligible":False,"artifacts":artifacts,"errors":errors}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--world-model",default="outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt")
    p.add_argument("--reward-model",default="outputs/world_model_final_20260917/a4_5c/response_reward_predictor.pt")
    p.add_argument("--progress",default="outputs/uamcts_cc4/progress/progress_ensemble_train_only.pt")
    p.add_argument("--normalizers",default="outputs/uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.pt")
    p.add_argument("--offline-prior-artifact"); p.add_argument("--report",default="outputs/uamcts_cc4/preflight.json")
    a=p.parse_args(argv); result=run_preflight(world_model=a.world_model,reward_model=a.reward_model,
        progress=a.progress,normalizers=a.normalizers,offline_prior_artifact=a.offline_prior_artifact)
    out=resolve(a.report); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["eligible"] else 2
if __name__=="__main__": raise SystemExit(main())
