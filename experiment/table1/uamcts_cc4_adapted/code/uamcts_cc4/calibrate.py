"""Validation-only UAMCTS uncertainty calibration; fail closed without all sources."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from formal_experiments.training.bootstrap_world_model import BootstrapProbabilisticWorldModel
from .artifacts import (FrozenProgressEnsemble, VALIDATION_SEEDS, _load_rows,
                        freeze_uncertainty_normalizers, sha256_file, UAMCTSPrototypeRetriever)
from shared.d27_projection import D27ProjectionContext, project_d27_tensor

ROOT=Path(__file__).resolve().parents[2]
def resolve(v): p=Path(v); return p if p.is_absolute() else ROOT/p


def collect(*, validation_replay, progress_artifact, world_model, prior_entropies=None,
            prototype_artifact=None,
            output="outputs/uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.pt",
            report="outputs/uamcts_cc4/calibration/calibration_report.json", device="cpu"):
    validation_replay=resolve(validation_replay); progress_artifact=resolve(progress_artifact)
    world_model=resolve(world_model); rows=_load_rows(validation_replay,allowed_seeds=VALIDATION_SEEDS)
    # Freeze calibration weighting to unique (agent, exact D27) validation states.
    unique=[]; seen=set()
    for row in rows:
        key=(row["agent_name"],np.asarray(row["state"],dtype="<f4").tobytes())
        if key not in seen: seen.add(key); unique.append(row)
    rows=unique
    progress=FrozenProgressEnsemble(progress_artifact,device=device)
    wm=BootstrapProbabilisticWorldModel.load_checkpoint(world_model,device=device)
    states=np.asarray([r["state"] for r in rows],np.float32)
    actions=np.asarray([r["requested_action_id"] for r in rows],np.int64)
    progress_u=[]
    for state in states: progress_u.append(progress.score(state)[1])
    wm_u=[]
    for start in range(0,len(rows),1024):
        pred=wm.predict_ensemble(states[start:start+1024],actions[start:start+1024])
        means=np.asarray(pred["member_means"],np.float32)
        for offset,row in enumerate(rows[start:start+1024]):
            context=D27ProjectionContext.from_root(row["state"],root_tick=int(row["global_tick_start"]),episode_steps=500)
            elapsed=int(row["decision_dt"])
            projected=project_d27_tensor(means[:,offset,:],context=context,elapsed_ticks=elapsed).cpu().numpy()
            wm_u.append(float(np.var(projected,axis=0).mean()))
    partial={"schema":"uamcts_validation_uncertainty_preflight_v1","split":"validation",
        "allowed_seeds":sorted(VALIDATION_SEEDS),"record_count":len(rows),
        "validation_replay_sha256":sha256_file(validation_replay),
        "progress_sha256":progress.sha256,"world_model_sha256":sha256_file(world_model),
        "progress_uncertainty":{"mean":float(np.mean(progress_u)),"std":float(np.std(progress_u))},
        "world_model_uncertainty":{"mean":float(np.mean(wm_u)),"std":float(np.std(wm_u))}}
    report_path=resolve(report); report_path.parent.mkdir(parents=True,exist_ok=True)
    if prototype_artifact is not None:
        prototype_path=resolve(prototype_artifact); retriever=UAMCTSPrototypeRetriever(prototype_path)
        entropy=[]; miss_examples=[]
        for row in rows:
            try:
                p=retriever.lookup(row["state"],agent_name=row["agent_name"]).numpy().astype(np.float64)
                p=p/p.sum(); entropy.append(float(-(p*np.log(np.maximum(p,1e-15))).sum()))
            except RuntimeError as exc:
                entropy.append(float("nan"))
                if len(miss_examples)<20: miss_examples.append({"agent_name":row["agent_name"],"error":str(exc)})
        prior_payload={"schema":"uamcts_validation_prior_entropy_v1","split":"validation",
            "allowed_seeds":sorted(VALIDATION_SEEDS),"aligned_unique_records":len(rows),
            "hits":int(np.isfinite(entropy).sum()),"misses":int((~np.isfinite(entropy)).sum()),
            "prototype_artifact_sha256":sha256_file(prototype_path),
            "prototype_sha256":retriever.payload["prototype_sha256"],"radius":retriever.payload["radius"],
            "entropy":[None if not np.isfinite(x) else x for x in entropy],"miss_examples":miss_examples,
            "provider_calls":0}
        prior_path=resolve("outputs/uamcts_cc4/calibration/validation_prior_entropy.json")
        prior_path.parent.mkdir(parents=True,exist_ok=True)
        prior_path.write_text(json.dumps(prior_payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        prior_entropies=prior_path
    if prior_entropies is None:
        partial.update({"status":"BLOCKED","blocker":"missing frozen offline-prior entropy artifact",
                        "normalizer_bundle_written":False})
        report_path.write_text(json.dumps(partial,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        return partial
    prior_path=resolve(prior_entropies); prior_payload=json.loads(prior_path.read_text(encoding="utf-8"))
    if prior_payload.get("split")!="validation" or prior_payload.get("allowed_seeds")!=sorted(VALIDATION_SEEDS):
        raise ValueError("prior entropy artifact must be frozen validation-only")
    prior_values=[float("nan") if x is None else float(x) for x in prior_payload.get("entropy",[])]
    frozen=freeze_uncertainty_normalizers(validation_replay=validation_replay,
        values={"world_model":wm_u,"progress":progress_u,"prior_entropy":prior_values},
        progress_sha256=progress.sha256,world_model_sha256=sha256_file(world_model),
        prior_artifact_sha256=sha256_file(prior_path),output=resolve(output),
        expected_record_count=len(rows))
    partial.update({"status":"PASS","normalizer_bundle_written":True,"bundle":frozen,
                    "prior_artifact_sha256":sha256_file(prior_path)})
    report_path.write_text(json.dumps(partial,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return partial


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--validation",default="outputs/formal_replay_final_20260917/validation.jsonl")
    p.add_argument("--progress",default="outputs/uamcts_cc4/progress/progress_ensemble_train_only.pt")
    p.add_argument("--world-model",default="outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt")
    p.add_argument("--prior-entropies"); p.add_argument("--prototype-artifact")
    p.add_argument("--output",default="outputs/uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.pt")
    p.add_argument("--report",default="outputs/uamcts_cc4/calibration/calibration_report.json"); p.add_argument("--device",default="cpu")
    a=p.parse_args(argv); result=collect(validation_replay=a.validation,progress_artifact=a.progress,
        world_model=a.world_model,prior_entropies=a.prior_entropies,
        prototype_artifact=a.prototype_artifact,output=a.output,report=a.report,device=a.device)
    print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
