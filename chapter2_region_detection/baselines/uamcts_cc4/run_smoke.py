"""Development-only real CC4 UAMCTS 1x20 smoke; fail closed on any prior miss."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
from formal_experiments.data_collection.collect_cc4_formal_replay import encode_decision_state,make_env,observation_success_bool,planner_visible_target_availability
from shared.action_contract import ACTION_CONTRACTS
from shared.cyborg_action_adapter import CybORGActionAdapter
from shared.formal_state import BLUE_AGENTS,FormalStateEncoder,ObservableHostEvidenceTracker
from .runtime import build_runtime
from shared.d27_projection import D27ProjectionContext, PROJECTION_SHA256, PROJECTION_VERSION

ROOT=Path(__file__).resolve().parents[2]
def run(*,seed=3000,device="cuda"):
    planners,priors=build_runtime(world_path=ROOT/"outputs/world_model_final_20260917/a4_5b/world_model_absolute.pt",
        reward_path=ROOT/"outputs/world_model_final_20260917/a4_5c/response_reward_predictor.pt",
        progress_path=ROOT/"outputs/uamcts_cc4/progress/progress_ensemble_train_only.pt",
        prototype_path=ROOT/"outputs/priorrl_cc4/prototypes/frozen_prototypes.json",
        normalizers_path=ROOT/"outputs/uamcts_cc4/calibration/uamcts_uncertainty_normalizers_frozen.pt",device=device)
    env,obs,_=make_env(seed=seed,steps=21,pad_spaces=False); controller=env.env.environment_controller
    adapter=CybORGActionAdapter(); encoder=FormalStateEncoder(); active={a:None for a in BLUE_AGENTS}; trackers={}
    for a in BLUE_AGENTS: trackers[a]=ObservableHostEvidenceTracker(a); trackers[a].reset(obs[a])
    decisions=[]; durations=tuple(int(x.duration_ticks) for x in ACTION_CONTRACTS)
    try:
      for _ in range(20):
        tick=int(controller.step_count); joint={}
        for agent in BLUE_AGENTS:
          if active[agent] is not None: continue
          state,scores,_=encode_decision_state(env=env,encoder=encoder,tracker=trackers[agent],observation=obs[agent],agent_name=agent,global_tick=tick,episode_steps=21)
          _,fam=planner_visible_target_availability(env=env,agent_name=agent,observable_host_scores=scores)
          available=[0]+[i for i,k in enumerate((None,"Analyse","Remove","Restore")) if i and fam[k]]
          context=D27ProjectionContext.from_root(state,root_tick=tick,episode_steps=21)
          started=time.perf_counter(); result=planners[agent].plan(state,available_actions=available,durations=durations,projection_context=context)
          resolution=adapter.resolve(env=env,agent_name=agent,action_id=result.action,observable_host_scores=scores)
          if resolution.fallback: raise RuntimeError("masked root action fell back")
          action=list(env.actions(agent))[resolution.executed_index]; duration=int(action.duration)
          active[agent]={"ready":tick+duration,"family":type(action).__name__,"target":resolution.target_host}
          joint[agent]=resolution.executed_index; decisions.append({"agent":agent,"tick":tick,"action":result.action,"duration":duration,"fallback":False,"latency":time.perf_counter()-started})
        new,_,term,trunc,_=env.step(actions=joint); end=int(controller.step_count)
        for agent in BLUE_AGENTS:
          meta=active[agent]; completed=end==meta["ready"]
          trackers[agent].update(observation=new[agent],global_tick=end,completed_action_family=meta["family"] if completed else None,completed_target_host=meta["target"] if completed else None,completed_action_success=observation_success_bool(new[agent]) if completed else None)
          obs[agent]=new[agent]
          if completed or term.get(agent,False) or trunc.get(agent,False): active[agent]=None
    except RuntimeError as exc:
      report={"status":"BLOCKED","formal_result_eligible":False,"error":str(exc),"ticks_completed":int(controller.step_count),"decisions":decisions,"prior_hits":sum(x.hits for x in priors.values()),"prior_misses":sum(x.misses for x in priors.values()),"projection_version":PROJECTION_VERSION,"projection_sha256":PROJECTION_SHA256}
    else:
      report={"status":"PASS","formal_result_eligible":False,"ticks_completed":20,"decisions":decisions,"prior_hits":sum(x.hits for x in priors.values()),"prior_misses":sum(x.misses for x in priors.values()),"projection_version":PROJECTION_VERSION,"projection_sha256":PROJECTION_SHA256}
    report["report_sha256"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest(); return report
if __name__=="__main__":
    result=run(); out=ROOT/"outputs/uamcts_cc4/smoke/real_1x20.json"
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
