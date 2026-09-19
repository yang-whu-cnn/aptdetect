# UAMCTS-CC4 (adapted)

This is a disclosed CC4 modality adaptation of *Uncertainty-Aware Planning
with Generative World- and Language-Models via Monte Carlo Tree Search*, not a
claim of bit-level reproduction. The source paper has no released official
implementation in the experiment taskbook and reports preliminary AI2-THOR
results.

## Preserved core

- Fresh-root MCTS with `N(s)`, `N(s,a)`, `W(s,a)`, `Q(s,a)`, prior `P(a|s)`,
  and separately retained world-model, progress and prior uncertainties.
- Paper hybrid score exactly:
  `Q - rho(phi)*u + c_puct*P*sqrt(N)/(1+Nsa) + c_u(phi)*u`, with
  `rho(phi)=rho_max*phi`, `c_u(phi)=c_u_max*(1-phi)`.
- Potential shaping, adapted for CC4 action duration:
  `r_v3 + beta*(gamma^d*Phi(s') - Phi(s))`.
- Root choice is highest visit count with a deterministic tie-break. Only that
  first A4 action is returned; the search tree is not reused next decision.

## CC4 adaptation

- RGB/VLM progress is replaced by a lightweight ensemble that accepts only a
  finite Blue-visible D27 vector. Its intended label is normalized future
  response return/progress from train replay only. Mean is `Phi`; ensemble
  variance is progress uncertainty.
- AI2-THOR actions become A4 and root availability is supplied by the shared
  action/target layer. Deeper imagined actions remain A4.
- The frozen five-member absolute-state WM supplies imagined D27 transitions
  and ensemble disagreement. The frozen Full-Reward predictor supplies `r_v3`.
- The multimodal online prior becomes a read-only offline prior cache. Cache
  miss fails closed; there is no online LLM fallback.

## Development gate (currently blocked)

The injectable MCTS core and fake-model tests are runnable. A real CC4 pilot or
formal result is **blocked** until all of the following exist and pass preflight:

1. frozen shared WM and frozen Full-Reward artifacts;
2. a train-replay-only D27 progress ensemble with documented target/split;
3. progress and all three uncertainty normalizers frozen from calibration;
4. an offline prior cache/retrieval identity with runtime coverage preflight.

`ProductionGate.require_ready()` enforces this status. No placeholder progress
scores, future truth, test returns, online LLM calls, or test-time model/scaler
updates may be used to clear it. Hyperparameters may be selected only on
calibration/validation; simulation budget candidates are 64 and 128.

Source paper SHA256:
`5dffdc568bf6e7fa04a430c44dcac9d293b588c4a2b5211cc5f14324f10af145`.

## Current development artifact evidence

The train-only five-member D27 progress ensemble is now frozen from seeds
`1000..1031`; its label is duration-aware normalized future response return and
its artifact SHA256 is recorded by the preflight. Validation seeds `2000..2007`
have been used to measure progress variance and frozen-WM disagreement only.
The validation offline-prior entropy artifact now covers all 10,515 unique
validation states and the three-source normalizer bundle is frozen. Artifact
preflight passes. The real 1x20 smoke nevertheless remains `BLOCKED`: the first
real root lookup hits, but an imagined world-model child is distance
`222.932761`, above the frozen validation radius `2.560749706662155`. The
fail-closed prototype contract therefore stops MCTS before tick 1. This must not
be widened with test or smoke states. Test seeds are never accepted by either
fitter.
