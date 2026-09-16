# Step A4.5a v2 — Corrective Formal Replay Recollection

日期：2026-09-16  
状态：PASS

## Collection

FormalState A4.1c 修正后重新采集：

- train seeds 1000..1031，32 episodes；
- validation seeds 2000..2007，8 episodes；
- episode steps = 100；
- pad_spaces = False；
- calibration/test seeds 未触碰。

Train：

- transitions = 9041；
- requested 0/1/2/3 = 2237 / 2260 / 2276 / 2268；
- executed Sleep/Analyse/Remove/Restore = 6114 / 925 / 984 / 1018；
- valid_target_rate = 0.4301881246；
- fallback_rate = 0.4288242451；
- incident_host_count = 97。

Validation：

- transitions = 2336；
- requested 0/1/2/3 = 577 / 584 / 588 / 587；
- executed Sleep/Analyse/Remove/Restore = 1641 / 214 / 233 / 248；
- valid_target_rate = 0.3951108584；
- fallback_rate = 0.4554794521；
- incident_host_count = 81。

## Full exact mapping audit

Train：

- overall accuracy = 1.0；
- targeted accuracy = 1.0；
- feature=1 but fallback = 0；
- feature=0 but non-fallback = 0；
- nonbinary feature count = 0。

Validation：

- overall accuracy = 1.0；
- targeted accuracy = 1.0；
- feature=1 but fallback = 0；
- feature=0 but non-fallback = 0；
- nonbinary feature count = 0。

## Old-vs-v2 integrity audit

Train：

- transitions = 9041；
- changed state[17] = 103；
- changed next_state[17] = 109；
- all other 26 state dimensions identical；
- all non-state replay fields identical。

Validation：

- transitions = 2336；
- changed state[17] = 30；
- changed next_state[17] = 30；
- all other 26 state dimensions identical；
- all non-state replay fields identical。

This proves A4.1c changed only the intended feature-17 semantics and did not alter the collection trajectory, requested/executed actions, async timing, reward bookkeeping, incident identities, or any other FormalState feature.

FINAL STATUS: PASS

## Next

Run A4.5b again on `outputs/formal_replay_v2/{train,validation}.jsonl` and write to a new v2 world-model output directory. Reapply the original pre-frozen absolute-vs-delta selection rule; do not assume the old absolute selection remains valid.
