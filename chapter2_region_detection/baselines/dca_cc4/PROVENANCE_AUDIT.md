# DCA legacy-code provenance audit

Audit date: 2026-09-19. Legacy source was inspected read-only at
`D:\paper\对比\相关代码\第四章区域检测对比文献代码\DCA\DCA`.
The legacy directory was not modified and is not an executable implementation
of the DCA paper. It is a CC4 replay conversion and metrics bridge written for
an earlier experiment protocol.

## File-level provenance

- `convert_apt_to_dca_replay.py` (`sha256:24dc3b72b819de9b23619110c88a3d29ab1e446087eaf25479529f796e6e50a0`):
  converts unavailable `apt_attack_path_*.json` logs into per-region JSONL. It
  derives alert type, severity and risk from the recorded Red action and its
  success flag, tracks Red sessions, and uses target/agent identity. This is a
  data generator, not DCA, and none of its policy inputs are reusable online.
- `dca_replay_to_metricsready.py` (`sha256:cad7bf5f155ca44632934e3924c1fbea199d02d1bff3870dd9043e05f9765c55`):
  a standard-library-only heuristic response simulator. It has no DBSCAN,
  abductive rules, Dyna, value iteration, or learned attack-path model. Its
  action choice directly queries reconstructed `compromised`, `escalated`, and
  `impacted` truth sets. Its custom rewards mutate those sets and are clipped to
  reduce variance. It therefore cannot be used as a Blue policy or W0 metric.
- `eval_metricsready_log_sourceref.py` (`sha256:77c29cfdcf6a1f370d3378c6b423d28e59e9c9868af7042ce7a289acabcb65d3`):
  summarizes the synthetic metrics-ready log, including compromise truth and
  Red last actions. It duplicates evaluator responsibilities and is not reusable
  under W0.
- `data/path_*/dca_replay_region*.jsonl`: 204 generated files across 53 path
  directories (1,297,923 bytes). Rows expose `src_agent`, `raw_action`, Red
  `success`, session-derived fields and the absolute path of the missing source
  replay. They are derived/tainted fixtures, not Blue-visible observations.
- `output/*`: generated plots, JSON and metrics-ready logs. They contain
  reconstructed compromise sets, Red actions and custom rewards and are not
  eligible evidence for the v3 result.

## Reuse decision

Directly reusable:

- No implementation file or generated data artifact.
- The generic idea of deterministic host canonicalization and a finite temporal
  evidence window is compatible in concept only; current code implements these
  independently over real Blue-visible observations.

Must be adapted:

- Legacy `Monitor/Analyse/Remove/Restore` labels map only conceptually to frozen
  A4 IDs. The legacy numeric IDs are `0/1/3/4`; v3 uses contiguous `0/1/2/3`
  through the shared action adapter and target resolver.
- Region/host handling must come from the shared observable target resolver,
  not hostname-prefix rules.
- Event correlation must operate on visible `Processes`, `Connections`, and
  `Files`. Red action names, success and session ownership cannot be substituted
  for alerts.
- Paper attack-path reasoning must remain a separately disclosed CC4 modality
  adaptation. The current finite observable state, Dyna replay and tabular value
  iteration are not sourced from the legacy bridge.

Cannot be used:

- Red action/success/session fields, reconstructed compromise/escalation/impact
  truth, synthetic risk and severity, oracle response selection, custom reward,
  local evaluator, generated replay data, and hard-coded absolute paths.
- The legacy heuristic must not be cited as reproduction of the paper's DBSCAN,
  physical-grid abductive rules, or attack-path planning because it implements
  none of them.

## Safe fidelity improvement adopted

The current runtime now evicts Blue-visible alert tokens older than the frozen
`time_window` before clustering and action selection. Evidence exactly at the
boundary remains eligible. This preserves finite-window temporal correlation
without importing legacy signals, reward, thresholds, data, or truth state.

## Remaining fidelity risks

- The paper's power-grid voltage/frequency/topology and loss-of-load abductive
  rules have no CC4 observable equivalent and remain explicitly unimplemented.
- CC4 evidence types and confidence constants are a disclosed modality adapter,
  not quantities learned or specified by the paper.
- The finite path model learns only within an episode from observable cluster
  transitions; it is not pretrained from the legacy replay corpus.
